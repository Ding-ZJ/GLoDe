# encoding: utf-8


import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim import SGD
import torch.nn.functional as F

from dataloaders.dataload import BERTNERDataset, BERTNERDataset_U
from dataloaders.collate_functions import collate_to_max_length, collate_to_max_length_U
from models.bert_model_spanner import BertNER
from models.config_spanner import XLMRNerConfig
from random_seed import set_random_seed
from eval_metric import span_f1,span_f1_prune,get_predict,get_predict_prune, get_pruning_predIdxs

import logging
logger = logging.getLogger(__name__)

import pickle
import numpy as np
import faiss



class BertNerTagger(pl.LightningModule):

    def __init__(self, args: argparse.Namespace):

        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        self.bert_dir = args.bert_config_dir
        self.data_dir = self.args.data_dir

        bert_config = XLMRNerConfig.from_pretrained(args.bert_config_dir,
                                                         hidden_dropout_prob=args.bert_dropout,
                                                         attention_probs_dropout_prob=args.bert_dropout,
                                                         model_dropout=args.model_dropout)
        self.vocab_size = bert_config.vocab_size
        
        self.model = BertNER.from_pretrained(args.bert_config_dir, config=bert_config, args=self.args)
        
        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

        self.optimizer = args.optimizer
        self.n_class = args.n_class

        self.max_spanLen = args.max_spanLen
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.classifier = torch.nn.Softmax(dim=-1)

        self.fwrite_epoch_res = open(args.fp_epoch_result, 'w')
        self.fwrite_epoch_res.write("f1, recall, precision, correct_pred, total_pred, total_golden\n")
        
        self.register_buffer('prototypes', torch.zeros(self.args.n_class, args.q_dim))
        self.register_buffer('proto_margin', torch.zeros(self.args.n_class))

        with open(self.args.load_soft, 'rb') as f:
            init_confidence_list = pickle.load(f)   # soft pseudo labels generated from source model
        self.num_span_upperb = self.args.max_spanLen * self.args.bert_max_length
        init_confidence = torch.tensor([[sent + [[1] + [0]*(self.args.n_class-1)]*(self.num_span_upperb-len(sent))] for sent in init_confidence_list]).squeeze(1)

        self.loss_fn_ul = partial_loss(init_confidence, postprocess_pseudo=self.args.postprocess_pseudo)

        self.faiss_index = None
        self.faiss_labels = None
        self.register_buffer('neigh_margin', torch.zeros(self.args.n_class))

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []


    @staticmethod
    def get_parser():
        def str2bool(v):
            if v.lower() in ('yes', 'true', 't', 'y', '1'):
                return True
            elif v.lower() in ('no', 'false', 'f', 'n', '0'):
                return False
            else:
                raise argparse.ArgumentTypeError('Boolean value expected.')

        parser = argparse.ArgumentParser(description="Training")

        parser.add_argument("--data_dir", type=str, required=True, help="json data dir")
        parser.add_argument("--bert_config_dir", type=str, required=True, help="PLMs path")
        parser.add_argument("--pretrained_checkpoint", default="", type=str, help="pretrained checkpoint path")
        parser.add_argument("--bert_max_length", type=int, default=128, help="max sequence length")
        parser.add_argument("--batch_size", type=int, default=10, help="batch size of dataloader")
        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--warmup_steps", default=50, type=int, help="warmup steps used for scheduler.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

        parser.add_argument("--model_dropout", type=float, default=0.2, help="dropout rate for the two layer non-linear classifier of the BertNER model")
        parser.add_argument("--bert_dropout", type=float, default=0.2, help="dropout rate for the PLMs")
        parser.add_argument("--final_div_factor", type=float, default=1e4, help="final div factor of linear decay scheduler")
        parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw", help="optimizer type")

        parser.add_argument("--gpus", type=str)
        parser.add_argument("--workers", type=int, default=0, help="num workers for dataloader")

        parser.add_argument("--dataname", default="conll03", help="the name of a dataset")
        parser.add_argument("--max_spanLen", type=int, default=4, help="max span length")

        parser.add_argument("--n_class", type=int, default=5, help="the classes of a task")
        parser.add_argument("--modelName",  default='test', help="the classes of a task")

        parser.add_argument('--use_tokenLen', type=str2bool, default=True, help='use the token-level spanLen as a feature', nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--tokenLen_emb_dim", type=int, default=50, help="the token-level spanLen embedding dim of a span")
        parser.add_argument('--span_combination_mode', default='x,y', help='combination mode of the start/end token reps of the span')

        parser.add_argument('--use_spanLen', type=str2bool, default=False, help='use the word-level spanLen as a feature', nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="the embedding dim of a span length")

        parser.add_argument('--use_morph', type=str2bool, default=True, help='use the morphological features of tokens in the span', nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--morph_emb_dim", type=int, default=100, help="the embedding dim of the morphology feature.")
        
        parser.add_argument('--morph2idx_list', type=list, help='a list to store a pair of (morph, index).', )
        parser.add_argument('--label2idx_list', type=list, help='a list to store a pair of (label, index).',)

        parser.add_argument('--param_name', type=str, default='param_name', help='a prexfix for a param file name', )
        parser.add_argument('--best_dev_f1', type=float, default=0.0, help='best_dev_f1 value', )

        parser.add_argument('--use_prune', type=str2bool, default=True, help='when using the model prediction, whether to handle the conflict prediction results', )

        parser.add_argument("--use_span_weight", type=str2bool, default=True, help="whether to use span weights for the loss.")
        parser.add_argument("--neg_span_weight", type=float,default=0.5, help="range: [0,1.0], the weight of negative span for the loss.")
        
        parser.add_argument("--seed", type=int, default=0, help='random seed')

        parser.add_argument('--do_mlm', action='store_true', help = 'Whether to do mlm task. No usage here, just used in generate_pseudo.py')

        parser.add_argument('--postprocess_prot', action='store_true', help = 'whether to process the confilict in the prototype_based logits when update pseudo probs')
        parser.add_argument('--postprocess_neigh', action='store_true', help = 'whether to process the confilict in the neigh_based logits when update pseudo probs')
        parser.add_argument('--postprocess_pseudo', action='store_true', help = 'Whether to postprocess overlapping pseudo labels of the pseudo probs when self-training')

        parser.add_argument("--load_soft", type=str, help='the file path of the init pseudo soft labels for the tgt training set')

        parser.add_argument("--q_dim", type=int, default=128, help='embedding dim of the compressed span rep')
        
        parser.add_argument('--cl_downsample', default=1.0, type=float, help='downsample rate of non-entity in contrastive learning')
        parser.add_argument('--postprocess_cl', action='store_true', help = 'Whether to postprocess overlapping spans for contrastive learning')

        parser.add_argument('--update_soft_start', default=1, type=int, help = 'epoch for starting Prototype Updating')
        parser.add_argument('--proto_m', default=0.99, type=float, help='momentum for computing the momving average of prototypes')
        
        
        parser.add_argument('--conf_ema_range', default='0.95,0.80', type=str, help='pseudo target updating coefficient (phi)')
        parser.add_argument('--mask_rate', type=float, default=0.2, help='mask rate for mlm task')
        
        # args for Trainer
        parser.add_argument('--max_epochs', type=int, default=10)
        parser.add_argument('--precision', type=int, default=32, help='training precision')
        parser.add_argument('--val_check_interval', type=float, default=1.0, help='check after a fraction of the training epoch')
        parser.add_argument('--accumulate_grad_batches', type=int, default=1)
        parser.add_argument('--default_root_dir', type=str, default='train_logs')
        parser.add_argument('--gradient_clip_val', type=float, default=1.0)
        
        return parser


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.optimizer == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters,
                              betas=(0.9, 0.98),
                              lr=self.args.lr,
                              eps=self.args.adam_epsilon,)
        else:
            optimizer = SGD(optimizer_grouped_parameters, lr=self.args.lr, momentum=0.9)
        num_gpus = len([x for x in str(self.args.gpus).split(",") if x.strip()])
        t_total = (len(self.train_dataloader()) // (self.args.accumulate_grad_batches * num_gpus) + 1) * self.args.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.args.lr, pct_start=float(self.args.warmup_steps/t_total),
            final_div_factor=self.args.final_div_factor,
            total_steps=t_total, anneal_strategy='linear'
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, loadall, all_span_lens, all_span_idxs_ltoken, input_ids, attention_mask, token_type_ids, output_q=False, task_type='ner', mlm_labels=None):
        return self.model(loadall, all_span_lens, all_span_idxs_ltoken, input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_q=output_q, task_type=task_type, mlm_labels=mlm_labels)

    def compute_loss(self, loadall, all_span_logits, span_label_ltoken, real_span_mask_ltoken, mode):
        '''
        :param loadall: all input features
        :param all_span_pred: shape: (bs, n_span, n_class) logits
        '''
        batch_size, n_span = span_label_ltoken.size()
        all_span_logits1 = all_span_logits.view(-1, self.n_class)
        span_label_ltoken1 = span_label_ltoken.view(-1)
        loss = self.cross_entropy(all_span_logits1, span_label_ltoken1)
        loss = loss.view(batch_size, n_span)
        if mode=='train' and self.args.use_span_weight:
            span_weight = loadall[6]
            loss = loss * span_weight
        loss = torch.masked_select(loss, real_span_mask_ltoken.bool())
        loss= torch.mean(loss)
        return loss

    def compute_mlm_loss(self, mlm_logits, mlm_labels):
        loss_fct = torch.nn.CrossEntropyLoss()
        logits = mlm_logits
        labels = mlm_labels.view(-1)
        labels = labels[(labels!=-100).bool()]
        if len(labels) == 0:
            mlm_loss = torch.tensor(0.).to(mlm_labels.device)
        else:
            mlm_loss = loss_fct(logits, labels)
            
        return mlm_loss


    def on_train_epoch_start(self):
        self.loss_fn_ul.set_conf_ema_m(self.current_epoch, self.args)
       
    def training_step(self, batch, batch_idx):
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch     
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights,
                   real_span_mask_ltoken, words, all_span_word, all_span_idxs]
        attention_mask = (tokens != 0).long()
        attention_mask[:, 0] = 1
        all_span_logits, q = self.forward(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids, output_q=True)
        predict_probs = self.classifier(all_span_logits)

        if self.args.use_prune:
            span_f1s, _ = span_f1_prune(all_span_idxs, predict_probs, span_label_ltoken, real_span_mask_ltoken)
        else:
            span_f1s = span_f1(predict_probs, span_label_ltoken, real_span_mask_ltoken)

        output = {}
        output["span_f1s"] = span_f1s
        ner_loss = self.compute_loss(loadall, all_span_logits, span_label_ltoken, real_span_mask_ltoken, mode='train')
        
        src_real_span_q_reps = q.detach()[real_span_mask_ltoken==1]
        src_real_span_labels = span_label_ltoken.detach()[real_span_mask_ltoken==1]
        # downsample non entity spans
        sample_rate = 0.03
        downsample_mask = torch.rand(src_real_span_labels.shape).to(self.device) < sample_rate
        downsample_mask = torch.logical_or(downsample_mask, src_real_span_labels != 0)
        src_real_span_q_reps = src_real_span_q_reps[downsample_mask]
        src_real_span_labels = src_real_span_labels[downsample_mask]

        output['src_q_reps'], output['src_q_labels'] = src_real_span_q_reps, src_real_span_labels

        # Update prototypes with source language labeled data
        for feats, true_labels, masks in zip(q.detach(), span_label_ltoken, real_span_mask_ltoken):
            for feat, true_label, mask in zip(feats, true_labels, masks):
                if mask == 1: # Excluding padding spans
                    self.prototypes[true_label] = self.prototypes[true_label]*self.args.proto_m + (1-self.args.proto_m)*feat
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # Unlabeled data
        ul_tokens, ul_token_type_ids, ul_all_span_idxs_ltoken, ul_morph_idxs, ul_span_label_ltoken, \
        ul_all_span_lens, ul_all_span_weights, ul_real_span_mask_ltoken, ul_words, \
        ul_all_span_word, ul_all_span_idxs, unlabel_guid, \
        ul_tokens_mlm, ul_attention_mlm, ul_mlm_labels = [t.to(self.device) if torch.is_tensor(t) else t for t in next(self.unlabel_loader)]      
        ul_loadall = [ul_tokens, ul_token_type_ids, ul_all_span_idxs_ltoken, ul_morph_idxs, ul_span_label_ltoken, ul_all_span_lens,
                      ul_all_span_weights, ul_real_span_mask_ltoken, ul_words, ul_all_span_word, ul_all_span_idxs]
        ul_attention_mask = (ul_tokens != 0).long()
        ul_attention_mask[:, 0] = 1
        ul_all_span_logits, ul_q = self.forward(ul_loadall, ul_all_span_lens, ul_all_span_idxs_ltoken, ul_tokens, ul_attention_mask, ul_token_type_ids, output_q=True)
        ul_predict_probs = self.classifier(ul_all_span_logits)
        
        tgt_pseudo_labels = torch.max(self.loss_fn_ul.confidence[unlabel_guid, :ul_predict_probs.shape[1]], dim=-1)[1]
        tgt_real_span_pseudo_labels = tgt_pseudo_labels[ul_real_span_mask_ltoken.bool()]
        output['unlabel_pseudo_label'] = tgt_real_span_pseudo_labels

        tgt_real_span_q_reps = ul_q.detach()[ul_real_span_mask_ltoken==1]
        tgt_real_span_labels = tgt_real_span_pseudo_labels
        
        downsample_mask = torch.rand(tgt_real_span_labels.shape).to(self.device) < sample_rate
        downsample_mask = torch.logical_or(downsample_mask, tgt_real_span_labels != 0)
        tgt_real_span_q_reps = tgt_real_span_q_reps[downsample_mask]
        tgt_real_span_labels = tgt_real_span_labels[downsample_mask]

        output['tgt_q_reps'], output['tgt_q_labels'] = tgt_real_span_q_reps, tgt_real_span_labels
        
        # update prototypes with target language pseudo labels
        for feats, pred_labels, masks in zip(ul_q.detach(), tgt_pseudo_labels, ul_real_span_mask_ltoken):
            for feat, pred_label, mask in zip(feats, pred_labels, masks):
                if mask == 1: # Excluding padding spans
                    self.prototypes[pred_label] = self.prototypes[pred_label]*self.args.proto_m + (1-self.args.proto_m)*feat  
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

        # compute prototypical based logits # (for target samples)
        prototypes_detached = self.prototypes.clone().detach()
        logits_prot = torch.matmul(ul_q.detach(), prototypes_detached.t())
        probs_prot = torch.softmax(logits_prot, dim=-1)
        output['logits_prot'] = logits_prot[ul_real_span_mask_ltoken.bool()]
        output['probs_prot'] = probs_prot[ul_real_span_mask_ltoken.bool()]
            
        # Update soft pseudo labels(based on prototypes)
        if self.current_epoch > self.args.update_soft_start:
            score_pad = torch.zeros([probs_prot.shape[0], self.num_span_upperb-probs_prot.shape[1], probs_prot.shape[-1]]).to(self.device)  # bs × pad_num × n_class
            score_pad[:,:,0] = 1.0 # default padding to 'O'
            
            classwise_proto_margin = self.proto_margin           

            # class specific update rates
            update_rate_proto = torch.tensor([1.0 for _ in range(self.args.n_class)]).to(self.device)
            update_rate_proto = update_rate_proto[None, None, :].expand(probs_prot.shape[0], probs_prot.shape[1], -1)
            update_rate_proto = torch.cat((update_rate_proto, score_pad), dim=1)
            
            score_prot_mask = (logits_prot > classwise_proto_margin).float()
            
            _, prot_labels = logits_prot.max(dim=-1)
            tmp = F.one_hot(prot_labels, logits_prot.shape[-1]).float()
            score_prot_mask[:, :, 0] = torch.logical_or(score_prot_mask[:, :, 0], tmp[:, :, 0]).float()
 
            score_prot_mask = F.normalize(score_prot_mask, p=1.0, dim=-1)

            update_rate_proto = torch.cat((score_prot_mask, score_pad), dim=1) # Do not update out of margin spans 
                
            if self.args.postprocess_prot:
                score_prot_idx = torch.max(probs_prot, dim=-1)[1]
                _, _, score_prot_idx_new = get_pruning_predIdxs(score_prot_idx, ul_all_span_idxs, probs_prot.tolist())
                score_prot_mask = torch.logical_or(score_prot_idx_new.to(torch.device('cuda')) != 0, score_prot_idx == 0).unsqueeze(-1).repeat(1, 1, self.args.n_class)
                update_rate_proto = update_rate_proto * torch.cat((score_prot_mask.long(), score_pad), dim=1)
                    
            probs_prot = torch.cat((probs_prot, score_pad), dim=1)

        # compute neighbor based logits # (for target samples)
        search_num = 300
        if self.current_epoch >= self.args.update_soft_start:
            shape = ul_q.shape
            queries = ul_q.detach()[ul_real_span_mask_ltoken==1].cpu().view(-1, self.args.q_dim).numpy()
            D, I = self.faiss_index.search(queries, search_num)
            tmp_neigh_idxs = torch.tensor(I).to(self.device)
            tmp_neigh_labels = self.faiss_labels[tmp_neigh_idxs]
            logits_neigh_tmp = torch.zeros(queries.shape[0], self.args.n_class).to(self.device)
            for class_id in range(self.args.n_class):   
                class_num = torch.sum((tmp_neigh_labels == class_id), dim=-1)
                logits_neigh_tmp[:, class_id] = class_num

            logits_neigh = torch.zeros(shape[0], shape[1], self.args.n_class).to(self.device)
            real_span_nums = torch.sum((ul_real_span_mask_ltoken == 1), dim=-1)
            s_idx = 0
            for i, num in enumerate(real_span_nums):
                logits_neigh[i][0: num] = logits_neigh_tmp[s_idx: s_idx+num]
                s_idx += num
            probs_neigh = F.normalize(logits_neigh, p=1)
            output['logits_neigh'] = logits_neigh[ul_real_span_mask_ltoken.bool()]
            
        # Update soft pseudo labels(based on neighbors)
        if self.current_epoch > self.args.update_soft_start:
            score_pad = torch.zeros([logits_neigh.shape[0], self.num_span_upperb-logits_neigh.shape[1], logits_neigh.shape[-1]]).to(self.device)
            score_pad[:,:,0] = 1.0 # default padding to 'O'
            
            classwise_neigh_margin = self.neigh_margin * 0.9  # test

            # class specific update rates
            update_rate_neigh = torch.tensor([1.0 for _ in range(self.args.n_class)]).to(self.device)
            update_rate_neigh = update_rate_neigh[None, None, :].expand(logits_neigh.shape[0], logits_neigh.shape[1], -1)
            update_rate_neigh = torch.cat((update_rate_neigh, score_pad), dim=1)  
                   
            score_neigh_mask = (logits_neigh > classwise_neigh_margin).float()

            _, neigh_labels = logits_neigh.max(dim=-1)
            tmp = F.one_hot(neigh_labels, logits_neigh.shape[-1]).float()
            score_neigh_mask[:, :, 0] = torch.logical_or(score_neigh_mask[:, :, 0], tmp[:, :, 0]).float()

            score_neigh_mask = F.normalize(score_neigh_mask, p=1.0, dim=-1)

            update_rate_neigh = torch.cat((score_neigh_mask, score_pad), dim=1) # Do not update out of margin spans
                
            if self.args.postprocess_neigh:
                score_neigh_idx = torch.max(probs_neigh, dim=-1)[1]
                _, _, score_neigh_idx_new = get_pruning_predIdxs(score_neigh_idx, ul_all_span_idxs, probs_neigh.tolist())
                score_neigh_mask = torch.logical_or(score_neigh_idx_new.to(torch.device('cuda')) != 0, score_neigh_idx == 0).unsqueeze(-1).repeat(1, 1, self.args.n_class)
                update_rate = update_rate * torch.cat((score_neigh_mask.long(), score_pad), dim=1)
                    
            probs_neigh = torch.cat((probs_neigh, score_pad), dim=1)

            # choice 1(default)
            update_rate = (update_rate_proto + update_rate_neigh)/2
            update_rate = F.normalize(update_rate, p=1.0, dim=-1)
            # # choice 2 → more sharp
            # temp = 0.5
            # update_rate = update_rate_proto + update_rate_neigh
            # mask = (update_rate!=0).long()
            # update_rate = F.softmax(update_rate/temp, dim=-1) * mask
            # update_rate = F.normalize(update_rate, p=1.0, dim=-1)
            
            self.loss_fn_ul.confidence_update(shape=probs_neigh.shape, batch_index=unlabel_guid, update_rate=update_rate)

        # tgt ner loss
        ul_all_span_weights = torch.ones_like(ul_all_span_weights)      # weights for all spans are set to 1, due to the inaccessible of gold labels
        loss_ul = self.loss_fn_ul(ul_predict_probs, unlabel_guid, ul_real_span_mask_ltoken.bool(), ul_all_span_weights, ul_all_span_idxs)
        
        # mlm loss
        ul_mlm_logits = self.forward(ul_loadall, ul_all_span_lens, ul_all_span_idxs_ltoken, ul_tokens_mlm, ul_attention_mlm, ul_token_type_ids, task_type='mlm', mlm_labels=ul_mlm_labels)
        mlm_loss = self.compute_mlm_loss(ul_mlm_logits, ul_mlm_labels)

        w = 0.001
        if 'conll03_es' in self.args.default_root_dir:
            w = 1e-5
        loss = ner_loss + loss_ul + mlm_loss*w
        
        self.log('train_ner_loss', ner_loss.item())
        self.log('train_selfT_loss', loss_ul.item())
        self.log('train_mlm_loss', mlm_loss.item())
        
        output[f"train_loss"] = loss
        output['loss'] = loss

        self.log('train_loss', loss.item())
        

        skip_overlap_pseudo = True
        if skip_overlap_pseudo:
            ul_span_f1s,ul_pred_label_idx = span_f1_prune(ul_all_span_idxs, 
                                                          self.loss_fn_ul.confidence[unlabel_guid,:ul_predict_probs.shape[1]], 
                                                          ul_span_label_ltoken, 
                                                          ul_real_span_mask_ltoken)
            ul_batch_preds = get_predict_prune(self.args, ul_all_span_word, ul_words, 
                                               ul_pred_label_idx, ul_span_label_ltoken, ul_all_span_idxs)
        else:
            ul_span_f1s = span_f1(self.loss_fn_ul.confidence[unlabel_guid,:ul_predict_probs.shape[1]], ul_span_label_ltoken, ul_real_span_mask_ltoken)
            ul_batch_preds = get_predict(self.args, ul_all_span_word, ul_words, 
                                         self.loss_fn_ul.confidence[unlabel_guid,:ul_predict_probs.shape[1]], 
                                         ul_span_label_ltoken, ul_all_span_idxs)
        output['ul_pseudo_span_f1s'] = ul_span_f1s
        output['ul_pseudo_batch_preds'] = ul_batch_preds   
        
        self.training_step_outputs.append(output)

        return output


    def on_train_epoch_end(self):
        print("\nuse... training_epoch_end: ", )
        outputs = self.training_step_outputs
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        precision =correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)
        self.fwrite_epoch_res.write("train: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        
        if self.current_epoch >= self.args.update_soft_start:
            epoch_logits_prot = torch.cat([x[f'logits_prot'] for x in outputs], dim=0).cpu().detach().numpy()
            unlabel_pred_label = torch.cat([x[f'unlabel_pseudo_label'] for x in outputs], dim=0).cpu().detach().numpy()
                
            # updating thresholds
            total_sim = [[] for _ in range(self.args.n_class)]
            for i in range(len(total_sim)):
                total_sim[i].append(np.array([0. for _ in range(self.args.n_class)]))
            for i in range(epoch_logits_prot.shape[0]):
                label = unlabel_pred_label[i]
                total_sim[label].append(epoch_logits_prot[i])

            proto_margin = []
            for j in range(self.args.n_class):
                cls_sim = np.stack(total_sim[j], axis=0)
                proto_margin.append(np.mean(cls_sim, axis=0)[j])
            self.proto_margin = torch.FloatTensor(proto_margin).to(self.device)


        all_q_reps = torch.cat([x['src_q_reps'] for x in outputs]).cpu().detach().numpy()
        all_q_labels = torch.cat([x['src_q_labels'] for x in outputs]).cpu().detach().numpy()
        all_q_reps_t = torch.cat([x['tgt_q_reps'] for x in outputs]).cpu().detach().numpy()
        all_q_labels_t = torch.cat([x['tgt_q_labels'] for x in outputs]).cpu().detach().numpy()
        all_q_reps = np.append(all_q_reps, all_q_reps_t, axis=0)
        all_q_labels = np.append(all_q_labels, all_q_labels_t, axis=0)
        
        faiss_index = faiss.IndexFlatIP(self.args.q_dim)
        faiss_index = faiss.index_cpu_to_all_gpus(faiss_index)
        faiss_index.add(all_q_reps)
        self.faiss_index = faiss_index
        self.faiss_labels = torch.tensor(all_q_labels).to(self.device)
        
        if self.current_epoch >= self.args.update_soft_start:
            epoch_logits_neigh = torch.cat([x[f'logits_neigh'] for x in outputs], dim=0).cpu().detach().numpy()
            total_sim = [[] for _ in range(self.args.n_class)]
            for i in range(len(total_sim)):
                total_sim[i].append(np.array([0 for _ in range(self.args.n_class)]))
            for i in range(epoch_logits_neigh.shape[0]):
                label = unlabel_pred_label[i]
                total_sim[label].append(epoch_logits_neigh[i])
            neigh_margin = []
            for j in range(self.args.n_class):
                cls_sim = np.stack(total_sim[j], axis=0)
                neigh_margin.append(np.mean(cls_sim, axis=0)[j])
            self.neigh_margin = torch.FloatTensor(neigh_margin).to(self.device)

        self.training_step_outputs.clear()


    def validation_step(self, batch, batch_idx, mode='val'):
        output = {}

        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]
        attention_mask = (tokens != 0).long()
        attention_mask[:, 0] = 1
        all_span_logits = self.forward(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
        predict_probs = self.classifier(all_span_logits)

        span_f1s = span_f1(predict_probs, span_label_ltoken, real_span_mask_ltoken)
        span_f1s_prune, pred_label_idx = span_f1_prune(all_span_idxs, predict_probs, span_label_ltoken, real_span_mask_ltoken)
        batch_preds = get_predict_prune(self.args, all_span_word, words, pred_label_idx, span_label_ltoken, all_span_idxs)

        loss = self.compute_loss(loadall, all_span_logits, span_label_ltoken, real_span_mask_ltoken, mode='test/dev')

        output["span_f1s"] = span_f1s
        output["span_f1s_prune"] = span_f1s_prune
        output["batch_preds"] =batch_preds
        output[f"val_loss"] = loss
        output["predicts"] = predict_probs
        output['all_span_word'] = all_span_word

        if mode == 'val':
            self.validation_step_outputs.append(output)
        elif mode == 'test':
            self.test_step_outputs.append(output)
        else:
            raise ValueError('unexpected mode, choose from "val" and "test"! ')
        return output

    def on_validation_epoch_end(self):
        print("\nuse... validation_epoch_end: ", )
        outputs = self.validation_step_outputs
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        precision =correct_pred / (total_pred+1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)
        self.log('val_span_f1', f1)
        self.fwrite_epoch_res.write("dev: %f, %f, %f, %d, %d, %d\n"%(f1, recall, precision, correct_pred, total_pred, total_golden) )

        self.validation_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, mode='test')

    def on_test_epoch_end(self):
        print("\nuse... test_epoch_end: ",)
        outputs = self.test_step_outputs

        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        precision = correct_pred / (total_pred + 1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)
        self.log('test_span_f1', f1)

        all_counts_prune = torch.stack([x[f'span_f1s_prune'] for x in outputs]).sum(0)
        correct_pred_p, total_pred_p, total_golden_p = all_counts_prune
        precision_p = correct_pred_p / (total_pred_p + 1e-10)
        recall_p = correct_pred_p / (total_golden_p + 1e-10)
        f1_p = precision_p * recall_p * 2 / (precision_p + recall_p + 1e-10)
        self.log('test_span_f1_prune', f1_p)

        pred_batch_results = [x['batch_preds'] for x in outputs]
        fp_write = self.args.default_root_dir + '/' + self.args.modelName + '_test.txt'
        fwrite = open(fp_write, 'w')
        for pred_batch_result in pred_batch_results:
            for pred_result in pred_batch_result:
                fwrite.write(pred_result+'\n')

        self.fwrite_epoch_res.write(
            "test: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        
        self.fwrite_epoch_res.write(
            "test_prune: %f, %f, %f, %d, %d, %d\n" % (f1_p, recall_p, precision_p, correct_pred_p, total_pred_p, total_golden_p))

        all_predicts = [list(x['predicts'].cpu()) for x in outputs]
        all_span_words = [list(x['all_span_word']) for x in outputs]

        label2idx = {}
        label2idx_list = self.args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)

        file_prob1 = self.args.default_root_dir + '/' + self.args.modelName + '_prob_test.pkl'
        print("the file path of probs: ", file_prob1)
        fwrite_prob = open(file_prob1, 'wb')
        pickle.dump([label2idx, all_predicts, all_span_words], fwrite_prob)

        self.test_step_outputs.clear()


    def prepare_data(self):
        # Create repeated loader for unlabeled data
        self.unlabel_loader = self.get_dataloader("unlabel")
        self.unlabel_loader = repeat_dataloader(self.unlabel_loader)
        
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")

    def val_dataloader(self):
        val_data = self.get_dataloader("dev")
        return val_data

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train") -> DataLoader:
        json_path = os.path.join(self.data_dir, f"spanner.{prefix}")
        print("json_path: ", json_path)
        print("use Tokenizer from pretrained XLMR")
        return_guid = True if prefix == 'unlabel' else False
        if prefix == 'unlabel':
            dataset = BERTNERDataset_U(self.args, json_path=json_path,
                                    tokenizer=Tokenizer.from_file(os.path.join(self.args.bert_config_dir,'tokenizer.json')),
                                    max_length=self.args.bert_max_length,
                                    pad_to_maxlen=False,
                                    mask_rate=self.args.mask_rate) # ※
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.args.batch_size,
                                    shuffle=True if (prefix == "train" or prefix == "unlabel") else False,
                                    drop_last=False,
                                    collate_fn=collate_to_max_length_U)
        else:
            dataset = BERTNERDataset(self.args, json_path=json_path,
                                    tokenizer=Tokenizer.from_file(os.path.join(self.args.bert_config_dir,'tokenizer.json')),
                                    max_length=self.args.bert_max_length,
                                    pad_to_maxlen=False,
                                    return_guid=return_guid)
            dataloader = DataLoader(dataset=dataset,
                                    batch_size=self.args.batch_size,
                                    shuffle=True if (prefix == "train" or prefix == "unlabel") else False,
                                    drop_last=False,
                                    collate_fn=collate_to_max_length)

        return dataloader
    

def repeat_dataloader(iterable):
    while True:
        for x in iterable:
            yield x


class partial_loss(torch.nn.Module):
    def __init__(self, confidence, conf_ema_m=0.99, postprocess_pseudo=False):
        super().__init__()
        self.confidence = confidence.to(torch.device('cuda'))
        self.init_conf = confidence.detach()
        self.conf_ema_m = conf_ema_m
        self.postprocess_pseudo = postprocess_pseudo
        if self.postprocess_pseudo:
            print("Will postprocess pseudo labels by filtering overlapping spans...")

    def set_conf_ema_m(self, epoch, args):
        # Set/Update tgt pseudo labels' updating coefficient
        start = args.conf_ema_range[0]
        end = args.conf_ema_range[1]
        self.conf_ema_m = 1. * (epoch - args.update_soft_start) / (args.max_epochs - args.update_soft_start) * (end - start) + start    

    def forward(self, outputs, index, pad_mask, weights, all_span_idxs=None):
        # ul_logits（bs × n_span × n_class）, unlabel_guid, ul_real_span_mask_ltoken.bool(), ul_all_span_weights, ul_all_span_idxs
        postprocess_mask = torch.ones_like(pad_mask, dtype=bool)
        if self.postprocess_pseudo:
            pseudos = self.confidence[index, :outputs.shape[1]]
            pseudo_label_idx = torch.max(pseudos, dim=-1)[1]
            span_probs = pseudos.tolist()
            _, _, pseudo_label_idx_new = get_pruning_predIdxs(pseudo_label_idx, all_span_idxs, span_probs)
            postprocess_mask = torch.logical_or(pseudo_label_idx_new.to(torch.device('cuda')) != 0, pseudo_label_idx == 0)
            filtered_span_mask = torch.logical_and(pseudo_label_idx!=0, pseudo_label_idx_new.to(torch.device('cuda')) == 0)
            filtered_logsm_outputs = F.log_softmax(outputs[filtered_span_mask * pad_mask], dim=1)
            filtered_outputs = filtered_logsm_outputs[:,0] # treat as one-hot O spans

        logsm_outputs = F.log_softmax(outputs[pad_mask * postprocess_mask], dim=1)
        final_outputs = logsm_outputs * self.confidence[index, :outputs.shape[1]][pad_mask * postprocess_mask]

        if self.postprocess_pseudo and filtered_outputs.nelement() > 0:
            average_loss = - torch.cat((final_outputs.sum(dim=1) * weights[pad_mask * postprocess_mask], filtered_outputs), dim=0).mean()
        else:
            average_loss = - (final_outputs.sum(dim=1) * weights[pad_mask * postprocess_mask]).mean()
        
        return average_loss

    def confidence_update_proto(self, temp_un_conf, batch_index, update_rate):
        with torch.no_grad():
            pseudo_label = torch.ones_like(temp_un_conf).cuda()
            pseudo_label = pseudo_label * update_rate
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
            self.confidence[batch_index, :] = F.normalize(self.confidence[batch_index, :], p=1.0, dim=-1)
            
        return None

    def confidence_update_neigh(self, temp_un_conf, batch_index, update_rate):
        with torch.no_grad():
            pseudo_label = torch.ones_like(temp_un_conf).cuda()
            pseudo_label = pseudo_label * update_rate
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
            self.confidence[batch_index, :] = F.normalize(self.confidence[batch_index, :], p=1.0, dim=-1)
            
        return None

    def confidence_update(self, shape, batch_index, update_rate):
        with torch.no_grad():
            pseudo_label = torch.ones(shape).cuda()
            # pseudo update rate
            pseudo_label = pseudo_label * update_rate
            self.confidence[batch_index, :] = self.conf_ema_m * self.confidence[batch_index, :] + (1 - self.conf_ema_m) * pseudo_label
            self.confidence[batch_index, :] = F.normalize(self.confidence[batch_index, :], p=1.0, dim=-1)



def main():

    parser = BertNerTagger.get_parser()

    args = parser.parse_args()
    args.conf_ema_range = [float(item) for item in args.conf_ema_range.split(',')]
    
    set_random_seed(args.seed)

    label2idx = {}
    if 'conll' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3, "MISC": 4}
    elif 'wikiann' in args.dataname:
        label2idx = {"O": 0, "ORG": 1, "PER": 2, "LOC": 3}

    label2idx_list = []
    for lab, idx in label2idx.items():
        pair = (lab, idx)
        label2idx_list.append(pair)
    args.label2idx_list = label2idx_list

    morph2idx_list = []
    morph2idx = {'isupper': 1, 'islower': 2, 'istitle': 3, 'isdigit': 4, 'other': 5}
    for morph, idx in morph2idx.items():
        pair = (morph, idx)
        morph2idx_list.append(pair)
    args.morph2idx_list = morph2idx_list

    args.default_root_dir = args.default_root_dir + '/run' + str(args.seed)

    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    fp_epoch_result = args.default_root_dir + '/epoch_results.txt'
    args.fp_epoch_result = fp_epoch_result

    args_text = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
    fn_path = args.default_root_dir + '/' +args.param_name + '.txt'
    if fn_path is not None:
        with open(fn_path, mode='w') as text_file:
            text_file.write(args_text)

    model = BertNerTagger(args)
    if args.pretrained_checkpoint:
        model.load_state_dict(torch.load(args.pretrained_checkpoint, map_location=torch.device('cpu'))["state_dict"])

    trainer = Trainer(max_epochs=args.max_epochs, precision=args.precision, val_check_interval=args.val_check_interval,
                      accumulate_grad_batches=args.accumulate_grad_batches, default_root_dir=args.default_root_dir, gradient_clip_val=args.gradient_clip_val)

    trainer.fit(model)
    trainer.test()


if __name__ == '__main__':
    main()
