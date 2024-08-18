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

from dataloaders.dataload import BERTNERDataset, MLMDataset
from dataloaders.collate_functions import collate_to_max_length, collate_to_max_length_for_mlmTask
from models.bert_model_spanner import BertNER
from models.config_spanner import XLMRNerConfig

from random_seed import set_random_seed
from eval_metric import span_f1, span_f1_prune, get_predict, get_predict_prune, get_pruning_predProbs

import logging
logger = logging.getLogger(__name__)

set_random_seed(0)

import pickle

"""
Train a model based on src ner task and tgt mlm task, and generate soft pseudo labels for tgt training set(unlabel set).
"""


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
        self.vocab_size = bert_config.vocab_size

        logging.info(str(args.__dict__ if isinstance(args, argparse.ArgumentParser) else args))

        self.optimizer = args.optimizer
        self.n_class = args.n_class

        self.max_spanLen = args.max_spanLen
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
        self.classifier = torch.nn.Softmax(dim=-1)

        self.fwrite_epoch_res = open(args.fp_epoch_result, 'w')
        self.fwrite_epoch_res.write("f1, recall, precision, correct_pred, total_pred, total_golden\n")

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

        parser.add_argument('--use_spanLen', type=str2bool, default=True, help='use the word-level spanLen as a feature', nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--spanLen_emb_dim", type=int, default=100, help="the embedding dim of a span length")

        parser.add_argument('--use_morph', type=str2bool, default=True, help='use the morphological features of tokens in the span', nargs='?',
                            choices=['yes (default)', True, 'no', False])
        parser.add_argument("--morph_emb_dim", type=int, default=100, help="the embedding dim of the morphology feature.")
        
        parser.add_argument('--morph2idx_list', type=list, help='a list to store a pair of (morph, index).', )
        parser.add_argument('--label2idx_list', type=list, help='a list to store a pair of (label, index).',)

        parser.add_argument('--random_int', type=str, default='0', help='play the role of random seed to name the output dir', )
        
        parser.add_argument('--param_name', type=str, default='param_name', help='a prexfix for a param file name', )
        parser.add_argument('--best_dev_f1', type=float, default=0.0, help='best_dev_f1 value', )
        
        parser.add_argument('--use_prune', type=str2bool, default=True, help='when using the model prediction, whether to handle the conflict prediction results', )

        parser.add_argument("--use_span_weight", type=str2bool, default=True, help="whether to use span weights for the loss.")
        parser.add_argument("--neg_span_weight", type=float,default=0.5, help="range: [0,1.0], the weight of negative span for the loss.")

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
                              betas = (0.9, 0.98),
                              lr = self.args.lr,
                              eps = self.args.adam_epsilon,)
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


    def forward(self, loadall, all_span_lens, all_span_idxs_ltoken, input_ids, attention_mask, token_type_ids, task_type='ner', mlm_labels=None):
        return self.model(loadall, all_span_lens, all_span_idxs_ltoken, input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, task_type=task_type, mlm_labels=mlm_labels)


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
        if mode == 'train' and self.args.use_span_weight:
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


    def training_step(self, batch, batch_idx):
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'])

        # src ner task
        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights,
                   real_span_mask_ltoken, words, all_span_word, all_span_idxs]
        attention_mask = (tokens != 0).long()
        attention_mask[:, 0] = 1
        all_span_logits = self.forward(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids, task_type='ner')
        predict_probs = self.classifier(all_span_logits)

        if self.args.use_prune: # default=True
            span_f1s, new_pred_label_idx = span_f1_prune(all_span_idxs, predict_probs, span_label_ltoken, real_span_mask_ltoken)
        else:
            span_f1s = span_f1(predict_probs, span_label_ltoken, real_span_mask_ltoken)

        output = {}
        output["span_f1s"] = span_f1s
        loss_ner = self.compute_loss(loadall, all_span_logits, span_label_ltoken, real_span_mask_ltoken, mode='train')
        
        # tgt mlm task
        tokens, token_type_ids, attention_mask, mlm_labels = [t.to(self.device) for t in next(self.mlm_loader)]
        all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = None, None, None, None, None, None, None, None, None 
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights,
                   real_span_mask_ltoken, words, all_span_word, all_span_idxs]
        mlm_logits = self.forward(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids, task_type='mlm', mlm_labels=mlm_labels)

        mlm_loss = self.compute_mlm_loss(mlm_logits, mlm_labels)


        self.log('train_ner_loss', loss_ner.item())
        self.log('train_mlm_loss', mlm_loss.item())

        w = 0.001
        if 'genpseudo_es' in self.args.default_root_dir:
            w = 1e-5
        loss = loss_ner + mlm_loss*w

        output[f"train_ner_loss"] = loss_ner
        output['loss'] = loss

        self.training_step_outputs.append(output)
        return output


    def on_train_epoch_end(self):
        print("\nuse... on_train_epoch_end: ", )
        outputs = self.training_step_outputs
        all_counts = torch.stack([x[f'span_f1s'] for x in outputs]).sum(0)
        correct_pred, total_pred, total_golden = all_counts
        precision = correct_pred / (total_pred + 1e-10)
        recall = correct_pred / (total_golden + 1e-10)
        f1 = precision * recall * 2 / (precision + recall + 1e-10)
        self.fwrite_epoch_res.write( "train: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))
        
        self.training_step_outputs.clear()  # free memory


    def validation_step(self, batch, batch_idx, mode='val'):
        tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs = batch
        loadall = [tokens, token_type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, words, all_span_word, all_span_idxs]

        attention_mask = (tokens != 0).long()
        attention_mask[:, 0] = 1
        all_span_logits = self.forward(loadall, all_span_lens, all_span_idxs_ltoken, tokens, attention_mask, token_type_ids)
        predict_probs = self.classifier(all_span_logits)

        if self.args.use_prune:     # default=True
            span_f1s, new_pred_label_idx = span_f1_prune(all_span_idxs, predict_probs, span_label_ltoken, real_span_mask_ltoken)
            batch_preds = get_predict_prune(self.args, all_span_word, words, new_pred_label_idx, span_label_ltoken, all_span_idxs)
        else:
            span_f1s = span_f1(predict_probs, span_label_ltoken, real_span_mask_ltoken)
            batch_preds = get_predict(self.args, all_span_word, words, predict_probs, span_label_ltoken, all_span_idxs)
        
        loss = self.compute_loss(loadall, all_span_logits, span_label_ltoken, real_span_mask_ltoken, mode='test/dev')

        output = {}
        output["span_f1s"] = span_f1s
        output["batch_preds"] = batch_preds
        output[f"val_loss"] = loss

        pred_label_idx = torch.max(predict_probs, dim=-1)[1]
        predict_probs = get_pruning_predProbs(pred_label_idx, all_span_idxs, predict_probs).to(self.device)  

        real_span_predict_probs = [p[m == 1].tolist() for p, m in zip(predict_probs, real_span_mask_ltoken)]
        output["predicts"] = real_span_predict_probs
        
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

        if self.current_epoch == (self.args.max_epochs-1):
            # save the prediction to text file ('sentence span::true_label::pred_label')
            pred_batch_results = [x['batch_preds'] for x in outputs]
            fp_write = self.args.default_root_dir +  '/' + self.args.modelName + '_tgtUnlabel.txt'  # the dev set is tgt unlabel training set
            fwrite = open(fp_write, 'w')
            for pred_batch_result in pred_batch_results:
                for pred_result in pred_batch_result:
                    fwrite.write(pred_result + '\n')
            self.args.best_dev_f1 = f1

            # save the pseudo probs of the real spans
            # only consider the real spans
            all_realSpan_predictProbs = [y for x in outputs for y in x['predicts']]

            file_prob1 = self.args.default_root_dir + '/' + self.args.modelName + '_prob_unlabel_ep{:02d}.pkl'.format(self.current_epoch)
            print("the file path of pseudo probs: ", file_prob1)
            fwrite_prob = open(file_prob1, 'wb')
            pickle.dump(all_realSpan_predictProbs, fwrite_prob)

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

        # save the prediction to text file ('sentence span::true_label::pred_label')
        pred_batch_results = [x['batch_preds'] for x in outputs]
        fp_write = self.args.default_root_dir + '/'+self.args.modelName + '_test.txt'
        fwrite = open(fp_write, 'w')
        for pred_batch_result in pred_batch_results:
            for pred_result in pred_batch_result:
                fwrite.write(pred_result + '\n')

        self.fwrite_epoch_res.write(
            "test: %f, %f, %f, %d, %d, %d\n" % (f1, recall, precision, correct_pred, total_pred, total_golden))

        # save the pseudo label probs of the real spans
        all_realSpan_predictProbs = [list(x['predicts']) for x in outputs]
        all_span_words = [list(x['all_span_word']) for x in outputs]

        label2idx = {}
        label2idx_list = self.args.label2idx_list
        for labidx in label2idx_list:
            lab, idx = labidx
            label2idx[lab] = int(idx)

        file_prob1 = self.args.default_root_dir + '/'+self.args.modelName + '_prob_test.pkl'
        print("the file path of probs: ", file_prob1)
        fwrite_prob = open(file_prob1, 'wb')
        pickle.dump([label2idx, all_realSpan_predictProbs, all_span_words], fwrite_prob)

        self.test_step_outputs.clear()

    # prepare the mlm_loader for the mlm task
    def prepare_data(self):
        prefix = 'unlabel'
        json_path = os.path.join(self.data_dir, f"spanner.{prefix}")
        dataset = MLMDataset(json_path=json_path, 
                             tokenizer=Tokenizer.from_file(os.path.join(self.args.bert_config_dir,'tokenizer.json')),
                             mask_rate = self.args.mask_rate)

        self.mlm_loader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate_to_max_length_for_mlmTask
        )
        self.mlm_loader = repeat_dataloader(self.mlm_loader)


    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train")


    def val_dataloader(self):
        val_data = self.get_dataloader("unlabel")
        return val_data

    def test_dataloader(self):
        return self.get_dataloader("test")

    def get_dataloader(self, prefix="train") -> DataLoader:
        json_path = os.path.join(self.data_dir, f"spanner.{prefix}")
        print("json_path: ", json_path)

        dataset = BERTNERDataset(self.args, json_path=json_path,
                                 tokenizer=Tokenizer.from_file(os.path.join(self.args.bert_config_dir,'tokenizer.json')),
                                 max_length=self.args.bert_max_length,
                                 pad_to_maxlen=False,
                                 )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True if prefix == "train" else False,
            drop_last=False,
            collate_fn=collate_to_max_length
        )
        return dataloader


def repeat_dataloader(iterable):
    while True:
        for x in iterable:
            yield x


def main():

    parser = BertNerTagger.get_parser()

    args = parser.parse_args()

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

    args.default_root_dir = args.default_root_dir + '_' + args.random_int

    if not os.path.exists(args.default_root_dir):
        os.makedirs(args.default_root_dir)

    fp_epoch_result = args.default_root_dir + '/epoch_results.txt'
    args.fp_epoch_result = fp_epoch_result

    args_text = '\n'.join([hp for hp in str(args).replace('Namespace(', '').replace(')', '').split(', ')])
    fn_path = args.default_root_dir + '/' + args.param_name + '.txt'
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
