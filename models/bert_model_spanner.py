import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLMHead

# from models.classifier import MultiNonLinearClassifier, SingleLinearClassifier
from allennlp.modules.span_extractors import EndpointSpanExtractor
from torch.nn import functional as F


class BertNER(RobertaPreTrainedModel):

    def __init__(self, config, args):
        super(BertNER, self).__init__(config)
        self.args = args
        self.roberta = XLMRobertaModel(config)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.span_combination_mode = self.args.span_combination_mode
        self.max_span_width = args.max_spanLen
        self.n_class = args.n_class
        self.tokenLen_emb_dim = self.args.tokenLen_emb_dim  # must set, when set a value to the max_span_width.

        print("self.max_span_width: ", self.max_span_width)
        print("self.tokenLen_emb_dim: ", self.tokenLen_emb_dim)

        self._endpoint_span_extractor = EndpointSpanExtractor(config.hidden_size,
                                                              combination=self.span_combination_mode,
                                                              num_width_embeddings=self.max_span_width,
                                                              span_width_embedding_dim=self.tokenLen_emb_dim,
                                                              bucket_widths=True)

        self.spanLen_emb_dim =args.spanLen_emb_dim
        self.morph_emb_dim = args.morph_emb_dim
        input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim
        if self.args.use_spanLen and not self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.spanLen_emb_dim
        elif not self.args.use_spanLen and self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.morph_emb_dim
        elif  self.args.use_spanLen and self.args.use_morph:
            input_dim = config.hidden_size * 2 + self.tokenLen_emb_dim + self.spanLen_emb_dim + self.morph_emb_dim

        self.spanLen_embedding = nn.Embedding(args.max_spanLen + 1, self.spanLen_emb_dim, padding_idx=0)
        self.morph_embedding = nn.Embedding(len(args.morph2idx_list) + 1, self.morph_emb_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(config.model_dropout)
        self.span_classifier = nn.Linear(input_dim, self.n_class)

        if 'q_dim' in args.__dict__:
            self.q_head = nn.Linear(input_dim, args.q_dim)

        self.lm_head = XLMRobertaLMHead(config)
        


    def forward(self, loadall, all_span_lens, all_span_idxs_ltoken, input_ids, token_type_ids=None, attention_mask=None, output_q=False, task_type='ner', mlm_labels=None):
        """
        Args:
            loadall: all input features
            output_q:
                if True: output = all_span_pred(batch, n_span, n_class), normalize of q_emb
                if False: output = all_span_pred
            task_type: 'ner' or 'mlm'
        Returns:
        'ner' task:
            all_span_pred   （pred_logits: (batch, n_span, n_class)）
            normalized_q_reps   (batch, n_span, n_class)    [optional]
            normalized_span_reps    (batch, n_span, n_class)    [optional]
        'mlm' task:
            mlm_logits  (batch, n_tokens, size_vocab)
        """
        bert_outputs = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs[0]  # [batch, seq_len, hidden]  # last_hidden_state
        if task_type == 'ner':
            all_span_rep = self._endpoint_span_extractor(sequence_output, all_span_idxs_ltoken.long()) # [batch, n_span, hidden]
            if not self.args.use_spanLen and not self.args.use_morph:
                pass
            elif self.args.use_spanLen and not self.args.use_morph:
                spanlen_rep = self.spanLen_embedding(all_span_lens) # (bs, n_span, len_dim)
                spanlen_rep = F.relu(spanlen_rep)
                all_span_rep = torch.cat((all_span_rep, spanlen_rep), dim=-1)
            elif not self.args.use_spanLen and self.args.use_morph:
                morph_idxs = loadall[3]
                span_morph_rep = self.morph_embedding(morph_idxs) # (bs, n_span, max_spanLen, dim)
                span_morph_rep = torch.sum(span_morph_rep, dim=2) # (bs, n_span, dim)
                all_span_rep = torch.cat((all_span_rep, span_morph_rep), dim=-1)
            elif self.args.use_spanLen and self.args.use_morph:
                morph_idxs = loadall[3]
                span_morph_rep = self.morph_embedding(morph_idxs) # (bs, n_span, max_spanLen, dim)
                span_morph_rep = torch.sum(span_morph_rep, dim=2) # (bs, n_span, dim)
                spanlen_rep = self.spanLen_embedding(all_span_lens)  # (bs, n_span, len_dim)
                spanlen_rep = F.relu(spanlen_rep)
                all_span_rep = torch.cat((all_span_rep, spanlen_rep, span_morph_rep), dim=-1)

            all_span_rep = self.dropout(all_span_rep)
            all_span_pred = self.span_classifier(all_span_rep)
            
            output = all_span_pred

            if output_q:
                q = self.q_head(all_span_rep)
                output = output, nn.functional.normalize(q, dim=-1)
        else:
            mlm_logits = self.lm_head(sequence_output)
            mlm_logits = mlm_logits.view(-1, self.vocab_size)[(mlm_labels.view(-1)!=-100).bool()]
            output = mlm_logits

        return output
