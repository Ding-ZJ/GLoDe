# encoding: utf-8

import json
import torch
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer, Tokenizer
from torch.utils.data import Dataset
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans
from torch.nn import CrossEntropyLoss
import numpy as np


class MLMDataset(Dataset):
	"""
	Args:
		json_path: path of spanner style data
		tokenizer:
		max_length: int, max length of sequence
	"""

	def __init__(self, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, mask_rate: float = 0.25):
		self.all_data = json.load(open(json_path, encoding="utf-8"))
		self.tokenzier = tokenizer
		self.max_length = max_length
		self.mask_rate = mask_rate

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, item):
		cls_tok = "<s>"
		sep_tok = "</s>"
		mask_tok = '<mask>'

		data = self.all_data[item]
		tokenizer = self.tokenzier

		context = data["context"].strip()
		if '\u200b' in context:
			context = context.replace('\u200b', '')
		elif '\ufeff' in context:
			context = context.replace('\ufeff', '')
		elif '  ' in context:
			context = context.replace('  ', ' ')
		elif '\u200e' in context:
			context = context.replace('\u200e', '') 
		elif '\u200c' in context:
			context = context.replace('\u200c', '') 
		elif '\u200d' in context:
			context = context.replace('\u200d', '')
		elif '\u200f' in context:
			context = context.replace('\u200f', '')

		context_tokens = tokenizer.encode(context, add_special_tokens=True)
		token_ids = context_tokens.ids
		type_ids = context_tokens.type_ids
		token_ids = token_ids[:self.max_length]
		type_ids = type_ids[:self.max_length]

		# make sure last token is sep_tok
		sep_token = tokenizer.token_to_id(sep_tok)
		if token_ids[-1] != sep_token:
			assert len(token_ids) == self.max_length
			token_ids = token_ids[:-1] + [sep_token]

		# mask process
		cls_tok_id, sep_tok_id = tokenizer.token_to_id(cls_tok), tokenizer.token_to_id(sep_tok)
		# avoid to mask some special tokens
		no_mask_tok_ids = [cls_tok_id, sep_tok_id, 6]

		mask_tok_id = tokenizer.token_to_id(mask_tok)
		sample_mask = torch.rand(len(token_ids)) < self.mask_rate
		
		attention_mask, mlm_labels = [], []
		loss_ignore_idx = CrossEntropyLoss().ignore_index
		for i, flag in enumerate(sample_mask):
			if flag and (token_ids[i] not in no_mask_tok_ids):
				mlm_labels.append(token_ids[i])
				attention_mask.append(0)
				token_ids[i] = mask_tok_id
			else:
				mlm_labels.append(loss_ignore_idx)
				attention_mask.append(1)

		attention_mask = attention_mask[:self.max_length]
		mlm_labels = mlm_labels[:self.max_length]


		token_ids = torch.LongTensor(token_ids)
		type_ids = torch.LongTensor(type_ids)
		attention_mask = torch.LongTensor(attention_mask)
		mlm_labels = torch.LongTensor(mlm_labels)

		output = [token_ids, type_ids, attention_mask, mlm_labels]
		
		return output


class BERTNERDataset(Dataset):
	"""
	Args:
		return_guid: whether to return sample id
	"""

	def __init__(self, args, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, pad_to_maxlen=False, return_guid=False):
		self.args = args
		self.all_data = json.load(open(json_path, encoding="utf-8"))
		self.tokenzier = tokenizer
		self.max_length = max_length
		self.pad_to_maxlen = pad_to_maxlen
		self.return_guid = return_guid

		self.max_spanLen = self.args.max_spanLen
		minus = int((self.max_spanLen + 1) * self.max_spanLen / 2)
		self.max_num_span = self.max_length * self.max_spanLen - minus	
		self.dataname = self.args.dataname
		self.spancase2idx_dic = {}

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, item):
		cls_tok = "<s>"
		sep_tok = "</s>"

		# get the label2idx dictionary
		label2idx = {}
		label2idx_list = self.args.label2idx_list
		for labidx in label2idx_list:
			lab, idx = labidx
			label2idx[lab] = int(idx)

		# get the morph2idx dictionary
		morph2idx = {}
		morph2idx_list = self.args.morph2idx_list
		for morphidx in morph2idx_list:
			morph, idx = morphidx
			morph2idx[morph] = int(idx)

		data = self.all_data[item]
		tokenizer = self.tokenzier

		context = data["context"].strip()
		if '\u200b' in context:
			context = context.replace('\u200b', '')
		elif '\ufeff' in context:
			context = context.replace('\ufeff', '')
		elif '  ' in context:
			context = context.replace('  ', ' ')
		elif '\u200e' in context:
			context = context.replace('\u200e', '') 
		elif '\u200c' in context:
			context = context.replace('\u200c', '') 
		elif '\u200d' in context:
			context = context.replace('\u200d', '')
		elif '\u200f' in context:
			context = context.replace('\u200f', '')

		span_idxLab = data["span_posLabel"]

		sidxs = []
		eidxs = []
		for seidx, label in span_idxLab.items():
			sidx, eidx = seidx.split(';')
			sidxs.append(int(sidx))
			eidxs.append(int(eidx))

		# add space offsets
		words = context.split()

		# convert the span position into the character index
		pos_span_idxs = []
		for sidx, eidx in zip(sidxs, eidxs):
			pos_span_idxs.append((sidx, eidx))

		all_span_idxs = enumerate_spans(context.split(), offset=0, max_span_width=self.args.max_spanLen)

		# compute the span weight
		all_span_weights = []
		for span_idx in all_span_idxs:
			weight = self.args.neg_span_weight
			if span_idx in pos_span_idxs:
				weight = 1.0
			all_span_weights.append(weight)

		all_span_lens = []
		for idxs in all_span_idxs:
			sid, eid = idxs
			slen = eid - sid + 1
			all_span_lens.append(slen)

		# morphological feature idxs
		morph_idxs = self.case_feature_tokenLevel(morph2idx, all_span_idxs, words, self.args.max_spanLen)

		context_tokens = tokenizer.encode(context, add_special_tokens=True)
		token_ids = context_tokens.ids
		type_ids = context_tokens.type_ids
		offsets = context_tokens.offsets

		# all_span_idsx_ltoken: token-level span idxs information
		# all_span_idxs_new_label: token-level span label information
		all_span_idxs_ltoken, all_span_word, all_span_idxs_new_label, select_mask = self.convert2tokenIdx(words, token_ids, type_ids, offsets, all_span_idxs, span_idxLab)
		span_label_ltoken = []
		for seidx_str, label in all_span_idxs_new_label.items():
			span_label_ltoken.append(label2idx[label])
		
		if sum(select_mask) != len(select_mask):	
			# the length of some sequences exceeds the specified threshold after tokenizing, thus the span containing out-of-length tokens should to be deleted
			span_label_ltoken = [e for e,i in zip(span_label_ltoken, select_mask) if i==1]
			all_span_lens = [e for e,i in zip(all_span_lens, select_mask) if i==1]
			all_span_weights = [e for e,i in zip(all_span_weights, select_mask) if i==1]
			morph_idxs = [e for e,i in zip(morph_idxs, select_mask) if i==1]
			words = [e for e,i in zip(words, select_mask) if i==1]
			all_span_idxs = [e for e,i in zip(all_span_idxs, select_mask) if i==1]

		token_ids = token_ids[:self.max_length]
		type_ids = type_ids[:self.max_length]
		all_span_idxs_ltoken = all_span_idxs_ltoken[:self.max_num_span]
		span_label_ltoken = span_label_ltoken[:self.max_num_span]
		all_span_lens = all_span_lens[:self.max_num_span]
		morph_idxs = morph_idxs[:self.max_num_span]
		all_span_weights = all_span_weights[:self.max_num_span]

		real_span_mask_ltoken = np.ones_like(span_label_ltoken)

		# make sure last token is sep_tok
		sep_token = tokenizer.token_to_id(sep_tok)
		if token_ids[-1] != sep_token:
			assert len(token_ids) == self.max_length
			token_ids = token_ids[:-1] + [sep_token]

		if self.pad_to_maxlen:		# default is false
			token_ids = self.pad(token_ids, 0)
			type_ids = self.pad(type_ids, 1)
			all_span_idxs_ltoken = self.pad(all_span_idxs_ltoken, value=(0, 0), max_length=self.max_num_span)
			real_span_mask_ltoken = self.pad(real_span_mask_ltoken, value=0, max_length=self.max_num_span)
			span_label_ltoken = self.pad(span_label_ltoken, value=0, max_length=self.max_num_span)
			all_span_lens = self.pad(all_span_lens, value=0, max_length=self.max_num_span)
			morph_idxs = self.pad(morph_idxs, value=0, max_length=self.max_num_span)
			all_span_weights = self.pad(all_span_weights, value=0, max_length=self.max_num_span)

		token_ids = torch.LongTensor(token_ids)
		type_ids = torch.LongTensor(type_ids)  # use to split the first and second sentence.
		all_span_idxs_ltoken = torch.LongTensor(all_span_idxs_ltoken)
		real_span_mask_ltoken = torch.LongTensor(real_span_mask_ltoken)
		span_label_ltoken = torch.LongTensor(span_label_ltoken)
		all_span_lens = torch.LongTensor(all_span_lens)
		morph_idxs = torch.LongTensor(morph_idxs)
		all_span_weights = torch.Tensor(all_span_weights)

		output = [token_ids, type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, 
		   			words, all_span_word, all_span_idxs]
		
		if self.return_guid:
			output.append(item)
		
		return output
		

	# Get the case, number and other morphological features of tokens in the span.
	def case_feature_tokenLevel(self, morph2idx, span_idxs, words, max_spanlen):
		caseidxs = []
		for idxs in span_idxs:
			sid, eid = idxs
			span_word = words[sid:eid + 1]
			caseidx1 = [0 for _ in range(max_spanlen)]
			for j, token in enumerate(span_word):
				tfeat = ''
				if token.isupper():
					tfeat = 'isupper'
				elif token.islower():
					tfeat = 'islower'
				elif token.istitle():
					tfeat = 'istitle'
				elif token.isdigit():
					tfeat = 'isdigit'
				else:
					tfeat = 'other'
				caseidx1[j] =morph2idx[tfeat]
			caseidxs.append(caseidx1)
		return caseidxs


	def pad(self, lst, value=None, max_length=None):
		max_length = max_length or self.max_length
		while len(lst) < max_length:
			lst.append(value)
		return lst


	# convert the word-level span information to token-level information
	def convert2tokenIdx(self, words, tokens, type_ids, offsets, span_idxs, span_idxLab):
		max_length = self.max_length

		sidxs = [x1 + sum([len(w) for w in words[:x1]]) for (x1, x2) in span_idxs]
		eidxs = [x2 + sum([len(w) for w in words[:x2 + 1]]) for (x1, x2) in span_idxs]

		span_idxs_new_label = {}
		for ns, ne, ose in zip(sidxs, eidxs, span_idxs):
			os, oe = ose
			oes_str = "{};{}".format(os, oe)	
			nes_str = "{};{}".format(ns, ne)	
			if oes_str in span_idxLab:
				label = span_idxLab[oes_str]
				span_idxs_new_label[nes_str] = label
			else:
				span_idxs_new_label[nes_str] = 'O'

		origin_offset2token_sidx = {}
		origin_offset2token_eidx = {}
		for token_idx in range(len(tokens)):
			token_start, token_end = offsets[token_idx]

			# skip cls_tok or sep_tok
			if token_start == token_end == 0:
				continue
			origin_offset2token_sidx[token_start] = token_idx
			origin_offset2token_eidx[token_end] = token_idx

		# convert the position from character-level to token-level.
		span_new_sidxs = []
		span_new_eidxs = []
		n_span_keep = 0        

		select_mask = []
		for start, end in zip(sidxs, eidxs):
			try:
				temp = (origin_offset2token_eidx[end], origin_offset2token_sidx[start])
			except KeyError:
				print('words', words)
				print('tokens', tokens)
				print('offsets', offsets)
				print('span_idxs', span_idxs)
				print('sidxs', sidxs)
				print('eidxs', eidxs) 
				print('origin_offset2token_eidx', len(origin_offset2token_eidx), origin_offset2token_eidx)
				print('origin_offset2token_sidx', len(origin_offset2token_sidx), origin_offset2token_sidx)

			if origin_offset2token_eidx[end] > max_length - 1 or origin_offset2token_sidx[start] > max_length - 1:
				select_mask.append(0)
				continue
			select_mask.append(1)
			span_new_sidxs.append(origin_offset2token_sidx[start])
			span_new_eidxs.append(origin_offset2token_eidx[end])
			n_span_keep += 1

		all_span_word = []
		for (sidx, eidx) in span_idxs:
			all_span_word.append(words[sidx:eidx + 1])
		all_span_word = [w for w,i in zip(all_span_word, select_mask) if i==1]

		span_idxs_ltoken = []
		for sidx, eidx in zip(span_new_sidxs, span_new_eidxs):
			span_idxs_ltoken.append((sidx, eidx))

		return span_idxs_ltoken, all_span_word, span_idxs_new_label, select_mask


class BERTNERDataset_U(Dataset):

	def __init__(self, args, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, pad_to_maxlen=False, mask_rate: float=0.15):

		self.args = args
		self.all_data = json.load(open(json_path, encoding="utf-8"))
		self.tokenzier = tokenizer
		self.max_length = max_length
		self.pad_to_maxlen = pad_to_maxlen

		self.max_spanLen = self.args.max_spanLen
		minus = int((self.max_spanLen + 1) * self.max_spanLen / 2)
		self.max_num_span = self.max_length * self.max_spanLen - minus	
		self.dataname = self.args.dataname
		self.spancase2idx_dic = {}

		self.mask_rate = mask_rate

	def __len__(self):
		return len(self.all_data)

	def __getitem__(self, item):
		cls_tok = "<s>"
		sep_tok = "</s>"
		mask_tok = '<mask>'

		label2idx = {}
		label2idx_list = self.args.label2idx_list
		for labidx in label2idx_list:
			lab, idx = labidx
			label2idx[lab] = int(idx)

		morph2idx = {}
		morph2idx_list = self.args.morph2idx_list
		for morphidx in morph2idx_list:
			morph, idx = morphidx
			morph2idx[morph] = int(idx)

		data = self.all_data[item]
		tokenizer = self.tokenzier

		context = data["context"].strip()
		if '\u200b' in context:
			context = context.replace('\u200b', '')
		elif '\ufeff' in context:
			context = context.replace('\ufeff', '')
		elif '  ' in context:
			context = context.replace('  ', ' ')
		elif '\u200e' in context:
			context = context.replace('\u200e', '') 
		elif '\u200c' in context:
			context = context.replace('\u200c', '') 
		elif '\u200d' in context:
			context = context.replace('\u200d', '')
		elif '\u200f' in context:
			context = context.replace('\u200f', '')

		span_idxLab = data["span_posLabel"]

		sidxs = []
		eidxs = []
		for seidx, label in span_idxLab.items():
			sidx, eidx = seidx.split(';')
			sidxs.append(int(sidx))
			eidxs.append(int(eidx))

		# add space offsets
		words = context.split()

		# convert the span position into the character index, space is also a position.
		pos_span_idxs = []
		for sidx, eidx in zip(sidxs, eidxs):
			pos_span_idxs.append((sidx, eidx))

		all_span_idxs = enumerate_spans(context.split(), offset=0, max_span_width=self.args.max_spanLen)

		# compute the span weight
		all_span_weights = []
		for span_idx in all_span_idxs:
			weight = self.args.neg_span_weight
			if span_idx in pos_span_idxs:
				weight = 1.0
			all_span_weights.append(weight)

		all_span_lens = []
		for idxs in all_span_idxs:
			sid, eid = idxs
			slen = eid - sid + 1
			all_span_lens.append(slen)

		morph_idxs = self.case_feature_tokenLevel(morph2idx, all_span_idxs, words, self.args.max_spanLen)

		context_tokens = tokenizer.encode(context, add_special_tokens=True)
		token_ids = context_tokens.ids
		type_ids = context_tokens.type_ids
		offsets = context_tokens.offsets

		all_span_idxs_ltoken, all_span_word, all_span_idxs_new_label, select_mask = self.convert2tokenIdx(words, token_ids, type_ids, offsets, all_span_idxs, span_idxLab)
		span_label_ltoken = []
		for seidx_str, label in all_span_idxs_new_label.items():
			span_label_ltoken.append(label2idx[label])
		
		if sum(select_mask) != len(select_mask):	
			span_label_ltoken = [e for e,i in zip(span_label_ltoken, select_mask) if i==1]
			all_span_lens = [e for e,i in zip(all_span_lens, select_mask) if i==1]
			all_span_weights = [e for e,i in zip(all_span_weights, select_mask) if i==1]
			morph_idxs = [e for e,i in zip(morph_idxs, select_mask) if i==1]
			words = [e for e,i in zip(words, select_mask) if i==1]
			all_span_idxs = [e for e,i in zip(all_span_idxs, select_mask) if i==1]

		token_ids = token_ids[:self.max_length]
		type_ids = type_ids[:self.max_length]
		all_span_idxs_ltoken = all_span_idxs_ltoken[:self.max_num_span]
		span_label_ltoken = span_label_ltoken[:self.max_num_span]
		all_span_lens = all_span_lens[:self.max_num_span]	# *word level span length
		morph_idxs = morph_idxs[:self.max_num_span]
		all_span_weights = all_span_weights[:self.max_num_span]

		real_span_mask_ltoken = np.ones_like(span_label_ltoken)

		sep_token = tokenizer.token_to_id(sep_tok)
		if token_ids[-1] != sep_token:
			assert len(token_ids) == self.max_length
			token_ids = token_ids[:-1] + [sep_token]

		if self.pad_to_maxlen:		# default to false
			token_ids = self.pad(token_ids, 0)
			type_ids = self.pad(type_ids, 1)
			all_span_idxs_ltoken = self.pad(all_span_idxs_ltoken, value=(0, 0), max_length=self.max_num_span)
			real_span_mask_ltoken = self.pad(real_span_mask_ltoken, value=0, max_length=self.max_num_span)
			span_label_ltoken = self.pad(span_label_ltoken, value=0, max_length=self.max_num_span)
			all_span_lens = self.pad(all_span_lens, value=0, max_length=self.max_num_span)
			morph_idxs = self.pad(morph_idxs, value=0, max_length=self.max_num_span)
			all_span_weights = self.pad(all_span_weights, value=0, max_length=self.max_num_span)

		token_ids = torch.LongTensor(token_ids)
		type_ids = torch.LongTensor(type_ids)  # use to split the first and second sentence.
		all_span_idxs_ltoken = torch.LongTensor(all_span_idxs_ltoken)
		real_span_mask_ltoken = torch.LongTensor(real_span_mask_ltoken)
		span_label_ltoken = torch.LongTensor(span_label_ltoken)
		all_span_lens = torch.LongTensor(all_span_lens)
		morph_idxs = torch.LongTensor(morph_idxs)
		all_span_weights = torch.Tensor(all_span_weights)

		output = [token_ids, type_ids, all_span_idxs_ltoken, morph_idxs, span_label_ltoken, all_span_lens, all_span_weights, real_span_mask_ltoken, 
		   			words, all_span_word, all_span_idxs]
		
		output.append(item)

		# mlm task
		token_ids_mlm = token_ids.clone()
		cls_tok_id, sep_tok_id = tokenizer.token_to_id(cls_tok), tokenizer.token_to_id(sep_tok)
		no_mask_tok_ids = [cls_tok_id, sep_tok_id, 6]

		mask_tok_id = tokenizer.token_to_id(mask_tok)
		sample_mask = torch.rand(len(token_ids)) < self.mask_rate
		
		attention_mask_mlm, mlm_labels = [], []
		loss_ignore_idx = CrossEntropyLoss().ignore_index
		for i, flag in enumerate(sample_mask):
			if flag and (token_ids[i] not in no_mask_tok_ids):
				mlm_labels.append(token_ids[i])
				attention_mask_mlm.append(0)
				token_ids_mlm[i] = mask_tok_id
			else:
				mlm_labels.append(loss_ignore_idx)
				attention_mask_mlm.append(1)

		attention_mask_mlm = torch.LongTensor(attention_mask_mlm)
		mlm_labels = torch.LongTensor(mlm_labels)
		
		output.append(token_ids_mlm)
		output.append(attention_mask_mlm)
		output.append(mlm_labels)

		return output
		

	def case_feature_tokenLevel(self, morph2idx, span_idxs, words, max_spanlen):
		caseidxs = []
		for idxs in span_idxs:
			sid, eid = idxs
			span_word = words[sid:eid + 1]
			caseidx1 = [0 for _ in range(max_spanlen)]
			for j, token in enumerate(span_word):
				tfeat = ''
				if token.isupper():
					tfeat = 'isupper'
				elif token.islower():
					tfeat = 'islower'
				elif token.istitle():
					tfeat = 'istitle'
				elif token.isdigit():
					tfeat = 'isdigit'
				else:
					tfeat = 'other'
				caseidx1[j] =morph2idx[tfeat]
			caseidxs.append(caseidx1)
		return caseidxs


	def pad(self, lst, value=None, max_length=None):
		max_length = max_length or self.max_length
		while len(lst) < max_length:
			lst.append(value)
		return lst


	def convert2tokenIdx(self, words, tokens, type_ids, offsets, span_idxs, span_idxLab):
		max_length = self.max_length
		sidxs = [x1 + sum([len(w) for w in words[:x1]]) for (x1, x2) in span_idxs]
		eidxs = [x2 + sum([len(w) for w in words[:x2 + 1]]) for (x1, x2) in span_idxs]

		span_idxs_new_label = {}
		for ns, ne, ose in zip(sidxs, eidxs, span_idxs):
			os, oe = ose
			oes_str = "{};{}".format(os, oe)
			nes_str = "{};{}".format(ns, ne)
			if oes_str in span_idxLab:
				label = span_idxLab[oes_str]
				span_idxs_new_label[nes_str] = label
			else:
				span_idxs_new_label[nes_str] = 'O'

		origin_offset2token_sidx = {}
		origin_offset2token_eidx = {}
		for token_idx in range(len(tokens)):
			token_start, token_end = offsets[token_idx]
			if token_start == token_end == 0:
				continue
			origin_offset2token_sidx[token_start] = token_idx
			origin_offset2token_eidx[token_end] = token_idx

		# convert the position from character-level to token-level.
		span_new_sidxs = []
		span_new_eidxs = []
		n_span_keep = 0        

		select_mask = []
		for start, end in zip(sidxs, eidxs):
			try:
				temp = (origin_offset2token_eidx[end], origin_offset2token_sidx[start])
			except KeyError:
				print('words', words)
				print('tokens', tokens)
				print('offsets', offsets)
				print('span_idxs', span_idxs)
				print('sidxs', sidxs)
				print('eidxs', eidxs) 
				print('origin_offset2token_eidx', len(origin_offset2token_eidx), origin_offset2token_eidx)
				print('origin_offset2token_sidx', len(origin_offset2token_sidx), origin_offset2token_sidx)

			if origin_offset2token_eidx[end] > max_length - 1 or origin_offset2token_sidx[start] > max_length - 1:
				select_mask.append(0)
				continue
			select_mask.append(1)
			span_new_sidxs.append(origin_offset2token_sidx[start])
			span_new_eidxs.append(origin_offset2token_eidx[end])
			n_span_keep += 1

		all_span_word = []
		for (sidx, eidx) in span_idxs:
			all_span_word.append(words[sidx:eidx + 1])

		all_span_word = [w for w,i in zip(all_span_word, select_mask) if i==1]

		span_idxs_ltoken = []
		for sidx, eidx in zip(span_new_sidxs, span_new_eidxs):
			span_idxs_ltoken.append((sidx, eidx))

		return span_idxs_ltoken, all_span_word, span_idxs_new_label, select_mask



if __name__ == '__main__':
	pass