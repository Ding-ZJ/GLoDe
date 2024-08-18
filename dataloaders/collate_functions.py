# encoding: utf-8

import torch
from typing import List
from torch.nn import CrossEntropyLoss

len_max = 128


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Dynamic padding (i.e., different batch has different max_seq_len, based on the longest seq in the batch)
    Args:
        batch: a batch of samples, each contains a list of field data (Tensor): tokens,type_ids,span_idxs_ltoken,morph_idxs, ...
    Returns:
        output: list of field batched data, which is shaped: [all_tokens, all_type_ids, all_span_idxs_ltoken, ……]
    """

    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch) 
    max_num_span = max(x[2].shape[0] for x in batch)
    output = []

    # padding for token_ids and type_ids
    for field_idx in range(2):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # padding for all_span_idx_ltoken
    pad_all_span_idxs_ltoken = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append((0,0))
        pad_all_span_idxs_ltoken.append(sma)
    pad_all_span_idxs_ltoken = torch.Tensor(pad_all_span_idxs_ltoken)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][2]
        pad_all_span_idxs_ltoken[sample_idx, : data.shape[0],:] = data
    output.append(pad_all_span_idxs_ltoken)

    # padding for morph_idxs
    pad_morph_len = len(batch[0][3][0])
    pad_morph = [0 for i in range(pad_morph_len)]
    pad_morph_idxs = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append(pad_morph)
        pad_morph_idxs.append(sma)
    pad_morph_idxs = torch.LongTensor(pad_morph_idxs)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_morph_idxs[sample_idx, : data.shape[0], :] = data
    output.append(pad_morph_idxs)

    # padding for span_label_ltoken, all_span_lens, all_span_weights, real_span_mask
    for field_idx in [4,5,6,7]:
        pad_output = torch.full([batch_size, max_num_span], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # 
    words = []
    for sample_idx in range(batch_size):
        words.append(batch[sample_idx][8])
    output.append(words)
    
    # 
    all_span_word = []
    for sample_idx in range(batch_size):
        all_span_word.append(batch[sample_idx][9])
    output.append(all_span_word)

    # 
    all_span_idxs = []
    for sample_idx in range(batch_size):
        all_span_idxs.append(batch[sample_idx][10])
    output.append(all_span_idxs)
    
    # GUID or mlm_labels
    if len(batch[0]) == 12:
        demo = batch[0][11]
        if isinstance(demo, int):
            guids = []
            for sample_idx in range(batch_size):
                guids.append(batch[sample_idx][11])
            output.append(torch.LongTensor(guids))
        # rm the mlm parts now
        elif isinstance(demo, torch.Tensor):
            pad_mlm_labels = torch.full([batch_size, max_length], -100, dtype=batch[0][11].dtype)
            for sample_idx in range(batch_size):
                data = batch[sample_idx][11]
                pad_mlm_labels[sample_idx][: data.shape[0]] = data
            output.append(pad_mlm_labels)
        else:
            raise ValueError('Unknown features of sample[11]!')            

    return output


def collate_to_max_length_for_mlmTask(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
             token_ids, type_ids, attention_mask, mlm_labels
    Returns:
        output: list of field batched data, which shape is [batch, max_length]  # *[all_token_ids, all_type_ids, all_attention_mask, all_mlm_labels]
    """

    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    # padding for token_ids, type_ids, attention_mask
    for field_idx in range(3):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # padding for mlm_labels
    loss_ignore_idx = CrossEntropyLoss().ignore_index	# -100
    pad_output = torch.full([batch_size, max_length], loss_ignore_idx, dtype=batch[0][3].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)
              
    return output



def collate_to_max_length_U(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    for the BERTNERDataset_U dataset
    """

    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    max_num_span = max(x[2].shape[0] for x in batch)
    output = []

    # padding for token_ids, type_ids
    for field_idx in range(2):
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # padding for all_span_idx_ltoken
    pad_all_span_idxs_ltoken = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append((0,0))
        pad_all_span_idxs_ltoken.append(sma)
    pad_all_span_idxs_ltoken = torch.Tensor(pad_all_span_idxs_ltoken)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][2]
        pad_all_span_idxs_ltoken[sample_idx, : data.shape[0],:] = data
    output.append(pad_all_span_idxs_ltoken)

    # padding for morph_idxs
    pad_morph_len = len(batch[0][3][0])
    pad_morph = [0 for i in range(pad_morph_len)]
    pad_morph_idxs = []
    for i in range(batch_size):
        sma = []
        for j in range(max_num_span):
            sma.append(pad_morph)
        pad_morph_idxs.append(sma)
    pad_morph_idxs = torch.LongTensor(pad_morph_idxs)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][3]
        pad_morph_idxs[sample_idx, : data.shape[0], :] = data
    output.append(pad_morph_idxs)

    # padding for span_label_ltoken, all_span_lens, all_span_weights, real_span_mask
    for field_idx in [4,5,6,7]:
        pad_output = torch.full([batch_size, max_num_span], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    #
    words = []
    for sample_idx in range(batch_size):
        words.append(batch[sample_idx][8])
    output.append(words)
    
    #
    all_span_word = []
    for sample_idx in range(batch_size):
        all_span_word.append(batch[sample_idx][9])
    output.append(all_span_word)

    #
    all_span_idxs = []
    for sample_idx in range(batch_size):
        all_span_idxs.append(batch[sample_idx][10])
    output.append(all_span_idxs)
    
    #
    guids = []
    for sample_idx in range(batch_size):
        guids.append(batch[sample_idx][11])
    output.append(torch.LongTensor(guids))

    # padding for tokens_mlm、attention_mlm
    for field_idx in [12, 13]:
        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # padding for mlm_labels
    loss_ignore_idx = CrossEntropyLoss().ignore_index
    pad_output = torch.full([batch_size, max_length], loss_ignore_idx, dtype=batch[0][14].dtype)
    for sample_idx in range(batch_size):
        data = batch[sample_idx][14]
        pad_output[sample_idx][: data.shape[0]] = data
    output.append(pad_output)
    
    return output
