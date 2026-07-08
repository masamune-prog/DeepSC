# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: EurDataset.py
@Time: 2021/3/31 23:20
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

# Special tokens in the custom vocabulary that should be stripped before
# passing text to the BERT tokenizer.
_CUSTOM_SPECIAL_TOKENS = {'<PAD>', '<START>', '<END>', '<UNK>'}


class EurDataset(Dataset):
    def __init__(self, split='train'):
        with open('txt/{}_data.pkl'.format(split), 'rb') as f:
            self.data = pickle.load(f)

    def __getitem__(self, index):
        sents = self.data[index]
        return sents

    def __len__(self):
        return len(self.data)


def collate_data(batch):
    batch_size = len(batch)
    max_len = max(map(lambda x: len(x), batch))   # get the max length of sentence in current batch
    sents = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x), reverse=True)

    for i, sent in enumerate(sort_by_len):
        length = len(sent)
        sents[i, :length] = sent  # padding the questions

    return torch.from_numpy(sents)


def _ids_to_text(token_ids, idx_to_token, end_idx, pad_idx):
    """Convert a sequence of custom-vocab integer IDs to a plain text string.

    Stops at ``end_idx`` and skips special tokens.
    """
    words = []
    for idx in token_ids:
        idx = int(idx)
        if idx == end_idx:
            break
        token = idx_to_token.get(idx, '')
        if token and token not in _CUSTOM_SPECIAL_TOKENS:
            words.append(token)
    return ' '.join(words)


def collate_data_bert(batch, bert_tokenizer, idx_to_token, end_idx, pad_idx,
                      max_length=30):
    """Collate function for the BERT-encoder DeepSC variant.

    Converts each sample from custom-vocab integer IDs back to raw text, then
    tokenises with the supplied BERT tokenizer.  Both the encoder source and the
    decoder target use the BERT vocabulary (the system is auto-encoding, so
    src == trg).

    Args:
        batch: list of numpy arrays containing custom-vocab token IDs.
        bert_tokenizer: a HuggingFace ``PreTrainedTokenizerFast`` instance.
        idx_to_token: dict mapping custom vocab int → token string.
        end_idx: integer ID of the <END> token in the custom vocab.
        pad_idx: integer ID of the <PAD> token in the custom vocab.
        max_length: maximum sequence length passed to the BERT tokenizer.

    Returns:
        dict with keys:
            ``input_ids``      – LongTensor [batch, seq_len] (BERT token IDs)
            ``attention_mask`` – LongTensor [batch, seq_len] (1=real, 0=pad)
    """
    sentences = [
        _ids_to_text(sample, idx_to_token, end_idx, pad_idx)
        for sample in batch
    ]

    encoded = bert_tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    # We only need input_ids and attention_mask; token_type_ids are not used.
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
    }