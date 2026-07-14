# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
FineWeb parquet data preprocessor for DeepSC.
Uses 0.parquet as the data source and applies the same preprocessing
pipeline as preprocess_text.py (normalize, cut, deduplicate, build vocab,
encode, and split 90/10 into train/test pickle files).
"""
import unicodedata
import re
from w3lib.html import remove_tags
import pickle
import argparse
import json
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input-parquet', default='0.parquet', type=str,
                    help='Path to the input 0.parquet FineWeb file')
parser.add_argument('--text-column', default='text', type=str,
                    help='Column name containing raw text in the parquet file')
parser.add_argument('--output-train-dir', default='finewebtxt/train_data.pkl', type=str)
parser.add_argument('--output-test-dir', default='finewebtxt/test_data.pkl', type=str)
parser.add_argument('--output-vocab', default='finewebtxt/vocab.json', type=str)

SPECIAL_TOKENS = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

# ---------------------------------------------------------------------------
# Text-processing helpers (identical to preprocess_text.py)
# ---------------------------------------------------------------------------

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    # normalize unicode characters
    s = unicode_to_ascii(s)
    # remove XML/HTML tags
    s = remove_tags(s)
    # add white space before !.?
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    # change to lower letter
    s = s.lower()
    return s


def cutted_data(cleaned, MIN_LENGTH=4, MAX_LENGTH=30):
    cutted_lines = list()
    for line in cleaned:
        length = len(line.split())
        if length > MIN_LENGTH and length < MAX_LENGTH:
            line = [word for word in line.split()]
            cutted_lines.append(' '.join(line))
    return cutted_lines


def save_clean_sentences(sentence, save_path):
    pickle.dump(sentence, open(save_path, 'wb'))
    print('Saved: %s' % save_path)


def tokenize(s, delim=' ', add_start_token=True, add_end_token=True,
             punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))

    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens


def build_vocab(sequences, token_to_idx={}, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}

    for seq in sequences:
        seq_tokens = tokenize(seq, delim=delim, punct_to_keep=punct_to_keep,
                              punct_to_remove=punct_to_remove,
                              add_start_token=False, add_end_token=False)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)

    return token_to_idx


def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)

# ---------------------------------------------------------------------------
# FineWeb-specific data loading
# ---------------------------------------------------------------------------

def load_parquet_sentences(parquet_path, text_column='text'):
    """
    Load the 0.parquet file, split each document into individual sentences
    (one per newline), and return a flat list of raw sentence strings.
    """
    print(f'Loading parquet: {parquet_path}')
    df = pd.read_parquet(parquet_path, columns=[text_column])
    sentences = []
    for doc in tqdm(df[text_column].dropna(), desc='Splitting documents'):
        # Split on newlines, just as preprocess_text.py splits on '\n'
        for line in doc.strip().split('\n'):
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f'Input parquet : {args.input_parquet}')
    print(f'Text column   : {args.text_column}')

    # 1. Load raw sentences from parquet
    raw_sentences = load_parquet_sentences(args.input_parquet, args.text_column)
    print(f'Raw sentence count (before filtering): {len(raw_sentences)}')

    # 2. Normalize & cut  (identical logic to preprocess_text.py)
    print('Normalizing sentences ...')
    normalized = [normalize_string(s) for s in tqdm(raw_sentences)]
    cleaned = cutted_data(normalized)

    # 3. Deduplicate (identical to preprocess_text.py)
    a = {}
    for sent in cleaned:
        if sent not in a:
            a[sent] = 0
        a[sent] += 1
    sentences = list(a.keys())
    print(f'Number of sentences after dedup: {len(sentences)}')

    # 4. Build vocab (identical settings to preprocess_text.py)
    print('Build Vocab')
    token_to_idx = build_vocab(
        sentences, SPECIAL_TOKENS,
        punct_to_keep=[';', ','], punct_to_remove=['?', '.']
    )
    vocab = {'token_to_idx': token_to_idx}
    print(f'Number of words in Vocab: {len(token_to_idx)}')

    if args.output_vocab != '':
        with open(args.output_vocab, 'w') as f:
            json.dump(vocab, f)
        print(f'Vocab saved to {args.output_vocab}')

    # 5. Encode sentences (identical to preprocess_text.py)
    print('Start encoding txt')
    results = []
    for seq in tqdm(sentences):
        words = tokenize(seq, punct_to_keep=[';', ','], punct_to_remove=['?', '.'])
        tokens = [token_to_idx[word] for word in words]
        results.append(tokens)

    # 6. 90 / 10 train-test split (identical to preprocess_text.py)
    print('Writing Data')
    train_data = results[: round(len(results) * 0.9)]
    test_data  = results[round(len(results) * 0.9):]

    with open(args.output_train_dir, 'wb') as f:
        pickle.dump(train_data, f)
    with open(args.output_test_dir, 'wb') as f:
        pickle.dump(test_data, f)

    print(f'Train sentences : {len(train_data)}')
    print(f'Test  sentences : {len(test_data)}')
    print('Done.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
