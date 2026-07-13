#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sbert_pca_embeddings.py
=======================
One-shot preprocessing script for the SBERT-modified DeepSC.

Procedure:
  1. Load the custom vocabulary (txt/vocab.json).
  2. Encode every token with the SBERT 'all-MiniLM-L6-v2' model  -> [V, 384].
  3. Fit PCA(n_components=128) and transform the embeddings        -> [V, 128].
  4. Save the result to txt/sbert_pca_embeddings.npy.

Usage:
    python sbert_pca_embeddings.py [--vocab-file txt/vocab.json]
                                   [--output    txt/sbert_pca_embeddings.npy]
                                   [--batch-size 256]
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate SBERT+PCA embeddings for DeepSC vocab.')
parser.add_argument('--vocab-file',  default='txt/vocab.json',
                    help='Path to the custom vocab JSON file.')
parser.add_argument('--output',      default='txt/sbert_pca_embeddings.npy',
                    help='Output path for the .npy embedding matrix.')
parser.add_argument('--model-name',  default='all-MiniLM-L6-v2',
                    help='SentenceTransformer model identifier.')
parser.add_argument('--pca-dim',     default=128, type=int,
                    help='Target PCA dimension (must match d_model).')
parser.add_argument('--batch-size',  default=256, type=int,
                    help='Encoding batch size.')
parser.add_argument('--random-seed', default=42,  type=int,
                    help='Random seed for PCA reproducibility.')


def main():
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load vocabulary
    # ------------------------------------------------------------------
    print(f'Loading vocabulary from: {args.vocab_file}')
    with open(args.vocab_file, 'r') as f:
        vocab = json.load(f)
    token_to_idx = vocab['token_to_idx']
    vocab_size = len(token_to_idx)

    # Build an ordered list of tokens by their integer index
    # (guarantees row i in the embedding matrix corresponds to token index i)
    tokens_by_idx = [''] * vocab_size
    for token, idx in token_to_idx.items():
        tokens_by_idx[idx] = token

    print(f'  Vocabulary size: {vocab_size}')
    print(f'  Sample tokens: {tokens_by_idx[:6]}')

    # ------------------------------------------------------------------
    # 2. Encode with SBERT (all-MiniLM-L6-v2)  ->  [V, 384]
    # ------------------------------------------------------------------
    print(f'\nLoading SBERT model: {args.model_name}')
    sbert = SentenceTransformer(args.model_name)

    print(f'Encoding {vocab_size} tokens (batch_size={args.batch_size}) ...')
    embeddings = sbert.encode(
        tokens_by_idx,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    print(f'  Raw embedding shape: {embeddings.shape}')   # (V, 384)
    assert embeddings.shape == (vocab_size, 384), (
        f'Unexpected SBERT output shape: {embeddings.shape}')

    # ------------------------------------------------------------------
    # 3. PCA  384 -> 128
    # ------------------------------------------------------------------
    print(f'\nFitting PCA({args.pca_dim}) ...')
    pca = PCA(n_components=args.pca_dim, random_state=args.random_seed)
    embeddings_pca = pca.fit_transform(embeddings)   # [V, 128]
    explained = pca.explained_variance_ratio_.sum()
    print(f'  PCA output shape   : {embeddings_pca.shape}')
    print(f'  Explained variance : {explained * 100:.2f}%')

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.output, embeddings_pca.astype(np.float32))
    print(f'\nSaved embeddings shape: {embeddings_pca.shape}  ->  {args.output}')


if __name__ == '__main__':
    main()
