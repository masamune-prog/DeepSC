# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from functools import partial
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data, collate_data_bert
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='txt/vocab.json', type=str,
                    help='Custom vocab file (only used when --use-bert-encoder is False)')
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str,
                    help='Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
# BERT encoder options
parser.add_argument('--use-bert-encoder', action='store_true',
                    help='Replace the custom Transformer encoder with a pretrained BERT encoder')
parser.add_argument('--bert-model-name', default='bert-base-multilingual-cased', type=str,
                    help='HuggingFace model identifier for the BERT encoder')
parser.add_argument('--finetune-bert', action='store_true',
                    help='Unfreeze BERT parameters during training (default: frozen)')
parser.add_argument('--snr-min', default=0, type=float,
                    help='Minimum SNR (dB) sampled per batch during training')
parser.add_argument('--snr-max', default=18, type=float,
                    help='Maximum SNR (dB) sampled per batch during training')
parser.add_argument('--patience', default=10, type=int,
                    help='Early-stopping patience: stop if val loss does not improve for this many epochs')
parser.add_argument('--train-channels', nargs='+',
                    default=['AWGN', 'Rayleigh', 'Rician'],
                    help='Channel types to sample from per batch during training. '
                         'E.g. --train-channels AWGN Rayleigh')
parser.add_argument('--log-file', default='training_log.csv', type=str,
                    help='CSV file to write per-epoch train/val loss for plotting')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net, collate_fn):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_fn)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            snr_db = np.random.uniform(args.snr_min, args.snr_max)
            noise_std = SNR_to_noise(snr_db)
            #pick random channel from training
            channel = np.random.choice(args.train_channels)
            if args.use_bert_encoder:
                # sents is a dict; src == trg for auto-encoding
                loss = val_step(net, sents, sents, noise_std, pad_idx,
                                criterion, channel)
            else:
                sents = sents.to(device)
                loss = val_step(net, sents, sents, 0.1, pad_idx,
                                criterion, channel)

            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total / len(test_iterator)


def train(epoch, args, net, collate_fn, mi_net=None):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_fn)
    pbar = tqdm(train_iterator)

    total_loss = 0.0
    n_batches = 0

    for sents in pbar:
        # Sample a fresh SNR and channel type per batch.
        # start with a high snr for the first few epochs to warm up
        if epoch < 5:
            snr_db = np.random.uniform(args.snr_min + 10, args.snr_max)
        else:
            snr_db = np.random.uniform(args.snr_min, args.snr_max)
        noise_std = SNR_to_noise(snr_db)
        batch_channel = np.random.choice(args.train_channels)

        if args.use_bert_encoder:
            # sents is a dict; src == trg for auto-encoding
            if mi_net is not None:
                mi = train_mi(net, mi_net, sents, noise_std, pad_idx, mi_opt, batch_channel)
                loss = train_step(net, sents, sents, noise_std, pad_idx,
                                  optimizer, criterion, batch_channel, mi_net)
                pbar.set_description(
                    'Epoch: {};  Type: Train; Ch: {}; SNR: {:.0f}dB; Loss: {:.5f}; MI {:.5f}'.format(
                        epoch + 1, batch_channel, snr_db, loss, mi
                    )
                )
            else:
                loss = train_step(net, sents, sents, noise_std, pad_idx,
                                  optimizer, criterion, batch_channel)
                pbar.set_description(
                    'Epoch: {};  Type: Train; Ch: {}; SNR: {:.0f}dB; Loss: {:.5f}'.format(
                        epoch + 1, batch_channel, snr_db, loss
                    )
                )
        else:
            sents = sents.to(device)
            if mi_net is not None:
                mi = train_mi(net, mi_net, sents, noise_std, pad_idx, mi_opt, batch_channel)
                loss = train_step(net, sents, sents, noise_std, pad_idx,
                                  optimizer, criterion, batch_channel, mi_net)
                pbar.set_description(
                    'Epoch: {};  Type: Train; Ch: {}; SNR: {:.0f}dB; Loss: {:.5f}; MI {:.5f}'.format(
                        epoch + 1, batch_channel, snr_db, loss, mi
                    )
                )
            else:
                loss = train_step(net, sents, sents, noise_std, pad_idx,
                                  optimizer, criterion, batch_channel)
                pbar.set_description(
                    'Epoch: {};  Type: Train; Ch: {}; SNR: {:.0f}dB; Loss: {:.5f}'.format(
                        epoch + 1, batch_channel, snr_db, loss
                    )
                )

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


if __name__ == '__main__':
    setup_seed(10)
    args = parser.parse_args()

    if args.use_bert_encoder:
        from transformers import BertTokenizerFast
        print(f'Loading BERT tokenizer: {args.bert_model_name}')
        bert_tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        bert_vocab_size = bert_tokenizer.vocab_size  # ~119,547

        # BERT special token IDs
        pad_idx = bert_tokenizer.pad_token_id      # 0
        start_idx = bert_tokenizer.cls_token_id    # 101
        end_idx = bert_tokenizer.sep_token_id      # 102

        # We still need the custom vocab to reverse-decode the existing pkl files
        # into raw text before re-tokenising with BERT.
        vocab = json.load(open(args.vocab_file, 'rb'))
        token_to_idx = vocab['token_to_idx']
        idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
        custom_end_idx = token_to_idx['<END>']
        custom_pad_idx = token_to_idx['<PAD>']

        collate_fn = partial(
            collate_data_bert,
            bert_tokenizer=bert_tokenizer,
            idx_to_token=idx_to_token,
            end_idx=custom_end_idx,
            pad_idx=custom_pad_idx,
            max_length=args.MAX_LENGTH,
        )

        deepsc = DeepSC(
            args.num_layers,
            src_vocab_size=bert_vocab_size,   # unused by BertSemanticEncoder
            trg_vocab_size=bert_vocab_size,
            src_max_len=args.MAX_LENGTH,
            trg_max_len=args.MAX_LENGTH,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dff=args.dff,
            dropout=0.1,
            use_bert_encoder=True,
            bert_model_name=args.bert_model_name,
            freeze_bert=not args.finetune_bert,
        ).to(device)

    else:
        """ preparing the dataset (custom vocab path) """
        vocab = json.load(open(args.vocab_file, 'rb'))
        token_to_idx = vocab['token_to_idx']
        num_vocab = len(token_to_idx)
        pad_idx = token_to_idx['<PAD>']
        start_idx = token_to_idx['<START>']
        end_idx = token_to_idx['<END>']

        collate_fn = collate_data

        deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')

    # Only optimise trainable parameters (BERT frozen params are excluded automatically)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, deepsc.parameters()),
        lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4
    )
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)

    initNetParams(deepsc)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Initialise CSV log
    import csv
    log_path = args.log_file
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'elapsed_s'])

    for epoch in range(args.epochs):
        start = time.time()

        avg_train_loss = train(epoch, args, deepsc, collate_fn)
        avg_val_loss   = validate(epoch, args, deepsc, collate_fn)
        elapsed = time.time() - start

        # Append to CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1,
                                    f'{avg_train_loss:.6f}',
                                    f'{avg_val_loss:.6f}',
                                    f'{elapsed:.1f}'])

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            ckpt_path = os.path.join(
                args.checkpoint_path,
                'checkpoint_{}.pt'.format(str(epoch + 1).zfill(2))
            )
            deepsc.eval()
            try:
                scripted = torch.jit.script(deepsc)
            except Exception as e:
                print(f'torch.jit.script failed ({e}); falling back to torch.jit.trace.')
                # Provide a dummy forward pass to trace; adjust as needed for your model's signature.
                scripted = torch.jit.trace(deepsc, example_inputs=None, strict=False)
            torch.jit.save(scripted, ckpt_path)
            deepsc.train()
            print('Epoch {:02d}: val loss improved to {:.5f} — checkpoint saved.'.format(
                epoch + 1, best_val_loss))
        else:
            epochs_no_improve += 1
            print('Epoch {:02d}: val loss {:.5f} (no improvement, patience {}/{})'.format(
                epoch + 1, avg_val_loss, epochs_no_improve, args.patience))
            if epochs_no_improve >= args.patience:
                print('Early stopping triggered after {} epochs with no improvement.'.format(
                    args.patience))
                break

        print('Epoch {:02d} elapsed: {:.1f}s'.format(epoch + 1, elapsed))

    print(f'Training log saved to: {log_path}')
