# -*- coding: utf-8 -*-
"""
Training script for RayleighTextDecoderDiffusion, structured to mirror
train.py (DeepSC) so the two approaches can be compared on identical
data, identical SNR sampling, and identical logging format.

Usage mirrors the DeepSC script, e.g.:
    python train_diffusion.py --checkpoint-path checkpoints/diffusion-Rayleigh \
        --log-file training_log_diffusion.csv --epochs 80 --batch-size 128

NOTE: adjust the model import below to match wherever you saved the
RayleighTextDecoderDiffusion class (I've assumed `rayleigh_diffusion.py`).
"""
import os
import csv
import json
import time
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EurDataset, collate_data
from models.diffusion_decoder import RayleighTextDecoderDiffusion

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='txt/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/diffusion-Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--embed-dim', default=128, type=int,
                    help='Matches --d-model in the DeepSC script for fair comparison')
parser.add_argument('--num-steps', default=500, type=int,
                    help='Diffusion timesteps')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--snr-min', default=0, type=float,
                    help='Minimum SNR (dB) sampled per batch during training')
parser.add_argument('--snr-max', default=18, type=float,
                    help='Maximum SNR (dB) sampled per batch during training')
parser.add_argument('--patience', default=10, type=int,
                    help='Early-stopping patience on val loss')
parser.add_argument('--log-file', default='training_log_diffusion.csv', type=str)
parser.add_argument('--seed', default=10, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(epoch, args, net, optimizer, collate_fn, pad_idx):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                 pin_memory=True, collate_fn=collate_fn)
    pbar = tqdm(train_iterator)

    total_loss = 0.0
    n_batches = 0

    net.train()
    for sents in pbar:
        # Sample a fresh SNR per batch -- same convention as the DeepSC script,
        # so both models see the same SNR distribution over training.
        snr_db = np.random.uniform(args.snr_min, args.snr_max)

        sents = sents.to(device)
        # 1 = real token, 0 = padding -- matches create_masks' (src == pad)
        # convention inverted, since our loss code expects 1=keep.
        pad_mask = (sents != pad_idx).long()

        optimizer.zero_grad()
        loss = net(sents, snr_db=snr_db, pad_mask=pad_mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        n_batches += 1

        pbar.set_description(
            'Epoch: {}; Type: Train; SNR: {:.0f}dB; Loss: {:.5f}'.format(
                epoch + 1, snr_db, loss_val
            )
        )

    return total_loss / n_batches


def validate(epoch, args, net, collate_fn, pad_idx):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_fn)
    net.eval()
    pbar = tqdm(test_iterator)

    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for sents in pbar:
            snr_db = np.random.uniform(args.snr_min, args.snr_max)
            sents = sents.to(device)
            pad_mask = (sents != pad_idx).long()

            loss = net(sents, snr_db=snr_db, pad_mask=pad_mask)
            loss_val = loss.item()
            total_loss += loss_val
            n_batches += 1

            pbar.set_description(
                'Epoch: {}; Type: VAL; SNR: {:.0f}dB; Loss: {:.5f}'.format(
                    epoch + 1, snr_db, loss_val
                )
            )

    return total_loss / n_batches


if __name__ == '__main__':
    setup_seed(10)
    args = parser.parse_args()
    setup_seed(args.seed)

    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx['<PAD>']

    collate_fn = collate_data

    net = RayleighTextDecoderDiffusion(
        vocab_size=num_vocab,
        embed_dim=args.embed_dim,
        # EurDataset sentences are MAX_LENGTH + 1 tokens long (DeepSC's decoder
        # consumes them shifted via trg_inp = trg[:, :-1] / trg_real = trg[:, 1:];
        # this model has no such shift, so it needs the full length).
        max_seq_len=args.MAX_LENGTH + 1,
        num_steps=args.num_steps,
    ).to(device)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    log_path = args.log_file
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'elapsed_s'])

    for epoch in range(args.epochs):
        start = time.time()

        avg_train_loss = train(epoch, args, net, optimizer, collate_fn, pad_idx)
        avg_val_loss = validate(epoch, args, net, collate_fn, pad_idx)
        elapsed = time.time() - start

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
            net.eval()
            torch.save(net, ckpt_path)
            net.train()
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