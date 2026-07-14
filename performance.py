# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
performance_diffusion.py

Evaluates RayleighTextDecoderDiffusion using the exact same performance()
harness, BLEU-1..4 breakdown, BERTScore option, and CSV schema as
performance.py (DeepSC), so the two results_*.csv / predictions_*.csv files
are directly comparable / concatenable.
"""
import os
import csv
import json
import torch
import argparse
import numpy as np
from functools import partial

from dataset import EurDataset, collate_data
from utils import SeqtoText, SNR_to_noise
from rayleigh_diffusion import RayleighTextDecoderDiffusion

# Reuse the exact same evaluation harness, BERTScore wrapper, and seeding
# as DeepSC's performance.py -- this is what guarantees the CSV schema and
# console output are identical between the two scripts.
from performance import performance, BertScoreEvaluator, setup_seed

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='txt/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/diffusion-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str,
                    help='Kept for CSV/log parity with performance.py. '
                         'rayleigh_diffusion.py only implements a Rayleigh channel '
                         '(matching utils.Channels.Rayleigh) -- a value other than '
                         '"Rayleigh" here only changes labeling, not behavior.')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--embed-dim', default=128, type=int,
                    help='Must match the embed_dim the checkpoint was trained with')
parser.add_argument('--num-steps', default=500, type=int,
                    help='Must match the num_steps the checkpoint was trained with')
parser.add_argument('--no-clamping', action='store_true',
                    help='Disable clamping during reverse diffusion sampling (default: on)')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--output-csv', default=None, type=str,
                    help='Defaults to results_<channel>_<model>.csv in the checkpoint directory.')
parser.add_argument('--predictions-csv', default=None, type=str,
                    help='Defaults to predictions_<channel>_<model>.csv in the checkpoint directory.')
parser.add_argument('--bert-score', action='store_true')
parser.add_argument('--bert-score-model', default='bert-base-multilingual-cased', type=str)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _decode_diffusion(net, sents, noise_std, StoT, use_clamping):
    """Matches the decode_fn(net, sents, noise_std) -> (predicted, references)
    contract expected by performance(). noise_std is inverted back to snr_db
    exactly (SNR_to_noise is a clean bijection), so both models are evaluated
    at the identical SNR point for each sweep value."""
    sents = sents.to(device)

    snr_linear = 1.0 / (2.0 * noise_std ** 2)
    snr_db = 10.0 * np.log10(snr_linear)

    x_0 = net.embedding(sents)
    y_corrupted = net.apply_rayleigh_channel(x_0, snr_db=snr_db)
    out = net.decode_channel_output(y_corrupted, use_clamping=use_clamping)

    sentences = out.cpu().numpy().tolist()
    predicted = list(map(StoT.sequence_to_text, sentences))
    target_sent = sents.cpu().numpy().tolist()
    references = list(map(StoT.sequence_to_text, target_sent))
    return predicted, references


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(10)
    SNR = [0, 3, 6, 9, 12, 15, 18]

    if args.channel != 'Rayleigh':
        print(f'[WARNING] --channel={args.channel} has no effect: '
              f'rayleigh_diffusion.py only implements a Rayleigh channel.')

    # -----------------------------------------------------------------------
    # 1. Locate the latest checkpoint -- same convention as performance.py.
    # -----------------------------------------------------------------------
    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth') and not fn.endswith('.pt'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    if not model_paths:
        raise FileNotFoundError(f'No .pt/.pth checkpoints found in {args.checkpoint_path}')

    model_paths.sort(key=lambda x: x[1])
    model_path, _ = model_paths[-1]
    print(f'Loading checkpoint: {model_path}')

    try:
        raw_ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        raw_ckpt = torch.load(model_path, map_location=device, weights_only=False)

    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    end_idx = token_to_idx['<END>']

    # train_diffusion.py checkpoints via torch.save(net, path) (full object),
    # same convention as train.py's DeepSC checkpoints -- but also accept a
    # plain state_dict for portability.
    if isinstance(raw_ckpt, RayleighTextDecoderDiffusion):
        net = raw_ckpt.to(device)
    elif isinstance(raw_ckpt, dict):
        net = RayleighTextDecoderDiffusion(
            vocab_size=num_vocab,
            embed_dim=args.embed_dim,
            # Must match train_diffusion.py: EurDataset sentences are
            # MAX_LENGTH + 1 tokens long.
            max_seq_len=args.MAX_LENGTH + 1,
            num_steps=args.num_steps,
        ).to(device)
        net.load_state_dict(raw_ckpt)
    else:
        raise TypeError(f'Unexpected checkpoint type: {type(raw_ckpt)}')

    print('Model loaded from:', model_path)

    StoT = SeqtoText(token_to_idx, end_idx)
    collate_fn = collate_data
    decode_fn = partial(_decode_diffusion, StoT=StoT, use_clamping=not args.no_clamping)

    # -----------------------------------------------------------------------
    # 2. Optional BERTScore evaluator -- identical to performance.py.
    # -----------------------------------------------------------------------
    bert_scorer = None
    if args.bert_score:
        print(f'Initialising BERTScore evaluator (model: {args.bert_score_model})...')
        bert_scorer = BertScoreEvaluator(
            model_type=args.bert_score_model,
            lang='en',
            batch_size=args.batch_size,
        )

    bleu_scores, bert_scores, all_sentences = performance(
        args, SNR, net, collate_fn, decode_fn, bert_scorer=bert_scorer
    )

    # -----------------------------------------------------------------------
    # 3. Console output + CSV writing -- identical schema to performance.py
    #    so results_*.csv / predictions_*.csv from both scripts line up.
    # -----------------------------------------------------------------------
    sep_width = 10 + 4 * 11 + (3 * 11 if bert_scores else 0)

    print('\n' + '=' * sep_width)
    print(' Results for ' + args.channel + ' Channel using ' + args.checkpoint_path + ' model')
    print('=' * sep_width)
    print(f'{"SNR (dB)":>10} | {"BLEU-1":>8} | {"BLEU-2":>8} | {"BLEU-3":>8} | {"BLEU-4":>8}', end='')
    if bert_scores:
        print(f' | {"BERT-F1":>8} | {"BERT-P":>8} | {"BERT-R":>8}', end='')
    print()
    print('-' * sep_width)
    for i, snr in enumerate(SNR):
        row = (f'{snr:>10}'
               f' | {bleu_scores["bleu1"][i]:>8.4f}'
               f' | {bleu_scores["bleu2"][i]:>8.4f}'
               f' | {bleu_scores["bleu3"][i]:>8.4f}'
               f' | {bleu_scores["bleu4"][i]:>8.4f}')
        if bert_scores:
            row += (f' | {bert_scores["f1"][i]:>8.4f}'
                    f' | {bert_scores["precision"][i]:>8.4f}'
                    f' | {bert_scores["recall"][i]:>8.4f}')
        print(row)

    model_tag = os.path.basename(args.checkpoint_path.rstrip('/\\'))

    csv_path = (args.output_csv
                if args.output_csv
                else os.path.join(args.checkpoint_path,
                                  f'results_{args.channel}_{model_tag}.csv'))

    fieldnames = ['channel', 'checkpoint', 'snr_db',
                  'bleu1', 'bleu2', 'bleu3', 'bleu4']
    if bert_scores:
        fieldnames += ['bert_f1', 'bert_precision', 'bert_recall']

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, snr in enumerate(SNR):
            record = {
                'channel':    args.channel,
                'checkpoint': model_tag,
                'snr_db':     snr,
                'bleu1':      round(float(bleu_scores['bleu1'][i]), 6),
                'bleu2':      round(float(bleu_scores['bleu2'][i]), 6),
                'bleu3':      round(float(bleu_scores['bleu3'][i]), 6),
                'bleu4':      round(float(bleu_scores['bleu4'][i]), 6),
            }
            if bert_scores:
                record['bert_f1']        = round(float(bert_scores['f1'][i]),        6)
                record['bert_precision'] = round(float(bert_scores['precision'][i]), 6)
                record['bert_recall']    = round(float(bert_scores['recall'][i]),    6)
            writer.writerow(record)

    print(f'\nResults saved to: {csv_path}')

    pred_csv_path = (args.predictions_csv
                     if args.predictions_csv
                     else os.path.join(args.checkpoint_path,
                                       f'predictions_{args.channel}_{model_tag}.csv'))

    pred_fieldnames = ['epoch', 'snr_db', 'sample_idx', 'predicted', 'reference']
    pred_write_header = not os.path.exists(pred_csv_path)
    with open(pred_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
        if pred_write_header:
            writer.writeheader()
        writer.writerows(all_sentences)

    print(f'Predictions saved to: {pred_csv_path}')