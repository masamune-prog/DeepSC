# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import csv
import json
import torch
import argparse
import numpy as np
from functools import partial
from dataset import EurDataset, collate_data, collate_data_bert
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText
from tqdm import tqdm
import random
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='txt/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='txt/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=None, type=int,
                    help='(Deprecated) Sets both encoder and decoder layers. '
                         'Prefer --num-enc-layers / --num-dec-layers.')
parser.add_argument('--num-enc-layers', default=3, type=int,
                    help='Number of Transformer encoder layers (overridden by checkpoint auto-detection)')
parser.add_argument('--num-dec-layers', default=3, type=int,
                    help='Number of Transformer decoder layers (overridden by checkpoint auto-detection)')
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--output-csv', default=None, type=str,
                    help='Path to CSV file for saving aggregate results. '
                         'Defaults to results_<channel>_<model>.csv in the checkpoint directory.')
parser.add_argument('--predictions-csv', default=None, type=str,
                    help='Path to CSV file for saving per-sentence predictions. '
                         'Defaults to predictions_<channel>_<model>.csv in the checkpoint directory.')
# BERT encoder options
parser.add_argument('--use-bert-encoder', action='store_true',
                    help='Use the BERT encoder variant of DeepSC')
parser.add_argument('--bert-model-name', default='bert-base-multilingual-cased', type=str,
                    help='HuggingFace model identifier for the BERT encoder')
# Semantic similarity options
parser.add_argument('--bert-score', action='store_true',
                    help='Compute BERTScore semantic similarity in addition to BLEU')
parser.add_argument('--bert-score-model', default='bert-base-multilingual-cased', type=str,
                    help='Model used by BERTScore for semantic similarity scoring. '
                         'Can differ from --bert-model-name (the encoder model). '
                         'E.g. roberta-large for higher quality scores.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BertScoreEvaluator:
    """Thin wrapper around the bert-score library.

    Lazily loads the scoring model on first call and caches it.  All scoring
    runs on the same device as the rest of the pipeline.

    Args:
        model_type: HuggingFace model identifier used for embedding extraction.
            Defaults to ``'bert-base-multilingual-cased'``.
            Use ``'roberta-large'`` for higher quality English-only scores.
        lang: Language hint passed to bert-score (``'en'``, ``'de'``, etc.).
            Set to ``'en'`` for Europarl English data.
        batch_size: How many sentence pairs to score at once.
    """

    def __init__(self, model_type: str = 'bert-base-multilingual-cased',
                 lang: str = 'en', batch_size: int = 64):
        self.model_type = model_type
        self.lang = lang
        self.batch_size = batch_size
        self._scorer = None  # lazy init

    def _get_scorer(self):
        if self._scorer is None:
            try:
                from bert_score import BERTScorer
            except ImportError:
                raise ImportError(
                    'bert-score is not installed. '
                    'Run: uv pip install bert-score'
                )
            use_gpu = torch.cuda.is_available()
            self._scorer = BERTScorer(
                model_type=self.model_type,
                lang=self.lang,
                device='cuda' if use_gpu else 'cpu',
                batch_size=self.batch_size,
            )
        return self._scorer

    def score(self, predictions: list, references: list) -> dict:
        """Compute BERTScore for a list of prediction/reference pairs.

        Args:
            predictions: List of predicted strings.
            references:  List of reference strings (same length).

        Returns:
            dict with keys ``'precision'``, ``'recall'``, ``'f1'``,
            each a float (mean across the batch).
        """
        scorer = self._get_scorer()
        P, R, F1 = scorer.score(predictions, references)
        return {
            'precision': P.mean().item(),
            'recall':    R.mean().item(),
            'f1':        F1.mean().item(),
        }


def performance(args, SNR, net, collate_fn, decode_fn, bert_scorer=None):
    """Evaluate net over a range of SNR values.

    Args:
        args: parsed arguments.
        SNR: list of SNR values (dB) to evaluate at.
        net: the DeepSC model.
        collate_fn: DataLoader collate function (standard or BERT).
        decode_fn: callable(net, sents, noise_std) -> (predicted, references).
        bert_scorer: optional ``BertScoreEvaluator`` instance. When provided,
            BERTScore F1/P/R are computed alongside BLEU for each SNR point.

    Returns:
        bleu_scores: dict with keys 'bleu1'..'bleu4', each an np.ndarray of
            shape [len(SNR)] containing the mean n-gram BLEU for that order.
        bert_scores: dict with keys 'f1', 'precision', 'recall', each
            np.ndarray of shape [len(SNR)].  Empty dict if bert_scorer is None.
        all_sentences: list of dicts, one per sentence, with keys
            'epoch', 'snr_db', 'sample_idx', 'predicted', 'reference'.
    """
    bleu_scorers = {
        'bleu1': BleuScore(1, 0, 0, 0),
        'bleu2': BleuScore(0, 1, 0, 0),
        'bleu3': BleuScore(0, 0, 1, 0),
        'bleu4': BleuScore(0, 0, 0, 1),
    }

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_fn)

    all_bleu = {k: [] for k in bleu_scorers}
    all_bert_f1, all_bert_p, all_bert_r = [], [], []
    all_sentences = []  # flat list of per-sentence records

    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:
                    predicted, references = decode_fn(net, sents, noise_std)
                    word += predicted
                    target_word += references

                Tx_word.append(word)
                Rx_word.append(target_word)

                # Collect per-sentence records for this SNR.
                for sample_idx, (pred, ref) in enumerate(zip(word, target_word)):
                    all_sentences.append({
                        'epoch':      epoch,
                        'snr_db':     snr,
                        'sample_idx': sample_idx,
                        'predicted':  pred,
                        'reference':  ref,
                    })

            bleu_epoch = {k: [] for k in bleu_scorers}
            bert_f1_epoch, bert_p_epoch, bert_r_epoch = [], [], []

            for snr_idx, (sent1, sent2) in enumerate(zip(Tx_word, Rx_word)):
                print(f"\n" + "="*80)
                print(f" SNR: {SNR[snr_idx]} dB | Sample Comparisons")
                print(f"="*80)
                for pred, ref in zip(sent1[:5], sent2[:5]):
                    print(f"Predicted: {pred}")
                    print(f"Actual   : {ref}")
                    print("-"*40)

                # BLEU-1 through BLEU-4
                for key, scorer in bleu_scorers.items():
                    bleu_epoch[key].append(scorer.compute_blue_score(sent1, sent2))

                # BERTScore (optional)
                if bert_scorer is not None:
                    bs = bert_scorer.score(sent1, sent2)
                    bert_f1_epoch.append(bs['f1'])
                    bert_p_epoch.append(bs['precision'])
                    bert_r_epoch.append(bs['recall'])
                    print(f"BERTScore  F1={bs['f1']:.4f}  P={bs['precision']:.4f}  R={bs['recall']:.4f}")

            for key in bleu_scorers:
                arr = np.array(bleu_epoch[key])          # shape [len(SNR), batch]
                all_bleu[key].append(np.mean(arr, axis=1))  # shape [len(SNR)]

            if bert_scorer is not None:
                all_bert_f1.append(bert_f1_epoch)
                all_bert_p.append(bert_p_epoch)
                all_bert_r.append(bert_r_epoch)

    bleu_scores = {k: np.mean(np.array(all_bleu[k]), axis=0) for k in bleu_scorers}

    bert_scores = {}
    if bert_scorer is not None:
        bert_scores['f1']        = np.mean(np.array(all_bert_f1), axis=0)
        bert_scores['precision'] = np.mean(np.array(all_bert_p),  axis=0)
        bert_scores['recall']    = np.mean(np.array(all_bert_r),  axis=0)

    return bleu_scores, bert_scores, all_sentences



# ---------------------------------------------------------------------------
# Decode helpers — one per encoder type
# ---------------------------------------------------------------------------

def _decode_custom(net, sents, noise_std,
                   StoT, pad_idx, start_idx, channel, max_length):
    """Decode a batch using the custom-vocab encoder."""
    sents = sents.to(device)
    out = greedy_decode(net, sents, noise_std, max_length, pad_idx,
                        start_idx, channel)
    sentences = out.cpu().numpy().tolist()
    predicted = list(map(StoT.sequence_to_text, sentences))
    target_sent = sents.cpu().numpy().tolist()
    references = list(map(StoT.sequence_to_text, target_sent))
    return predicted, references


def _decode_bert(net, sents, noise_std,
                 bert_tokenizer, pad_idx, start_idx, channel, max_length):
    """Decode a batch using the BERT encoder."""
    input_ids = sents['input_ids'].to(device)
    attention_mask = sents['attention_mask'].to(device)

    out = greedy_decode(net, input_ids, noise_std, max_length, pad_idx,
                        start_idx, channel, attention_mask=attention_mask)

    # Decode BERT token IDs → strings (skip special tokens)
    predicted = bert_tokenizer.batch_decode(out.cpu().tolist(),
                                            skip_special_tokens=True)
    references = bert_tokenizer.batch_decode(input_ids.cpu().tolist(),
                                             skip_special_tokens=True)
    return predicted, references

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(10)
    SNR = [0, 3, 6, 9, 12, 15, 18]

    # -----------------------------------------------------------------------
    # 1. Locate and load the latest checkpoint FIRST — before building the
    #    model — so we can infer the exact architecture used during training.
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

    # Try safe weights-only load; fall back for legacy full-model serialisation.
    try:
        raw_ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        raw_ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Normalise to a plain state_dict regardless of how it was saved.
    if isinstance(raw_ckpt, DeepSC):
        ckpt_state = raw_ckpt.state_dict()
    elif isinstance(raw_ckpt, dict):
        ckpt_state = raw_ckpt
    else:
        raise TypeError(f'Unexpected checkpoint type: {type(raw_ckpt)}')

    # Infer encoder / decoder layer counts from the checkpoint keys.
    enc_layers_in_ckpt = len({k.split('.')[2] for k in ckpt_state
                               if k.startswith('encoder.enc_layers.')})
    dec_layers_in_ckpt = len({k.split('.')[2] for k in ckpt_state
                               if k.startswith('decoder.dec_layers.')})

    if enc_layers_in_ckpt or dec_layers_in_ckpt:
        _arg_enc = args.num_layers if args.num_layers is not None else args.num_enc_layers
        _arg_dec = args.num_layers if args.num_layers is not None else args.num_dec_layers
        enc_num_layers = enc_layers_in_ckpt if enc_layers_in_ckpt else _arg_enc
        dec_num_layers = dec_layers_in_ckpt if dec_layers_in_ckpt else _arg_dec
        if enc_num_layers != _arg_enc or dec_num_layers != _arg_dec:
            print(f'[INFO] Checkpoint has {enc_num_layers} encoder layer(s) and '
                  f'{dec_num_layers} decoder layer(s); '
                  f'overriding CLI args.')
    else:
        # BERT encoder path: no enc_layers keys — use args for both.
        enc_num_layers = args.num_layers if args.num_layers is not None else args.num_enc_layers
        dec_num_layers = args.num_layers if args.num_layers is not None else args.num_dec_layers

    # -----------------------------------------------------------------------
    # 2. Build model with the inferred architecture, then load weights.
    # -----------------------------------------------------------------------
    if args.use_bert_encoder:
        from transformers import BertTokenizerFast
        print(f'Loading BERT tokenizer: {args.bert_model_name}')
        bert_tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        bert_vocab_size = bert_tokenizer.vocab_size

        pad_idx   = bert_tokenizer.pad_token_id    # 0
        start_idx = bert_tokenizer.cls_token_id    # 101
        end_idx   = bert_tokenizer.sep_token_id    # 102

        # Custom vocab still needed to reverse the existing pkl -> raw text
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

        decode_fn = partial(
            _decode_bert,
            bert_tokenizer=bert_tokenizer,
            pad_idx=pad_idx,
            start_idx=start_idx,
            channel=args.channel,
            max_length=args.MAX_LENGTH,
        )

        deepsc = DeepSC(
            dec_num_layers,
            src_vocab_size=bert_vocab_size,
            trg_vocab_size=bert_vocab_size,
            src_max_len=args.MAX_LENGTH,
            trg_max_len=args.MAX_LENGTH,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dff=args.dff,
            dropout=0.1,
            use_bert_encoder=True,
            bert_model_name=args.bert_model_name,
            freeze_bert=True,  # always frozen at eval time
        ).to(device)

    else:
        vocab = json.load(open(args.vocab_file, 'rb'))
        token_to_idx = vocab['token_to_idx']
        idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
        num_vocab  = len(token_to_idx)
        pad_idx    = token_to_idx['<PAD>']
        start_idx  = token_to_idx['<START>']
        end_idx    = token_to_idx['<END>']

        StoT = SeqtoText(token_to_idx, end_idx)
        collate_fn = collate_data

        decode_fn = partial(
            _decode_custom,
            StoT=StoT,
            pad_idx=pad_idx,
            start_idx=start_idx,
            channel=args.channel,
            max_length=args.MAX_LENGTH,
        )

        # DeepSC constructor uses num_layers for both encoder and decoder, so
        # we build it with enc_num_layers and then patch the decoder if needed.
        deepsc = DeepSC(enc_num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1).to(device)

        if dec_num_layers != enc_num_layers:
            # Rebuild decoder with the correct depth.
            from models.transceiver import Decoder
            deepsc.decoder = Decoder(dec_num_layers, num_vocab, num_vocab,
                                     args.d_model, args.num_heads, args.dff, 0.1).to(device)

    # Load weights into the freshly constructed model.
    deepsc.load_state_dict(ckpt_state)
    print('Model loaded from:', model_path)

    # -----------------------------------------------------------------------
    # 3. Optionally build BERTScore evaluator.
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
        args, SNR, deepsc, collate_fn, decode_fn, bert_scorer=bert_scorer
    )

    # Dynamic column width based on optional BERT columns
    sep_width = 10 + 4 * 11 + (3 * 11 if bert_scores else 0)  # approx separator width

    print('\n' + '='*sep_width)
    print(' Results for ' + args.channel + ' Channel using' + args.checkpoint_path + ' model')
    print('='*sep_width)
    print(f'{"SNR (dB)":>10} | {"BLEU-1":>8} | {"BLEU-2":>8} | {"BLEU-3":>8} | {"BLEU-4":>8}', end='')
    if bert_scores:
        print(f' | {"BERT-F1":>8} | {"BERT-P":>8} | {"BERT-R":>8}', end='')
    print()
    print('-'*sep_width)
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

    # -----------------------------------------------------------------------
    # Save aggregate metrics to CSV (one row per SNR point).
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Save per-sentence predictions to a second CSV
    # (one row per test sentence × SNR × epoch).
    # -----------------------------------------------------------------------
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