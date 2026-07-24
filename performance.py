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
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText, Channels, PowerNormalize, subsequent_mask
from tqdm import tqdm
import time
import random
parser = argparse.ArgumentParser()
#data-dir is unused, kept for testing purposes
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
parser.add_argument('--warmup-batches', default=2, type=int,
                    help='Number of warmup batches before measuring inference time')
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
                import bert_score.utils
                from bert_score import BERTScorer
            except ImportError:
                raise ImportError(
                    'bert-score is not installed. '
                    'Run: uv pip install bert-score'
                )

            # Patch bert_score.utils.sent_encode for transformers compatibility
            # where BertTokenizer lacks build_inputs_with_special_tokens on empty inputs
            if not getattr(bert_score.utils, '_sent_encode_patched', False):
                _orig_sent_encode = bert_score.utils.sent_encode
                def _safe_sent_encode(tokenizer, sent):
                    if sent.strip() == '':
                        return tokenizer.encode('', add_special_tokens=True)
                    return _orig_sent_encode(tokenizer, sent)
                bert_score.utils.sent_encode = _safe_sent_encode
                bert_score.utils._sent_encode_patched = True

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
        if not predictions or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        P, R, F1 = scorer.score(predictions, references)
        return {
            'precision': P.mean().item(),
            'recall':    R.mean().item(),
            'f1':        F1.mean().item(),
        }


def print_timing_report(pipeline_timings: dict, detailed_breakdown: dict, total_sentences: int):
    """Print out the time taken and bottleneck for each step in decoding and pipeline execution."""
    print('\n' + '='*80)
    print(' TIMING AND BOTTLENECK ANALYSIS REPORT')
    print('='*80)

    total_decode_time = sum(detailed_breakdown.values())
    if total_decode_time > 0 and total_sentences > 0:
        bottleneck_step = max(detailed_breakdown, key=detailed_breakdown.get)
        step_names = {
            'data_prep':        '1. GPU Data Prep & Transfer',
            'encoder':          '2. Encoder & Power Normalization',
            'channel_sim':      '3. Physical Channel Simulation',
            'channel_dec':      '4. Channel Decoder',
            'ar_loop':          '5. Autoregressive Decoder Loop',
            'text_postprocess': '6. Text Post-Processing (Tokenizer)',
        }

        print('\n[A] Model Inference & Decoding Sub-Step Breakdown:')
        print(f'{"Sub-Step":<38} | {"Total (s)":>10} | {"Per-Sent (ms)":>14} | {"Share (%)":>10} | {"Status":<12}')
        print('-'*92)

        for key, name in step_names.items():
            t_sec = detailed_breakdown.get(key, 0.0)
            t_ms_per_sent = (t_sec / total_sentences) * 1000.0
            pct = (t_sec / total_decode_time) * 100.0 if total_decode_time > 0 else 0.0
            is_bottleneck = (key == bottleneck_step)
            status = '[BOTTLENECK]' if is_bottleneck else ''
            print(f'{name:<38} | {t_sec:>10.4f} | {t_ms_per_sent:>14.4f} | {pct:>9.2f}% | {status:<12}')

        print('-'*92)
        print(f'{"Total Decoding Time":<38} | {total_decode_time:>10.4f} | {(total_decode_time/total_sentences)*1000.0:>14.4f} | {"100.00%":>10} |')
        print(f'-> Inference Bottleneck: {step_names[bottleneck_step]} ({detailed_breakdown[bottleneck_step]/total_decode_time*100:.2f}% of total decoding time)')

    total_pipeline_time = sum(pipeline_timings.values())
    if total_pipeline_time > 0:
        pipeline_bottleneck = max(pipeline_timings, key=pipeline_timings.get)

        print('\n[B] End-to-End Execution Pipeline Step Breakdown:')
        print(f'{"Pipeline Step":<38} | {"Time (s)":>10} | {"Share (%)":>10} | {"Status":<12}')
        print('-'*76)

        for name, t_sec in pipeline_timings.items():
            pct = (t_sec / total_pipeline_time) * 100.0 if total_pipeline_time > 0 else 0.0
            is_bottleneck = (name == pipeline_bottleneck)
            status = '[BOTTLENECK]' if is_bottleneck else ''
            print(f'{name:<38} | {t_sec:>10.4f} | {pct:>9.2f}% | {status:<12}')

        print('-'*76)
        print(f'{"Total Execution Time":<38} | {total_pipeline_time:>10.4f} | {"100.00%":>10} |')
        print(f'-> Overall Pipeline Bottleneck: {pipeline_bottleneck} ({pipeline_timings[pipeline_bottleneck]/total_pipeline_time*100:.2f}% of total execution time)')

    print('='*80 + '\n')


def performance(args, SNR, net, collate_fn, decode_fn, bert_scorer=None):
    """Evaluate net over a range of SNR values.

    Args:
        args: parsed arguments.
        SNR: list of SNR values (dB) to evaluate at.
        net: the DeepSC model.
        collate_fn: DataLoader collate function (standard or BERT).
        decode_fn: callable(net, sents, noise_std) -> (predicted, references, timing_info).
        bert_scorer: optional ``BertScoreEvaluator`` instance. When provided,
            BERTScore F1/P/R are computed alongside BLEU for each SNR point.

    Returns:
        bleu_scores: dict with keys 'bleu1'..'bleu4', each an np.ndarray of
            shape [len(SNR)] containing the mean n-gram BLEU for that order.
        bert_scores: dict with keys 'f1', 'precision', 'recall', each
            np.ndarray of shape [len(SNR)].  Empty dict if bert_scorer is None.
        time_stats: dict with keys 'latency_ms', 'model_latency_ms', 'throughput_sps',
            each an np.ndarray of shape [len(SNR)].
        all_sentences: list of dicts, one per sentence, with keys
            'epoch', 'snr_db', 'sample_idx', 'predicted', 'reference'.
        accumulated_breakdown: dict mapping decoding sub-step name to total time in seconds.
        eval_step_timings: dict mapping metric evaluation step name to total time in seconds.
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

    all_latency_ms       = []
    all_model_latency_ms = []
    all_throughput_sps   = []

    accumulated_breakdown = {
        'data_prep': 0.0,
        'encoder': 0.0,
        'channel_sim': 0.0,
        'channel_dec': 0.0,
        'ar_loop': 0.0,
        'text_postprocess': 0.0,
    }
    eval_step_timings = {
        'bleu_evaluation': 0.0,
        'bert_evaluation': 0.0,
    }

    # Optional Warmup
    warmup_cnt = getattr(args, 'warmup_batches', 0)
    if warmup_cnt > 0:
        net.eval()
        with torch.no_grad():
            warmup_snr = SNR[0] if len(SNR) > 0 else 0
            noise_std  = SNR_to_noise(warmup_snr)
            for i, sents in enumerate(test_iterator):
                if i >= warmup_cnt:
                    break
                decode_fn(net, sents, noise_std)

    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            epoch_latency_ms       = []
            epoch_model_latency_ms = []
            epoch_throughput_sps   = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                snr_total_time       = 0.0
                snr_model_total_time = 0.0
                snr_total_samples    = 0

                for sents in test_iterator:
                    predicted, references, t_info = decode_fn(net, sents, noise_std)
                    word += predicted
                    target_word += references

                    snr_total_time       += t_info['total_time']
                    snr_model_total_time += t_info['model_time']
                    snr_total_samples    += t_info['batch_size']

                    if 'breakdown' in t_info:
                        for k, v in t_info['breakdown'].items():
                            accumulated_breakdown[k] = accumulated_breakdown.get(k, 0.0) + v

                Tx_word.append(word)
                Rx_word.append(target_word)

                if snr_total_samples > 0 and snr_total_time > 0:
                    lat_ms     = (snr_total_time / snr_total_samples) * 1000.0
                    mod_lat_ms = (snr_model_total_time / snr_total_samples) * 1000.0
                    tput_sps   = snr_total_samples / snr_total_time
                else:
                    lat_ms, mod_lat_ms, tput_sps = 0.0, 0.0, 0.0

                epoch_latency_ms.append(lat_ms)
                epoch_model_latency_ms.append(mod_lat_ms)
                epoch_throughput_sps.append(tput_sps)

                # Collect per-sentence records for this SNR.
                for sample_idx, (pred, ref) in enumerate(zip(word, target_word)):
                    all_sentences.append({
                        'epoch':      epoch,
                        'snr_db':     snr,
                        'sample_idx': sample_idx,
                        'predicted':  pred,
                        'reference':  ref,
                    })

            all_latency_ms.append(epoch_latency_ms)
            all_model_latency_ms.append(epoch_model_latency_ms)
            all_throughput_sps.append(epoch_throughput_sps)

            bleu_epoch = {k: [] for k in bleu_scorers}
            bert_f1_epoch, bert_p_epoch, bert_r_epoch = [], [], []

            for snr_idx, (sent1, sent2) in enumerate(zip(Tx_word, Rx_word)):
                print(f"\n" + "="*80)
                print(f" SNR: {SNR[snr_idx]} dB | Sample Comparisons")
                print(f" Average Latency: {epoch_latency_ms[snr_idx]:.3f} ms/sent (Model: {epoch_model_latency_ms[snr_idx]:.3f} ms/sent) | Throughput: {epoch_throughput_sps[snr_idx]:.1f} sent/s")
                print(f"="*80)
                for pred, ref in zip(sent1[:5], sent2[:5]):
                    print(f"Predicted: {pred}")
                    print(f"Actual   : {ref}")
                    print("-"*40)

                # BLEU-1 through BLEU-4
                t_bleu_st = time.perf_counter()
                for key, scorer in bleu_scorers.items():
                    bleu_epoch[key].append(scorer.compute_blue_score(sent1, sent2))
                eval_step_timings['bleu_evaluation'] += (time.perf_counter() - t_bleu_st)

                # BERTScore (optional)
                if bert_scorer is not None:
                    t_bert_st = time.perf_counter()
                    bs = bert_scorer.score(sent1, sent2)
                    eval_step_timings['bert_evaluation'] += (time.perf_counter() - t_bert_st)
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

    time_stats = {
        'latency_ms':       np.mean(np.array(all_latency_ms), axis=0),
        'model_latency_ms': np.mean(np.array(all_model_latency_ms), axis=0),
        'throughput_sps':   np.mean(np.array(all_throughput_sps), axis=0),
    }

    return bleu_scores, bert_scores, time_stats, all_sentences, accumulated_breakdown, eval_step_timings


# ---------------------------------------------------------------------------
# Decode helpers — one per encoder type
# ---------------------------------------------------------------------------

def greedy_decode_timed(model, src, n_var, max_len, padding_idx, start_symbol, channel,
                        attention_mask=None):
    """Greedy (argmax) decoder with per-step timing breakdown.

    Steps timed:
    1. Data Prep & GPU Transfer
    2. Encoder & Power Normalization
    3. Physical Channel Simulation
    4. Channel Decoder
    5. Autoregressive Decoder Loop
    """
    timing_breakdown = {
        'data_prep': 0.0,
        'encoder': 0.0,
        'channel_sim': 0.0,
        'channel_dec': 0.0,
        'ar_loop': 0.0,
    }
    channels = Channels()

    # --- Step 1: GPU Data Prep & Transfer ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    if model.use_bert_encoder:
        src_input_ids = src.to(device)
        src_attn_mask = attention_mask.to(device) if attention_mask is not None else None
        src_mask = None
        if src_attn_mask is not None:
            src_mask = (1 - src_attn_mask).unsqueeze(-2).type(torch.FloatTensor).to(device)
    else:
        src_input_ids = src.to(device)
        src_mask = (src_input_ids == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    timing_breakdown['data_prep'] = t1 - t0

    # --- Step 2: Encoder & Power Normalization ---
    if model.use_bert_encoder:
        enc_output = model.encoder(src_input_ids, attention_mask=src_attn_mask)
    else:
        enc_output = model.encoder(src_input_ids, src_mask)

    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    timing_breakdown['encoder'] = t2 - t1

    # --- Step 3: Physical Channel Simulation ---
    if channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from AWGN, Rayleigh, and Rician")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    timing_breakdown['channel_sim'] = t3 - t2

    # --- Step 4: Channel Decoder ---
    memory = model.channel_decoder(Rx_sig)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t4 = time.perf_counter()
    timing_breakdown['channel_dec'] = t4 - t3

    # --- Step 5: Autoregressive Decoder Loop ---
    outputs = torch.ones(src_input_ids.size(0), 1).fill_(start_symbol).type_as(src_input_ids.data)

    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        dec_output = model.decoder(outputs, memory, combined_mask, src_mask)
        pred = model.dense(dec_output)

        prob = pred[:, -1:, :]
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t5 = time.perf_counter()
    timing_breakdown['ar_loop'] = t5 - t4

    return outputs, timing_breakdown


def _decode_custom(net, sents, noise_std,
                   StoT, pad_idx, start_idx, channel, max_length):
    """Decode a batch using the custom-vocab encoder."""
    B = sents.size(0)

    out, breakdown = greedy_decode_timed(
        net, sents, noise_std, max_length, pad_idx, start_idx, channel
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_post_start = time.perf_counter()

    sentences = out.cpu().numpy().tolist()
    predicted = list(map(StoT.sequence_to_text, sentences))
    target_sent = sents.cpu().numpy().tolist() if isinstance(sents, torch.Tensor) else sents
    references = list(map(StoT.sequence_to_text, target_sent))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_post_end = time.perf_counter()
    breakdown['text_postprocess'] = t_post_end - t_post_start

    model_time = (breakdown['encoder'] + breakdown['channel_sim'] +
                  breakdown['channel_dec'] + breakdown['ar_loop'])
    total_time = sum(breakdown.values())

    timing_info = {
        'model_time': model_time,
        'total_time': total_time,
        'batch_size': B,
        'breakdown': breakdown,
    }
    return predicted, references, timing_info


def _decode_bert(net, sents, noise_std,
                 bert_tokenizer, pad_idx, start_idx, channel, max_length):
    """Decode a batch using the BERT encoder."""
    input_ids = sents['input_ids']
    attention_mask = sents['attention_mask']
    B = input_ids.size(0)

    out, breakdown = greedy_decode_timed(
        net, input_ids, noise_std, max_length, pad_idx, start_idx, channel,
        attention_mask=attention_mask
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_post_start = time.perf_counter()

    predicted = bert_tokenizer.batch_decode(out.cpu().tolist(),
                                            skip_special_tokens=True)
    references = bert_tokenizer.batch_decode(input_ids.cpu().tolist(),
                                             skip_special_tokens=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_post_end = time.perf_counter()
    breakdown['text_postprocess'] = t_post_end - t_post_start

    model_time = (breakdown['encoder'] + breakdown['channel_sim'] +
                  breakdown['channel_dec'] + breakdown['ar_loop'])
    total_time = sum(breakdown.values())

    timing_info = {
        'model_time': model_time,
        'total_time': total_time,
        'batch_size': B,
        'breakdown': breakdown,
    }
    return predicted, references, timing_info

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

    pipeline_timings = {}

    # -----------------------------------------------------------------------
    # 1. Locate and load the latest checkpoint FIRST — before building the
    #    model — so we can infer the exact architecture used during training.
    # -----------------------------------------------------------------------
    t_step1 = time.perf_counter()
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

    pipeline_timings['1. Checkpoint Loading & Inspection'] = time.perf_counter() - t_step1

    # -----------------------------------------------------------------------
    # 2. Build model with the inferred architecture, then load weights.
    # -----------------------------------------------------------------------
    t_step2 = time.perf_counter()
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
    pipeline_timings['2. Model Construction & Weight Load'] = time.perf_counter() - t_step2

    # -----------------------------------------------------------------------
    # 3. Optionally build BERTScore evaluator.
    # -----------------------------------------------------------------------
    t_step3 = time.perf_counter()
    bert_scorer = None
    if args.bert_score:
        print(f'Initialising BERTScore evaluator (model: {args.bert_score_model})...')
        bert_scorer = BertScoreEvaluator(
            model_type=args.bert_score_model,
            lang='en',
            batch_size=args.batch_size,
        )
    pipeline_timings['3. BERTScore Evaluator Init'] = time.perf_counter() - t_step3

    bleu_scores, bert_scores, time_stats, all_sentences, accumulated_breakdown, eval_step_timings = performance(
        args, SNR, deepsc, collate_fn, decode_fn, bert_scorer=bert_scorer
    )

    pipeline_timings['4. Model Decoding (Inference)'] = sum(accumulated_breakdown.values())
    pipeline_timings['5. BLEU Metric Evaluation']    = eval_step_timings['bleu_evaluation']
    if args.bert_score:
        pipeline_timings['6. BERTScore Evaluation']    = eval_step_timings['bert_evaluation']

    # Dynamic column width based on optional BERT columns
    sep_width = 10 + 4 * 11 + 3 * 16 + (3 * 11 if bert_scores else 0)

    print('\n' + '='*sep_width)
    print(' Results for ' + args.channel + ' Channel using ' + args.checkpoint_path + ' model')
    print('='*sep_width)
    print(f'{"SNR (dB)":>10} | {"BLEU-1":>8} | {"BLEU-2":>8} | {"BLEU-3":>8} | {"BLEU-4":>8}', end='')
    if bert_scores:
        print(f' | {"BERT-F1":>8} | {"BERT-P":>8} | {"BERT-R":>8}', end='')
    print(f' | {"Latency(ms)":>12} | {"Model(ms)":>12} | {"Throughput(sps)":>15}')
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
        row += (f' | {time_stats["latency_ms"][i]:>12.3f}'
                f' | {time_stats["model_latency_ms"][i]:>12.3f}'
                f' | {time_stats["throughput_sps"][i]:>15.1f}')
        print(row)

    print('-'*sep_width)
    row_avg = (f'{"Average":>10}'
               f' | {np.mean(bleu_scores["bleu1"]):>8.4f}'
               f' | {np.mean(bleu_scores["bleu2"]):>8.4f}'
               f' | {np.mean(bleu_scores["bleu3"]):>8.4f}'
               f' | {np.mean(bleu_scores["bleu4"]):>8.4f}')
    if bert_scores:
        row_avg += (f' | {np.mean(bert_scores["f1"]):>8.4f}'
                    f' | {np.mean(bert_scores["precision"]):>8.4f}'
                    f' | {np.mean(bert_scores["recall"]):>8.4f}')
    row_avg += (f' | {np.mean(time_stats["latency_ms"]):>12.3f}'
                f' | {np.mean(time_stats["model_latency_ms"]):>12.3f}'
                f' | {np.mean(time_stats["throughput_sps"]):>15.1f}')
    print(row_avg)

    model_tag = os.path.basename(args.checkpoint_path.rstrip('/\\'))

    # -----------------------------------------------------------------------
    # Save aggregate metrics to CSV (one row per SNR point).
    # -----------------------------------------------------------------------
    t_step_csv = time.perf_counter()
    csv_path = (args.output_csv
                if args.output_csv
                else os.path.join(args.checkpoint_path,
                                  f'results_{args.channel}_{model_tag}.csv'))

    fieldnames = ['channel', 'checkpoint', 'snr_db',
                  'bleu1', 'bleu2', 'bleu3', 'bleu4']
    if bert_scores:
        fieldnames += ['bert_f1', 'bert_precision', 'bert_recall']
    fieldnames += ['latency_ms_per_sent', 'model_latency_ms_per_sent', 'throughput_sent_per_sec']

    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for i, snr in enumerate(SNR):
            record = {
                'channel':                   args.channel,
                'checkpoint':                model_tag,
                'snr_db':                    snr,
                'bleu1':                     round(float(bleu_scores['bleu1'][i]), 6),
                'bleu2':                     round(float(bleu_scores['bleu2'][i]), 6),
                'bleu3':                     round(float(bleu_scores['bleu3'][i]), 6),
                'bleu4':                     round(float(bleu_scores['bleu4'][i]), 6),
                'latency_ms_per_sent':       round(float(time_stats['latency_ms'][i]), 4),
                'model_latency_ms_per_sent': round(float(time_stats['model_latency_ms'][i]), 4),
                'throughput_sent_per_sec':   round(float(time_stats['throughput_sps'][i]), 2),
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
    pipeline_timings['7. Results Export to CSV'] = time.perf_counter() - t_step_csv

    # Print out comprehensive timing and bottleneck report
    total_sentences_processed = len(all_sentences)
    print_timing_report(pipeline_timings, accumulated_breakdown, total_sentences_processed)