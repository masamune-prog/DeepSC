# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from functools import partial
from dataset import EurDataset, collate_data, collate_data_bert
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, diffusion_decode, SeqtoText
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
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int)
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
parser.add_argument('--use-diffusion-decoder', action='store_true',
                    help='Use the diffusion decoder variant (must match checkpoint)')
parser.add_argument('--diff-steps', default=100, type=int,
                    help='Total DDPM forward-process timesteps T')
parser.add_argument('--diff-sampling-steps', default=50, type=int,
                    help='Number of DDIM reverse steps at inference')

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

    return bleu_scores, bert_scores



# ---------------------------------------------------------------------------
# Decode helpers — one per encoder type
# ---------------------------------------------------------------------------

def _decode_custom(net, sents, noise_std,
                   StoT, pad_idx, start_idx, channel, max_length):
    """Decode a batch using the custom-vocab encoder (Transformer decoder)."""
    sents = sents.to(device)
    out = greedy_decode(net, sents, noise_std, max_length, pad_idx,
                        start_idx, channel)
    sentences = out.cpu().numpy().tolist()
    predicted = list(map(StoT.sequence_to_text, sentences))
    target_sent = sents.cpu().numpy().tolist()
    references = list(map(StoT.sequence_to_text, target_sent))
    return predicted, references


def _decode_custom_diffusion(net, sents, noise_std,
                              StoT, pad_idx, channel, max_length):
    """Decode a batch using the custom-vocab encoder + diffusion decoder (DDIM)."""
    sents = sents.to(device)
    out = diffusion_decode(net, sents, noise_std, max_length, pad_idx, channel)
    sentences = out.cpu().numpy().tolist()
    predicted = list(map(StoT.sequence_to_text, sentences))
    target_sent = sents.cpu().numpy().tolist()
    references = list(map(StoT.sequence_to_text, target_sent))
    return predicted, references


def _decode_bert(net, sents, noise_std,
                 bert_tokenizer, pad_idx, start_idx, channel, max_length):
    """Decode a batch using the BERT encoder (Transformer decoder)."""
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


def _decode_bert_diffusion(net, sents, noise_std,
                            bert_tokenizer, pad_idx, channel, max_length):
    """Decode a batch using the BERT encoder + diffusion decoder (DDIM)."""
    input_ids = sents['input_ids'].to(device)
    attention_mask = sents['attention_mask'].to(device)

    out = diffusion_decode(net, input_ids, noise_std, max_length, pad_idx,
                           channel, attention_mask=attention_mask)

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

    if args.use_bert_encoder:
        from transformers import BertTokenizerFast
        print(f'Loading BERT tokenizer: {args.bert_model_name}')
        bert_tokenizer = BertTokenizerFast.from_pretrained(args.bert_model_name)
        bert_vocab_size = bert_tokenizer.vocab_size

        pad_idx = bert_tokenizer.pad_token_id      # 0
        start_idx = bert_tokenizer.cls_token_id    # 101
        end_idx = bert_tokenizer.sep_token_id      # 102

        # Custom vocab still needed to reverse the existing pkl → raw text
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
            _decode_bert_diffusion if args.use_diffusion_decoder else _decode_bert,
            bert_tokenizer=bert_tokenizer,
            pad_idx=pad_idx,
            **({} if args.use_diffusion_decoder else {'start_idx': start_idx}),
            channel=args.channel,
            max_length=args.MAX_LENGTH,
        )

        deepsc = DeepSC(
            args.num_layers,
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
            use_diffusion_decoder=args.use_diffusion_decoder,
            diff_steps=args.diff_steps,
            diff_sampling_steps=args.diff_sampling_steps,
        ).to(device)

    else:
        vocab = json.load(open(args.vocab_file, 'rb'))
        token_to_idx = vocab['token_to_idx']
        idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
        num_vocab = len(token_to_idx)
        pad_idx = token_to_idx['<PAD>']
        start_idx = token_to_idx['<START>']
        end_idx = token_to_idx['<END>']

        StoT = SeqtoText(token_to_idx, end_idx)
        collate_fn = collate_data

        if args.use_diffusion_decoder:
            decode_fn = partial(
                _decode_custom_diffusion,
                StoT=StoT,
                pad_idx=pad_idx,
                channel=args.channel,
                max_length=args.MAX_LENGTH,
            )
        else:
            decode_fn = partial(
                _decode_custom,
                StoT=StoT,
                pad_idx=pad_idx,
                start_idx=start_idx,
                channel=args.channel,
                max_length=args.MAX_LENGTH,
            )

        deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                        num_vocab, num_vocab, args.d_model, args.num_heads,
                        args.dff, 0.1,
                        use_diffusion_decoder=args.use_diffusion_decoder,
                        diff_steps=args.diff_steps,
                        diff_sampling_steps=args.diff_sampling_steps).to(device)

    # Load latest checkpoint
    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])
    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path, map_location=device)

    # --- auto-recover architecture config from the checkpoint directory ------
    _cfg_path = os.path.join(args.checkpoint_path, 'config.json')
    if os.path.exists(_cfg_path):
        import json as _json
        with open(_cfg_path) as _f:
            _cfg = _json.load(_f)
        # Only override if not explicitly set by the user on the command line
        if _cfg.get('use_diffusion_decoder') and not args.use_diffusion_decoder:
            print('[INFO] config.json: enabling --use-diffusion-decoder automatically.')
            args.use_diffusion_decoder = True
        if 'diff_steps' in _cfg:
            args.diff_steps = _cfg['diff_steps']
        if 'diff_sampling_steps' in _cfg:
            args.diff_sampling_steps = _cfg['diff_sampling_steps']
        print(f'[INFO] Loaded architecture config from {_cfg_path}')

    # strict=False: tolerates mismatches between decoder types (Transformer vs.
    # diffusion).  Shared weights (encoder, channel_encoder, channel_decoder,
    # dense) always load correctly.  Decoder-specific mismatches are reported.
    result = deepsc.load_state_dict(checkpoint, strict=False)
    print('Model loaded from:', model_path)
    if result.missing_keys:
        print(f'  [WARNING] {len(result.missing_keys)} weight(s) absent in checkpoint '
              f'— randomly initialised:')
        for k in result.missing_keys[:5]:
            print(f'    - {k}')
        if len(result.missing_keys) > 5:
            print(f'    ... and {len(result.missing_keys) - 5} more.')
    if result.unexpected_keys:
        print(f'  [INFO]    {len(result.unexpected_keys)} checkpoint weight(s) not used '
              f'by current model — skipped:')
        for k in result.unexpected_keys[:5]:
            print(f'    - {k}')
        if len(result.unexpected_keys) > 5:
            print(f'    ... and {len(result.unexpected_keys) - 5} more.')

    # Optionally build BERTScore evaluator
    bert_scorer = None
    if args.bert_score:
        print(f'Initialising BERTScore evaluator (model: {args.bert_score_model})...')
        bert_scorer = BertScoreEvaluator(
            model_type=args.bert_score_model,
            lang='en',
            batch_size=args.batch_size,
        )

    bleu_scores, bert_scores = performance(
        args, SNR, deepsc, collate_fn, decode_fn, bert_scorer=bert_scorer
    )

    # Dynamic column width based on optional BERT columns
    bert_cols = f' | {"BERT-F1":>8} | {"BERT-P":>8} | {"BERT-R":>8}' if bert_scores else ''
    sep_width = 10 + 4 * 11 + (3 * 11 if bert_scores else 0)  # approx separator width

    print('\n' + '='*sep_width)
    print(' Results')
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
