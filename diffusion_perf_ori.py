# -*- coding: utf-8 -*-
"""diffusion_performance.py (fixed)

Evaluates a trained ChannelDiffusionLM checkpoint over a range of SNR values,
mirroring the methodology in performance.py.

Fixes applied vs. the original script, each tagged [FIX n] at the site of the change,
kept consistent with the fixes made to diffusion_attempt.py so a checkpoint trained
with the fixed training script is evaluated the way it was actually trained:

  1. Channels.Rayleigh / Rician previously drew a single fading coefficient H shared
     by the WHOLE batch (`size=[1]`). Changed to one H per example (`size=[B]`),
     matching the training script — otherwise a single unlucky batch-wide draw can
     corrupt every sentence in that batch/SNR step at once, and the eval channel
     distribution doesn't match what the model was trained on.
  2. Same raw torch.inverse(H) zero-forcing issue as training: no floor, so a
     near-singular H blows up into inf/NaN. Added the same ridge-regularized
     inverse used in the fixed training script.
  3. decode_batch() hardcoded t_denoise=1 regardless of the actual channel noise.
     That was a reasonable workaround against the ORIGINAL training script (where t
     was sampled independently of the real noise anyway), but the fixed training
     script now derives t from the real noise via sigma_to_t, so the model expects
     t to track actual corruption strength. Now derives t_denoise from noise_std via
     sigma_to_t, mirroring training.
  4. PAD_ID silently fell back to 0 if '<PAD>' wasn't found in the vocab. Now raises
     a clear error instead.
  5. Both output CSVs were opened in append mode unconditionally, so re-running the
     same channel/checkpoint after a fix (like this one) silently doubles up rows
     rather than overwriting. Now overwrites each run.

  - Uses the SAME fixed test split (txt/test_data.pkl) as performance.py
  - BLEU-1 through BLEU-4 (sentence-level)
  - Optional BERTScore F1 / Precision / Recall
  - Aggregate results saved to CSV
  - Per-sentence predictions saved to a second CSV
"""

import os
import csv
import time
import math
import json
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Re-use BLEU helpers from the existing DeepSC utils
# ---------------------------------------------------------------------------
from utils import BleuScore

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Evaluate ChannelDiffusionLM performance")
parser.add_argument('--checkpoint-path', default='channel_diffusion_ckpt.pt', type=str,
                    help='Path to the saved ChannelDiffusionLM checkpoint (.pt file)')
parser.add_argument('--vocab-file', default='txt/diffusion_vocab.json', type=str,
                    help='Path to the diffusion tokenizer JSON (saved by diffusion_attempt.py)')
parser.add_argument('--deepsc-vocab-file', default='txt/vocab.json', type=str,
                    help='Path to the custom DeepSC vocab.json used to decode txt/test_data.pkl')
parser.add_argument('--test-pkl', default='txt/test_data.pkl', type=str,
                    help='Pre-built test pickle produced by the DeepSC preprocessing pipeline '
                         '(same file used by performance.py). '
                         'Each row is a list of custom-vocab integer IDs.')
parser.add_argument('--channel', default='AWGN', type=str,
                    choices=['AWGN', 'Rayleigh', 'Rician', 'None'],
                    help='Physical channel type used during evaluation')
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type=int,
                    help='Number of evaluation sweeps over the test set')
parser.add_argument('--warmup-batches', default=1, type=int,
                    help='Number of warmup batches before measuring inference time')
parser.add_argument('--output-csv', default=None, type=str,
                    help='Path for the aggregate results CSV. Defaults to '
                         'diffusion_results_<channel>.csv in the checkpoint directory.')
parser.add_argument('--predictions-csv', default=None, type=str,
                    help='Path for the per-sentence predictions CSV.')
# BERTScore options (mirrors performance.py)
parser.add_argument('--bert-score', action='store_true',
                    help='Compute BERTScore semantic similarity in addition to BLEU')
parser.add_argument('--bert-score-model', default='bert-base-multilingual-cased', type=str,
                    help='HuggingFace model used by BERTScore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# BERTScore evaluator (identical to performance.py)
# ---------------------------------------------------------------------------

class BertScoreEvaluator:
    """Thin wrapper around the bert-score library."""

    def __init__(self, model_type: str = 'bert-base-multilingual-cased',
                 lang: str = 'en', batch_size: int = 64):
        self.model_type = model_type
        self.lang = lang
        self.batch_size = batch_size
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            try:
                import bert_score.utils
                from bert_score import BERTScorer
            except ImportError:
                raise ImportError(
                    'bert-score is not installed. Run: uv pip install bert-score'
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

            self._scorer = BERTScorer(
                model_type=self.model_type,
                lang=self.lang,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                batch_size=self.batch_size,
            )
        return self._scorer

    def score(self, predictions: list, references: list) -> dict:
        scorer = self._get_scorer()
        if not predictions or not references:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        P, R, F1 = scorer.score(predictions, references)
        return {
            'precision': P.mean().item(),
            'recall':    R.mean().item(),
            'f1':        F1.mean().item(),
        }


# ---------------------------------------------------------------------------
# Model definitions (self-contained copy from diffusion_attempt.py so this
# script can load checkpoints without importing the notebook file)
# ---------------------------------------------------------------------------

@dataclass
class ChannelDiffusionConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_enc_layers: int
    n_dec_layers: int
    n_heads: int
    d_ff: int
    dropout: float = 0.1
    diffusion_steps: int = 5
    sigma_min: float = 0.002
    sigma_max: float = 1.0
    channel_type: str = "AWGN"
    snr_gamma: float = 5.0
    n_snr_bins: int = 19          # discrete SNR bins covering 0..18 dB evaluation range


def SNR_to_noise(snr_db: float) -> float:
    """Convert SNR in dB to noise standard deviation (AWGN convention)."""
    snr_linear = 10 ** (snr_db / 10)
    return 1.0 / math.sqrt(2 * snr_linear)


class Channels:
    def AWGN(self, Tx_sig, n_var):
        return Tx_sig + torch.randn_like(Tx_sig) * n_var

    @staticmethod
    def _regularized_inverse(H, eps=1e-2):
        """[FIX 2] Ridge-regularized inverse: H^{-1} ~= (H^T H + eps*I)^{-1} H^T.

        Kept identical to the fixed training script's Channels._regularized_inverse
        so eval uses the same equalization behavior the model was trained against.
        A raw torch.inverse(H) blows up whenever a fading draw lands close to
        singular (a deep fade), injecting inf/NaN into z_t.
        """
        B = H.shape[0]
        I = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
        Ht = H.transpose(-1, -2)
        return torch.bmm(torch.inverse(torch.bmm(Ht, H) + eps * I), Ht)

    def Rayleigh(self, Tx_sig, n_var):
        # [FIX 1] One fading coefficient PER EXAMPLE (was: a single scalar H shared
        # by the entire batch), matching the fixed training script's Channels.Rayleigh.
        shape = Tx_sig.shape                       # [B, L, D]
        B, dev = shape[0], Tx_sig.device

        H_real = torch.normal(0, math.sqrt(1 / 2), size=(B,), device=dev)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=(B,), device=dev)
        H = torch.stack([
            torch.stack([H_real, -H_imag], dim=-1),
            torch.stack([H_imag, H_real], dim=-1),
        ], dim=-2)                                  # [B, 2, 2]

        Tx_pairs = Tx_sig.reshape(B, -1, 2)          # [B, L*D/2, 2]
        Tx_faded = torch.bmm(Tx_pairs, H)
        Rx_faded = self.AWGN(Tx_faded, n_var)
        H_inv = self._regularized_inverse(H)         # [FIX 2]
        Rx_sig = torch.bmm(Rx_faded, H_inv).reshape(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        # [FIX 1] Same per-example fix as Rayleigh above.
        shape = Tx_sig.shape
        B, dev = shape[0], Tx_sig.device
        mean = math.sqrt(K / (K + 1))
        std  = math.sqrt(1 / (K + 1))

        H_real = torch.normal(mean, std, size=(B,), device=dev)
        H_imag = torch.normal(mean, std, size=(B,), device=dev)
        H = torch.stack([
            torch.stack([H_real, -H_imag], dim=-1),
            torch.stack([H_imag, H_real], dim=-1),
        ], dim=-2)                                  # [B, 2, 2]

        Tx_pairs = Tx_sig.reshape(B, -1, 2)
        Tx_faded = torch.bmm(Tx_pairs, H)
        Rx_faded = self.AWGN(Tx_faded, n_var)
        H_inv = self._regularized_inverse(H)         # [FIX 2]
        Rx_sig = torch.bmm(Rx_faded, H_inv).reshape(shape)
        return Rx_sig

    def forward(self, Tx_sig, n_var, channel_type="AWGN", K=1):
        if channel_type == "AWGN":
            return self.AWGN(Tx_sig, n_var)
        elif channel_type == "Rayleigh":
            return self.Rayleigh(Tx_sig, n_var)
        elif channel_type == "Rician":
            return self.Rician(Tx_sig, n_var, K=K)
        elif channel_type is None or channel_type == "None":
            return Tx_sig
        else:
            raise ValueError(f"Unknown channel_type: {channel_type}")


def make_sigma_schedule(T, sigma_min=0.01, sigma_max=1.0, device="cpu"):
    steps = torch.arange(T + 1, device=device).float() / T
    sigmas = sigma_min * (sigma_max / sigma_min) ** steps
    return sigmas


class ChannelDiffusionLM(nn.Module):
    def __init__(self, cfg: ChannelDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.channels = Channels()

        self.tok_emb  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb  = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)
        self.mode_emb = nn.Embedding(2, cfg.d_model)
        self.snr_emb  = nn.Embedding(cfg.n_snr_bins, cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
            ), num_layers=cfg.n_enc_layers
        )
        self.z_ln = nn.LayerNorm(cfg.d_model)

        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
            ), num_layers=cfg.n_dec_layers
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        sigmas = make_sigma_schedule(cfg.diffusion_steps, cfg.sigma_min, cfg.sigma_max)
        self.register_buffer("sigmas", sigmas)

    def encode(self, input_ids, attention_mask=None):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        src_key_padding_mask = None if attention_mask is None else ~attention_mask
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.z_ln(x)

    def denoise_step(self, z_t, t, mode_idx=0, attention_mask=None, snr_bin=None):
        B, L, D = z_t.shape
        t_emb = self.time_emb(t).unsqueeze(1)
        modes = torch.full((B,), mode_idx, dtype=torch.long, device=z_t.device)
        m_emb = self.mode_emb(modes).unsqueeze(1)
        pos   = torch.arange(L, device=z_t.device).unsqueeze(0)
        p_emb = self.pos_emb(pos)
        x     = z_t + t_emb + m_emb + p_emb
        if snr_bin is not None:
            s_emb = self.snr_emb(snr_bin).unsqueeze(1)  # [B, 1, D]
            x = x + s_emb
        src_key_padding_mask = None if attention_mask is None else ~attention_mask
        x = self.denoiser(x, src_key_padding_mask=src_key_padding_mask)
        return self.ln_f(x)

    def forward(self, input_ids, t, mode_idx=0, attention_mask=None, channel_type=None,
                snr_bin=None):
        z0 = self.encode(input_ids, attention_mask)
        if mode_idx == 1:
            t_zero  = torch.zeros_like(t)
            z0_hat  = self.denoise_step(z0, t_zero, mode_idx=1, attention_mask=attention_mask,
                                        snr_bin=snr_bin)
        else:
            sigma_t = self.sigmas[t].view(-1, 1, 1)
            z_t     = self.channels.AWGN(z0, sigma_t)
            z0_hat  = self.denoise_step(z_t, t, mode_idx=0, attention_mask=attention_mask,
                                        snr_bin=snr_bin)
        logits = self.lm_head(z0_hat) / math.sqrt(self.cfg.d_model)
        return logits, z0, z0_hat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigma_to_t(model: ChannelDiffusionLM, sigma: float) -> int:
    """Invert the geometric noise schedule to find the matching discrete timestep.

    Kept identical to the fixed training script's sigma_to_t so eval derives t
    from the real channel noise the same way training does. [used by FIX 3]
    """
    T = model.cfg.diffusion_steps
    sigma_clamped = max(model.cfg.sigma_min, min(model.cfg.sigma_max, sigma))
    frac = math.log(sigma_clamped / model.cfg.sigma_min) / \
           math.log(model.cfg.sigma_max / model.cfg.sigma_min)
    return max(1, min(T, round(frac * T)))


def noise_to_snr_bin(noise_std: float, n_snr_bins: int = 19) -> int:
    """Convert a noise standard deviation to the nearest discrete SNR bin.

    Inverse of SNR_to_noise: computes SNR_dB from noise_std, then clips to
    the integer bin index [0, n_snr_bins-1] covering 0..18 dB.
    """
    snr_linear = 1.0 / max(2.0 * noise_std ** 2, 1e-8)
    snr_db     = 10.0 * math.log10(snr_linear)
    return max(0, min(n_snr_bins - 1, round(snr_db)))


# Maximum DDIM denoising iterations. Running too many steps (e.g. 100+) causes
# cumulative latent drift that collapses token predictions to repeated tokens.
MAX_DDIM_STEPS = 10


@torch.no_grad()
def reverse_sample(model: ChannelDiffusionLM, z_T, attention_mask=None,
                   n_steps=None, t_start=None, snr_bin=None, temperature=0.8):
    """DDIM-style deterministic reverse sampling (mirrors diffusion_attempt.py).

    n_steps is capped at MAX_DDIM_STEPS=10 to prevent drift/token-collapse.
    snr_bin: LongTensor [B] for SNR conditioning (0..n_snr_bins-1).
    temperature: final decode temperature; <1 reduces token collapse.
    """
    T = model.cfg.diffusion_steps
    if t_start is None:
        t_start = T
    t_start = max(1, min(T, int(t_start)))
    n_steps = min(n_steps or t_start, MAX_DDIM_STEPS)
    B   = z_T.size(0)
    z_t = z_T
    ts  = torch.linspace(t_start, 1, n_steps, device=z_T.device).long()
    SIGMA_EARLY_EXIT = model.sigmas[2].item() if model.cfg.diffusion_steps >= 2 else 0.0

    for i in range(n_steps):
        t_cur     = ts[i].expand(B)
        sigma_cur = model.sigmas[t_cur].view(-1, 1, 1)
        z0_hat    = model.denoise_step(z_t, t_cur, mode_idx=0, attention_mask=attention_mask,
                                       snr_bin=snr_bin)
        is_last   = (i == n_steps - 1)
        is_clean  = (sigma_cur.mean().item() <= SIGMA_EARLY_EXIT)
        if is_last or is_clean:
            z_t = z0_hat
            break
        sigma_next     = model.sigmas[ts[i + 1].expand(B)].view(-1, 1, 1)
        raw_direction  = (z_t - z0_hat) / sigma_cur.clamp_min(1e-4)
        direction_norm = raw_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction      = raw_direction * (direction_norm.clamp_max(10.0) / direction_norm)
        z_t            = z0_hat + sigma_next * direction

    snr_bin_final = snr_bin if snr_bin is not None else \
                    torch.full((B,), model.cfg.n_snr_bins - 1, dtype=torch.long, device=z_T.device)
    z_final   = model.denoise_step(z_t, torch.zeros(B, dtype=torch.long, device=z_T.device),
                                   mode_idx=1, attention_mask=attention_mask,
                                   snr_bin=snr_bin_final)
    logits    = model.lm_head(z_final) / math.sqrt(model.cfg.d_model)

    # Temperature-scaled sampling reduces token-collapse vs. greedy argmax.
    if temperature > 0 and temperature != 1.0:
        probs = F.softmax(logits / temperature, dim=-1)
        B_, L_, V_ = probs.shape
        token_ids = torch.multinomial(probs.view(-1, V_), num_samples=1).view(B_, L_)
    else:
        token_ids = logits.argmax(-1)
    return token_ids, logits


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

from torch.utils.data import IterableDataset, DataLoader
from transformers import PreTrainedTokenizerFast

# Special tokens in the custom DeepSC vocab that should be stripped.
_CUSTOM_SPECIAL_TOKENS = {'<PAD>', '<END>', '<UNK>'}
# <START> is kept in references so they match the performance.py format
# (SeqtoText.sequence_to_text preserves <START> in both pred & reference).
_NON_REFERENCE_SPECIAL_TOKENS = {'<PAD>', '<START>', '<END>', '<UNK>'}


def _ids_to_text(token_ids, idx_to_token: dict, end_idx: int,
                 keep_start: bool = False) -> str:
    """Convert custom-vocab integer IDs to a plain text string.

    Args:
        token_ids:    Iterable of integer token IDs.
        idx_to_token: Mapping from int ID to token string.
        end_idx:      ID of the <END> token; iteration stops here.
        keep_start:   If True, preserve the leading <START> token so that
                      reference strings match the ``<START> …`` format
                      produced by ``SeqtoText.sequence_to_text`` in
                      performance.py.  Set to False for prediction strings
                      (the diffusion decoder never emits <START>).
    """
    skip_set = _NON_REFERENCE_SPECIAL_TOKENS if not keep_start else _CUSTOM_SPECIAL_TOKENS
    words = []
    for idx in token_ids:
        idx = int(idx)
        if idx == end_idx:
            break
        tok = idx_to_token.get(idx)
        if tok and tok not in skip_set:
            words.append(tok)
    return ' '.join(words)


class TokenSentenceDataset(IterableDataset):
    """Yields (fixed-length padded tensor, original text string, orig_len) per sentence.

    orig_len is the length of the source custom-vocab token ID array (before
    BPE re-tokenization).  It is used by collate_blocks to sort each batch by
    the same criterion as collate_data in dataset.py (sort_by_len on the raw
    pickle sequence), so the first-5 printed samples match performance.py.
    """
    def __init__(self, tokenized_ds, original_texts, orig_lens, seq_len, pad_id=0, eos_id=None):
        self.tokenized_ds   = tokenized_ds
        self.original_texts = original_texts
        self.orig_lens      = orig_lens
        self.seq_len        = seq_len
        self.pad_id         = pad_id
        self.eos_id         = eos_id

    def __iter__(self):
        for ids, text, orig_len in zip(self.tokenized_ds, self.original_texts, self.orig_lens):
            if self.eos_id is not None:
                ids = ids[:self.seq_len - 1] + [self.eos_id]
            else:
                ids = ids[:self.seq_len]
            pad_len = self.seq_len - len(ids)
            ids     = ids + [self.pad_id] * pad_len
            yield torch.tensor(ids, dtype=torch.long), text, orig_len


# ---------------------------------------------------------------------------
# Decode helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def decode_batch(model: ChannelDiffusionLM, batch: dict, noise_std: float,
                 tokenizer, channel_type: str = "AWGN"):
    """
    Run the full encode -> channel -> denoise -> decode pipeline for one batch.

    Evaluation pipeline (mirrors channel_diffusion_loss exactly):
      1. Encode tokens to clean latent z0.
      2. Apply ONLY physical channel noise  ->  z_t = channel(z0, noise_std).
      3. Single denoise_step (mode=0, SNR-conditioned)  ->  z0_hat.
      4. Last-mile decode step  (mode=1, SNR-conditioned)  ->  z_final.
      5. Greedy argmax over lm_head logits -> token ids.
    """
    input_ids      = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    references     = batch["reference_texts"]

    B = input_ids.size(0)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start_timer = time.perf_counter()

    # Step 1 — encode (sender side: mask is available here legitimately)
    z0 = model.encode(input_ids, attention_mask)

    # Step 2 — physical channel only (AWGN / Rayleigh / Rician)
    # Only z_t crosses the channel; the attention_mask is NOT transmitted.
    z_t = model.channels.forward(z0, noise_std, channel_type=channel_type)

    # SNR conditioning bin (shared by both denoising steps)
    snr_bin_val = noise_to_snr_bin(noise_std, model.cfg.n_snr_bins)
    snr_bin     = torch.full((B,), snr_bin_val, dtype=torch.long, device=device)

    # Step 3 — single denoising step (mode=0).
    t_val     = sigma_to_t(model, noise_std)
    t_denoise = torch.full((B,), t_val, dtype=torch.long, device=device)
    z0_hat    = model.denoise_step(z_t, t_denoise, mode_idx=0,
                                   attention_mask=None, snr_bin=snr_bin)

    # Step 4 — last-mile decode (mode=1, t=0); still mask-free.
    t_zeros = torch.zeros(B, dtype=torch.long, device=device)
    z_final = model.denoise_step(z0_hat, t_zeros, mode_idx=1,
                                 attention_mask=None, snr_bin=snr_bin)

    # Step 5 — greedy decode
    logits    = model.lm_head(z_final) / math.sqrt(model.cfg.d_model)
    token_ids = logits.argmax(-1)                       # [B, seq_len]

    # Receiver-side boundary detection (vectorized on GPU):
    # Truncate each sequence at the first EOS token (<END>) if present.
    # Replacing from the EOS index onwards with pad_id ensures both the <END>
    # token itself and any spurious tokens generated after it are removed.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.convert_tokens_to_ids('<PAD>')
    if pad_id is None:
        pad_id = 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids('<END>')
    if eos_id is not None and eos_id >= 0:
        post_eos  = (token_ids == eos_id).cumsum(dim=-1) > 0
        token_ids = torch.where(post_eos, torch.tensor(pad_id, device=device, dtype=token_ids.dtype), token_ids)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_model_end = time.perf_counter()

    predicted = tokenizer.batch_decode(token_ids.cpu(),
                                       skip_special_tokens=True)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_total_end = time.perf_counter()

    timing_info = {
        'model_time': t_model_end - t_start_timer,
        'total_time': t_total_end - t_start_timer,
        'batch_size': B,
    }

    return predicted, references, timing_info



# ---------------------------------------------------------------------------
# Main performance evaluation loop — mirrors performance() in performance.py
# ---------------------------------------------------------------------------

def performance(args, SNR, model, test_loader, tokenizer, bert_scorer=None):
    """
    Evaluate model over a range of SNR values.

    Returns:
        bleu_scores:   dict 'bleu1'..'bleu4' -> np.ndarray [len(SNR)]
        bert_scores:   dict 'f1','precision','recall' -> np.ndarray (empty if no scorer)
        time_stats:    dict 'latency_ms', 'model_latency_ms', 'throughput_sps' -> np.ndarray [len(SNR)]
        all_sentences: list of per-sentence dicts
    """
    bleu_scorers = {
        'bleu1': BleuScore(1, 0, 0, 0),
        'bleu2': BleuScore(0, 1, 0, 0),
        'bleu3': BleuScore(0, 0, 1, 0),
        'bleu4': BleuScore(0, 0, 0, 1),
    }

    all_bleu                                                  = {k: [] for k in bleu_scorers}
    all_bert_f1, all_bert_p, all_bert_r                           = [], [], []
    all_latency_ms, all_model_latency_ms, all_throughput_sps = [], [], []
    all_sentences                                             = []

    # Optional Warmup
    warmup_cnt = getattr(args, 'warmup_batches', 1)
    if warmup_cnt > 0:
        model.eval()
        with torch.no_grad():
            warmup_snr = SNR[0] if len(SNR) > 0 else 0
            noise_std  = SNR_to_noise(warmup_snr)
            for i, batch in enumerate(test_loader):
                if i >= warmup_cnt:
                    break
                decode_batch(model, batch, noise_std, tokenizer, channel_type=args.channel)

    model.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            epoch_latency_ms       = []
            epoch_model_latency_ms = []
            epoch_throughput_sps   = []

            for snr in tqdm(SNR, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                word        = []
                target_word = []
                noise_std   = SNR_to_noise(snr)

                snr_total_time       = 0.0
                snr_model_total_time = 0.0
                snr_total_samples    = 0

                for batch in test_loader:
                    predicted, references, t_info = decode_batch(
                        model, batch, noise_std, tokenizer,
                        channel_type=args.channel
                    )
                    word        += predicted
                    target_word += references

                    snr_total_time       += t_info['total_time']
                    snr_model_total_time += t_info['model_time']
                    snr_total_samples    += t_info['batch_size']

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

                for sample_idx, (pred, ref) in enumerate(zip(word, target_word)):
                    ref = ref[len('<START>'):].lstrip() if ref.startswith('<START>') else ref
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

            bleu_epoch                        = {k: [] for k in bleu_scorers}
            bert_f1_epoch, bert_p_epoch, bert_r_epoch = [], [], []

            for snr_idx, (sent1, sent2) in enumerate(zip(Tx_word, Rx_word)):
                print(f"\n" + "="*80)
                print(f" SNR: {SNR[snr_idx]} dB | Sample Comparisons")
                print(f" Average Latency: {epoch_latency_ms[snr_idx]:.3f} ms/sent (Model: {epoch_model_latency_ms[snr_idx]:.3f} ms/sent) | Throughput: {epoch_throughput_sps[snr_idx]:.1f} sent/s")
                print(f"="*80)
                for pred, ref in zip(sent1[:5], sent2[:5]):
                    ref_display = ref[len('<START>'):].lstrip() if ref.startswith('<START>') else ref
                    print(f"Predicted: {pred}")
                    print(f"Actual   : {ref_display}")
                    print("-"*40)

                # Remove '<START> ' from references for fair score calculation, 
                # since the diffusion model is not trained to generate '<START>'.
                sent2_eval = [
                    r[len('<START>'):].lstrip() if r.startswith('<START>') else r
                    for r in sent2
                ]

                for key, scorer in bleu_scorers.items():
                    bleu_epoch[key].append(scorer.compute_blue_score(sent1, sent2_eval))

                if bert_scorer is not None:
                    bs = bert_scorer.score(sent1, sent2_eval)
                    bert_f1_epoch.append(bs['f1'])
                    bert_p_epoch.append(bs['precision'])
                    bert_r_epoch.append(bs['recall'])
                    print(f"BERTScore  F1={bs['f1']:.4f}  P={bs['precision']:.4f}  R={bs['recall']:.4f}")

            for key in bleu_scorers:
                arr = np.array(bleu_epoch[key])
                all_bleu[key].append(np.mean(arr, axis=1))

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
        'latency_ms':       np.mean(np.array(all_latency_ms),       axis=0),
        'model_latency_ms': np.mean(np.array(all_model_latency_ms), axis=0),
        'throughput_sps':   np.mean(np.array(all_throughput_sps),   axis=0),
    }

    return bleu_scores, bert_scores, time_stats, all_sentences


# ---------------------------------------------------------------------------
# Seed helper
# ---------------------------------------------------------------------------

def setup_seed(seed: int = 10):
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
    # 1. Load checkpoint
    # -----------------------------------------------------------------------
    ckpt_file = args.checkpoint_path
    if not os.path.isfile(ckpt_file):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_file}')

    print(f'Loading checkpoint: {ckpt_file}')
    try:
        ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    except Exception:
        ckpt = torch.load(ckpt_file, map_location=device)

    # The training script saves: {"model_state_dict": ..., "cfg": ..., "step": ..., "val_loss": ...}
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
        cfg        = ckpt.get('cfg', None)
        saved_step = ckpt.get('step', '?')
        print(f'  checkpoint at step {saved_step}, val_loss={ckpt.get("val_loss", "?")}')
    elif isinstance(ckpt, dict):
        # Bare state dict
        state_dict = ckpt
        cfg        = None
    else:
        raise TypeError(f'Unexpected checkpoint format: {type(ckpt)}')

    if cfg is None:
        raise ValueError(
            'Checkpoint does not contain model config. '
            'Make sure the checkpoint was saved by diffusion_attempt.py (includes "cfg" key).'
        )

    # -----------------------------------------------------------------------
    # 2. Rebuild model from saved config
    # -----------------------------------------------------------------------
    model = ChannelDiffusionLM(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}')

    # -----------------------------------------------------------------------
    # 3. Load tokenizer
    # -----------------------------------------------------------------------
    if not os.path.isfile(args.vocab_file):
        raise FileNotFoundError(f'Vocab file not found: {args.vocab_file}')

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.vocab_file)
    # [FIX 4] Don't silently fall back to id 0 if '<PAD>' isn't in the vocab — that
    # could quietly corrupt every attention mask if 0 happens to be a real token.
    _pad_id = tokenizer.convert_tokens_to_ids('<PAD>')
    if _pad_id is None or _pad_id < 0:
        raise ValueError(
            "'<PAD>' token not found in the diffusion tokenizer vocab — refusing to "
            "silently fall back to id 0, since that may be a real token."
        )
    PAD_ID = _pad_id

    _eos_id = tokenizer.convert_tokens_to_ids('<END>')
    EOS_ID = _eos_id if (_eos_id is not None and _eos_id >= 0) else None
    tokenizer.pad_token = '<PAD>'
    tokenizer.eos_token = '<END>'

    print(f'Tokenizer vocab size: {len(tokenizer)}')

    # -----------------------------------------------------------------------
    # 4. Build test data loader from the SAME fixed pickle as performance.py
    #    (txt/test_data.pkl).  The pickle contains custom-vocab integer IDs,
    #    so we decode each sentence back to text using vocab.json, then
    #    re-tokenise with the diffusion tokenizer.
    # -----------------------------------------------------------------------
    if not os.path.isfile(args.test_pkl):
        raise FileNotFoundError(
            f'Test pickle not found: {args.test_pkl}\n'
            f'Run the DeepSC preprocessing pipeline first (preprocess_text.py).'
        )
    if not os.path.isfile(args.deepsc_vocab_file):
        raise FileNotFoundError(f'DeepSC vocab file not found: {args.deepsc_vocab_file}')

    # Load the custom vocab for decoding the pickle IDs.
    with open(args.deepsc_vocab_file, 'rb') as f:
        deepsc_vocab = json.load(f)
    token_to_idx = deepsc_vocab['token_to_idx']
    idx_to_token = {int(v): k for k, v in token_to_idx.items()}
    end_idx      = token_to_idx['<END>']

    # Load the fixed test split.
    with open(args.test_pkl, 'rb') as f:
        raw_test_data = pickle.load(f)
    print(f'Loaded {len(raw_test_data)} sentences from {args.test_pkl}')

    # Decode custom-vocab IDs → text.
    # keep_start=True so reference strings begin with '<START>', matching
    # the SeqtoText.sequence_to_text format used by performance.py.
    # Filter raw_test_data in parallel so indices stay aligned.
    pairs = [
        (row, _ids_to_text(row, idx_to_token, end_idx, keep_start=True))
        for row in raw_test_data
    ]
    pairs = [(row, text) for row, text in pairs if text.strip()]
    raw_test_data_filtered, test_sentences = zip(*pairs) if pairs else ([], [])
    raw_test_data_filtered = list(raw_test_data_filtered)
    test_sentences         = list(test_sentences)
    # Original custom-vocab lengths — used by collate_blocks to mirror
    # collate_data's sort_by_len (which sorts by len(raw_pickle_row)).
    orig_lens = [len(row) for row in raw_test_data_filtered]
    print(f'  {len(test_sentences)} non-empty sentences after decoding')

    # Re-tokenise with the diffusion BPE tokenizer.
    # Strip <START> before re-tokenising so the BPE tokenizer sees clean text
    # (the diffusion model was never trained on sequences that begin with
    # the literal string "<START>").
    from tokenizers import Tokenizer as _RawTok
    raw_tok = _RawTok.from_file(args.vocab_file)
    # test_sentences_for_tok: same sentences but with <START> stripped for encoding.
    test_sentences_for_tok = [
        s[len('<START>'):].lstrip() if s.startswith('<START>') else s
        for s in test_sentences
    ]
    encoded   = raw_tok.encode_batch(test_sentences_for_tok)
    test_data = [enc.ids for enc in encoded]

    SEQ_LEN = cfg.seq_len

    def collate_blocks(batch):
        # batch is a list of (tensor, original_text_string, orig_len) tuples.
        # Sort by orig_len descending — exactly mirrors collate_data's sort_by_len
        # which uses len(x) on the raw custom-vocab ID arrays from the pickle.
        # This guarantees the same batch ordering and the same first-5 printed
        # samples as performance.py.
        batch = sorted(batch, key=lambda x: x[2], reverse=True)
        tensors, texts, _ = zip(*batch)
        input_ids      = torch.stack(tensors, dim=0)
        attention_mask = (input_ids != PAD_ID)
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "reference_texts": list(texts)}

    test_dataset = TokenSentenceDataset(test_data, test_sentences, orig_lens, SEQ_LEN, pad_id=PAD_ID, eos_id=EOS_ID)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size,
                              collate_fn=collate_blocks, num_workers=0, pin_memory=True)

    # -----------------------------------------------------------------------
    # 5. Optionally build BERTScore evaluator
    # -----------------------------------------------------------------------
    bert_scorer = None
    if args.bert_score:
        print(f'Initialising BERTScore evaluator (model: {args.bert_score_model})...')
        bert_scorer = BertScoreEvaluator(
            model_type=args.bert_score_model,
            lang='en',
            batch_size=args.batch_size,
        )

    # -----------------------------------------------------------------------
    # 6. Run evaluation
    # -----------------------------------------------------------------------
    bleu_scores, bert_scores, time_stats, all_sentences = performance(
        args, SNR, model, test_loader, tokenizer, bert_scorer=bert_scorer
    )

    # -----------------------------------------------------------------------
    # 7. Pretty-print results table (identical format to performance.py)
    # -----------------------------------------------------------------------
    sep_width = 10 + 4 * 11 + 3 * 16 + (3 * 11 if bert_scores else 0)
    model_tag = os.path.splitext(os.path.basename(ckpt_file))[0]

    print('\n' + '='*sep_width)
    print(' Results for ' + args.channel + ' Channel using ' + ckpt_file + ' model')
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
    print('='*sep_width)

    # -----------------------------------------------------------------------
    # 8. Save aggregate CSV
    # -----------------------------------------------------------------------
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_file))
    csv_path = (args.output_csv
                if args.output_csv
                else os.path.join(ckpt_dir, f'diffusion_results_{args.channel}_{model_tag}.csv'))

    fieldnames = ['channel', 'checkpoint', 'snr_db',
                  'bleu1', 'bleu2', 'bleu3', 'bleu4']
    if bert_scores:
        fieldnames += ['bert_f1', 'bert_precision', 'bert_recall']
    fieldnames += ['latency_ms_per_sent', 'model_latency_ms_per_sent', 'throughput_sent_per_sec']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    # 9. Save per-sentence predictions CSV
    # -----------------------------------------------------------------------
    pred_csv_path = (args.predictions_csv
                     if args.predictions_csv
                     else os.path.join(ckpt_dir,
                                       f'diffusion_predictions_{args.channel}_{model_tag}.csv'))

    pred_fieldnames = ['epoch', 'snr_db', 'sample_idx', 'predicted', 'reference']
    # [FIX 5] Overwrite each run instead of silently appending forever.
    with open(pred_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=pred_fieldnames)
        writer.writeheader()
        writer.writerows(all_sentences)

    print(f'Predictions saved to: {pred_csv_path}')