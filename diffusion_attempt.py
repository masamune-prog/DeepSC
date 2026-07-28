# -*- coding: utf-8 -*-
"""Diffusion attempt (fixed)

Fixes applied vs. the original script, each tagged [FIX n] at the site of the change:

  1. `forward()` was accidentally nested inside `denoise_step()` (over-indented), so it
     never became a real method of ChannelDiffusionLM. Dedented it back to class level.
  2. The RUN_MODE-selected LR / WEIGHT_DECAY / WARMUP_STEPS were being silently
     overwritten by a second, unconditional assignment further down. Removed the
     duplicate assignment so the profile's choices actually take effect.
  3. GRAD_ACCUM was defined per-profile but never used. Implemented real gradient
     accumulation in train().
  4. The diffusion timestep `t` fed into the time embedding was sampled independently
     of the actual noise magnitude used to corrupt z0 (`noise_stds`), so the time
     embedding carried no information correlated with the real corruption strength ­—
     a train/inference mismatch, since inference derives t_start from the real sigma
     via sigma_to_t(). Now t is derived from noise_stds via a vectorized sigma_to_t,
     so training and inference use the same (noise -> t) mapping.
  5. snr_bin was computed with `.long()` (truncation), which meant draws had to land
     essentially exactly on 18.0 dB to reach the top bin — starving the bin used by
     default at eval/decode time. Switched to `.round().long()`.
  6. Rayleigh/Rician equalization used a raw `torch.inverse(H)` with no floor, which
     blows up on near-singular (deep-fade) H draws. Added ridge regularization before
     inverting.
  7. PAD_ID silently fell back to 0 if '<PAD>' wasn't found in the vocab, which could
     silently corrupt the attention mask. Now raises a clear error instead.
  8. `mode_mask` was sampled per-example but never actually used to select which rows
     go through the decode path (ce_decode ran on the full batch regardless, then got
     scaled by an unrelated scalar). Reworked so the decode path genuinely runs only
     on the selected subset.

Original file is located at
    https://colab.research.google.com/#fileId=https%3A//storage.googleapis.com/kaggle-colab-exported-notebooks/taurusilver/diffusion-attempt.cf1540c2-d7d2-4ba1-9f13-64e6cf0e1847.ipynb
"""

import os
import math, time, json, random, inspect
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import IterableDataset, DataLoader
import pickle
from dataclasses import dataclass
from tokenizers import Tokenizer as _RawTok
from transformers import PreTrainedTokenizerFast

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# ---------------------------------------------------------------------------
# Dataset loading — uses the SAME canonical pkl splits as main.py
# ---------------------------------------------------------------------------
DEEPSC_VOCAB_FILE = 'txt/vocab.json'
DIFFUSION_VOCAB_FILE = 'txt/diffusion_vocab.json'
_SPECIAL_TOKENS = {'<PAD>', '<START>', '<END>', '<UNK>'}

print('Loading DeepSC vocab...')
with open(DEEPSC_VOCAB_FILE) as f:
    _deepsc_vocab = json.load(f)
_token_to_idx = _deepsc_vocab['token_to_idx']
_idx_to_token = {int(v): k for k, v in _token_to_idx.items()}
_end_idx = _token_to_idx['<END>']


def _ids_to_text(token_ids):
    """Convert a list of custom-vocab IDs to a plain text string."""
    words = []
    for idx in token_ids:
        idx = int(idx)
        if idx == _end_idx:
            break
        tok = _idx_to_token.get(idx, '')
        if tok and tok not in _SPECIAL_TOKENS:
            words.append(tok)
    return ' '.join(words)


def _load_split(split_name):
    pkl_path = f'txt/{split_name}_data.pkl'
    print(f'  Loading {pkl_path}...')
    with open(pkl_path, 'rb') as f:
        raw = pickle.load(f)
    texts = [_ids_to_text(row) for row in raw]
    texts = [t for t in texts if t.strip()]
    print(f'  {len(texts)} non-empty sentences from {split_name}')
    return texts


print('Loading canonical train/val/test splits from pkl files...')
train_texts = _load_split('train')
val_texts = _load_split('val')
test_texts = _load_split('test')
print(f'Total: {len(train_texts)+len(val_texts)+len(test_texts)} sentences '
      f'({len(train_texts)} train / {len(val_texts)} val / {len(test_texts)} test)')
print('Sample train sentence:', train_texts[0])

print('Re-tokenizing with diffusion tokenizer...')
_raw_tok = _RawTok.from_file(DIFFUSION_VOCAB_FILE)
hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=DIFFUSION_VOCAB_FILE)

# [FIX 7] Don't silently fall back to id 0 if '<PAD>' isn't in the vocab — that could
# quietly corrupt every attention mask if 0 happens to be a real token.
_pad_id = hf_tokenizer.convert_tokens_to_ids('<PAD>')
if _pad_id is None:
    raise ValueError(
        "'<PAD>' token not found in the diffusion tokenizer vocab — refusing to "
        "silently fall back to id 0, since that may be a real token."
    )
PAD_ID = _pad_id

_eos_id = hf_tokenizer.convert_tokens_to_ids('<END>')
if _eos_id is None:
    raise ValueError(
        "'<END>' token not found in the diffusion tokenizer vocab."
    )
EOS_ID = _eos_id
hf_tokenizer.pad_token = '<PAD>'
hf_tokenizer.eos_token = '<END>'


def _tokenize(texts):
    return [enc.ids for enc in _raw_tok.encode_batch(texts)]


train_data = _tokenize(train_texts)
val_data = _tokenize(val_texts)
test_data = _tokenize(test_texts)
print(f'Tokenized.  Vocab size: {len(hf_tokenizer)}')
print(f'Example encoding: {train_data[0][:12]}')

# =======================
# 0) Choose a run profile
# =======================

RUN_MODE = "budget_100"   # "quick" or "budget_100"

if RUN_MODE == "quick":
    SEQ_LEN = 32
    D_MODEL = 384
    N_LAYERS = 6
    N_HEADS = 6
    D_FF = 4 * D_MODEL
    DIFFUSION_STEPS = 64
    BATCH_SIZE = 32
    GRAD_ACCUM = 1
    LR = 3e-4
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 200

elif RUN_MODE == "budget_100":
    SEQ_LEN = 512
    D_MODEL = 512
    N_LAYERS = 10
    N_HEADS = 8
    D_FF = 4 * D_MODEL
    DIFFUSION_STEPS = 128
    BATCH_SIZE = 128
    GRAD_ACCUM = 2
    LR = 2e-4
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 1_000

else:
    raise ValueError("RUN_MODE must be 'quick' or 'budget_100'")

print("RUN_MODE:", RUN_MODE)
# [FIX 2] These are now the ONLY definitions of LR / WEIGHT_DECAY / WARMUP_STEPS —
# the training-config section below no longer redefines them, so the profile you
# pick above actually determines the optimizer hyperparameters.

# =======================
# Model Configuration
# =======================


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
    channel_type: str = "Rayleigh"
    snr_gamma: float = 5.0
    n_snr_bins: int = 19


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


class Channels():
    def AWGN(self, Tx_sig, n_var):
        Rx_sig = Tx_sig + torch.randn_like(Tx_sig) * n_var
        return Rx_sig

    @staticmethod
    def _regularized_inverse(H, eps=1e-2):
        """[FIX 6] Ridge-regularized inverse: H^{-1} ~= (H^T H + eps*I)^{-1} H^T.

        A raw torch.inverse(H) blows up whenever a fading draw lands close to
        singular (a deep fade), which happens with non-trivial probability across
        a training run and can inject inf/NaN into z_t. Ridge regularization caps
        the amplification of near-singular draws while leaving well-conditioned
        draws essentially unaffected.
        """
        B = H.shape[0]
        I = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype).unsqueeze(0).expand(B, -1, -1)
        Ht = H.transpose(-1, -2)
        return torch.bmm(torch.inverse(torch.bmm(Ht, H) + eps * I), Ht)

    def Rayleigh(self, Tx_sig, n_var):
        shape = Tx_sig.shape                       # [B, L, D]
        B, device = shape[0], Tx_sig.device

        H_real = torch.normal(0, math.sqrt(1 / 2), size=(B,), device=device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=(B,), device=device)
        H = torch.stack([
            torch.stack([H_real, -H_imag], dim=-1),
            torch.stack([H_imag, H_real], dim=-1),
        ], dim=-2)                                  # [B, 2, 2]

        Tx_pairs = Tx_sig.reshape(B, -1, 2)          # [B, L*D/2, 2]
        Tx_faded = torch.bmm(Tx_pairs, H)            # [B, L*D/2, 2]
        Rx_faded = self.AWGN(Tx_faded, n_var)        # n_var: [B,1,1] broadcasts fine
        H_inv = self._regularized_inverse(H)         # [FIX 6]
        Rx_sig = torch.bmm(Rx_faded, H_inv).reshape(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        shape = Tx_sig.shape
        B, device = shape[0], Tx_sig.device
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))

        H_real = torch.normal(mean, std, size=(B,), device=device)
        H_imag = torch.normal(mean, std, size=(B,), device=device)
        H = torch.stack([
            torch.stack([H_real, -H_imag], dim=-1),
            torch.stack([H_imag, H_real], dim=-1),
        ], dim=-2)                                  # [B, 2, 2]

        Tx_pairs = Tx_sig.reshape(B, -1, 2)
        Tx_faded = torch.bmm(Tx_pairs, H)
        Rx_faded = self.AWGN(Tx_faded, n_var)
        H_inv = self._regularized_inverse(H)         # [FIX 6]
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


class TokenSentenceDataset(IterableDataset):
    def __init__(self, tokenized_ds, seq_len, shuffle=False, seed=0, pad_id=0, eos_id=None):
        self.tokenized_ds = tokenized_ds
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.pad_id = pad_id
        self.eos_id = eos_id

    def __iter__(self):
        indices = list(range(len(self.tokenized_ds)))
        if self.shuffle:
            epoch_seed = self.seed + random.randint(0, 1000000)
            rng = random.Random(epoch_seed)
            rng.shuffle(indices)

        for idx in indices:
            raw_ids = self.tokenized_ds[idx]
            if self.eos_id is not None:
                ids = raw_ids[:self.seq_len - 1] + [self.eos_id]
            else:
                ids = raw_ids[:self.seq_len]
            pad_len = self.seq_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            yield torch.tensor(ids, dtype=torch.long)


train_blocks = TokenSentenceDataset(train_data, SEQ_LEN, shuffle=True, seed=10, pad_id=PAD_ID, eos_id=EOS_ID)
val_blocks = TokenSentenceDataset(val_data, SEQ_LEN, shuffle=False, pad_id=PAD_ID, eos_id=EOS_ID)
test_blocks = TokenSentenceDataset(test_data, SEQ_LEN, shuffle=False, pad_id=PAD_ID, eos_id=EOS_ID)


def collate_blocks(batch):
    input_ids = torch.stack(batch, dim=0)  # [B, L]
    attention_mask = (input_ids != PAD_ID)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


train_loader = DataLoader(train_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)
val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)
test_loader = DataLoader(test_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)

b = next(iter(train_loader))
print({k: v.shape for k, v in b.items()})
print("Decoded snippet:\n", hf_tokenizer.decode(b["input_ids"][0][:120].tolist()))
print(f"Dataset sizes: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")


def make_sigma_schedule(T, sigma_min=0.01, sigma_max=1.0, device="cuda"):
    steps = torch.arange(T + 1, device=device).float() / T
    sigmas = sigma_min * (sigma_max / sigma_min) ** steps
    return sigmas  # shape [T+1], index with integer t in [0, T]


class ChannelDiffusionLM(nn.Module):
    def __init__(self, cfg: ChannelDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.channels = Channels()

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)
        self.mode_emb = nn.Embedding(2, cfg.d_model)
        self.snr_emb = nn.Embedding(cfg.n_snr_bins, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)

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
        z0 = self.z_ln(x)
        # [FIX 9] Zero out padded positions so their encoder outputs (which are
        # nonzero even under src_key_padding_mask) don't propagate noise into
        # the channel or pollute real-token attention in the denoiser.
        if attention_mask is not None:
            z0 = z0 * attention_mask.unsqueeze(-1).float()
        return z0

    def denoise_step(self, z_t, t, mode_idx=0, attention_mask=None, snr_bin=None):
        B, L, D = z_t.shape
        t_emb = self.time_emb(t).unsqueeze(1)  # [B, 1, D]

        modes = torch.full((B,), mode_idx, dtype=torch.long, device=z_t.device)
        m_emb = self.mode_emb(modes).unsqueeze(1)  # [B, 1, D]

        pos = torch.arange(L, device=z_t.device).unsqueeze(0)  # [1, L]
        p_emb = self.pos_emb(pos)                                 # [1, L, D]

        # [FIX 9] Zero padded positions in z_t before adding positional/time
        # conditioning, so channel-corrupted pad noise doesn't influence
        # real-token attention even before src_key_padding_mask takes effect.
        if attention_mask is not None:
            z_t = z_t * attention_mask.unsqueeze(-1).float()

        x = z_t + t_emb + m_emb + p_emb

        if snr_bin is not None:
            s_emb = self.snr_emb(snr_bin).unsqueeze(1)  # [B, 1, D]
            x = x + s_emb

        # [FIX 9] Zero padded positions again after conditioning addition, so
        # the added time/mode/pos embeddings don't pollute masked slots either.
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()

        src_key_padding_mask = None if attention_mask is None else ~attention_mask
        x = self.denoiser(x, src_key_padding_mask=src_key_padding_mask)
        z0_hat = self.ln_f(x)

        # [FIX 9] Zero padded output positions so downstream loss/logit
        # computations can never accidentally receive pad-slot activations.
        if attention_mask is not None:
            z0_hat = z0_hat * attention_mask.unsqueeze(-1).float()
        return z0_hat

    # [FIX 1] This was previously nested inside denoise_step() (over-indented), so it
    # never became a real method — model(...) would have hit nn.Module's default
    # forward and raised NotImplementedError. Dedented to class-method level.
    def forward(self, input_ids, t, mode_idx=0, attention_mask=None, channel_type=None,
                snr_bin=None):
        ctype = channel_type if channel_type is not None else self.cfg.channel_type
        z0 = self.encode(input_ids, attention_mask)

        if mode_idx == 1:
            t_zero = torch.zeros_like(t)
            z0_hat = self.denoise_step(z0, t_zero, mode_idx=1, attention_mask=attention_mask,
                                        snr_bin=snr_bin)
        else:
            sigma_t = self.sigmas[t].view(-1, 1, 1)  # [B, 1, 1]
            z_t = self.channels.forward(z0, sigma_t, channel_type=ctype)
            z0_hat = self.denoise_step(z_t, t, mode_idx=0, attention_mask=attention_mask,
                                        snr_bin=snr_bin)

        logits = self.lm_head(z0_hat) / math.sqrt(self.cfg.d_model)
        return logits, z0, z0_hat


def sigma_to_t(model, sigma: float) -> int:
    """Scalar version — inverts the geometric noise schedule for a single sigma."""
    T = model.cfg.diffusion_steps
    sigma_clamped = max(model.cfg.sigma_min, min(model.cfg.sigma_max, sigma))
    frac = math.log(sigma_clamped / model.cfg.sigma_min) / math.log(model.cfg.sigma_max / model.cfg.sigma_min)
    t = round(frac * T)
    return max(1, min(T, t))


def sigma_to_t_batch(model, sigma: torch.Tensor) -> torch.Tensor:
    """[FIX 4] Vectorized version of sigma_to_t, used so the time embedding fed to the
    denoiser during TRAINING is derived from the same (noise -> t) mapping used at
    inference, instead of being an independently-sampled, uncorrelated random index.

    sigma: any-shape float tensor of per-example noise std devs.
    Returns a LongTensor of the same shape, clamped to [1, T].
    """
    T = model.cfg.diffusion_steps
    sigma_clamped = sigma.clamp(model.cfg.sigma_min, model.cfg.sigma_max)
    frac = torch.log(sigma_clamped / model.cfg.sigma_min) / math.log(model.cfg.sigma_max / model.cfg.sigma_min)
    t = torch.round(frac * T).long()
    return t.clamp(1, T)


def noise_to_snr_bin(noise_std, n_snr_bins: int = 19):
    """Convert noise standard deviation(s) to discrete SNR bin(s).

    Works for both a python float (returns int) and a torch.Tensor (returns a
    LongTensor of the same shape), matching how it's used in eval vs. training.
    """
    if isinstance(noise_std, torch.Tensor):
        snr_linear = 1.0 / (2.0 * noise_std.clamp_min(1e-8) ** 2)
        snr_db = 10.0 * torch.log10(snr_linear)
        return snr_db.round().long().clamp(0, n_snr_bins - 1)  # [FIX 5] round, not truncate
    snr_linear = 1.0 / max(2.0 * noise_std ** 2, 1e-8)
    snr_db = 10.0 * math.log10(snr_linear)
    return max(0, min(n_snr_bins - 1, round(snr_db)))


def channel_diffusion_loss(model, batch, lambda_ce=1.0, lambda_ce_denoise=0.1,
                            channel_type=None):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    B = input_ids.size(0)
    mask = attention_mask.unsqueeze(-1).float()

    z0 = model.encode(input_ids, attention_mask)

    snr_db_vals = torch.empty(B, device=device).uniform_(0, 18)
    snr_linear = 10 ** (snr_db_vals / 10)
    noise_stds = (1.0 / (2.0 * snr_linear).sqrt()).view(-1, 1, 1)
    # [FIX 5] round instead of truncate, so the top SNR bin is actually reachable.
    snr_bin = noise_to_snr_bin(noise_stds.view(-1), model.cfg.n_snr_bins)

    # [FIX 4] Derive t from the SAME noise_stds that actually corrupt z0, instead of
    # sampling t independently. This keeps the time embedding correlated with the real
    # corruption strength during training, matching what reverse_sample() does at
    # inference (t_start = sigma_to_t(model, noise_std)).
    t = sigma_to_t_batch(model, noise_stds.view(-1))

    ctype = channel_type if channel_type is not None else model.cfg.channel_type
    z_t = model.channels.forward(z0, noise_stds, channel_type=ctype)

    z0_hat = model.denoise_step(z_t, t, mode_idx=0, attention_mask=attention_mask,
                                 snr_bin=snr_bin)

    snr_t = 1.0 / noise_stds.clamp_min(1e-6) ** 2           # [B, 1, 1]
    weight_t = snr_t.clamp(max=model.cfg.snr_gamma) / snr_t  # [B, 1, 1] in (0, 1]

    # [FIX 9] z0_hat padded positions are already zeroed inside denoise_step;
    # applying mask here as well makes the loss computation self-documenting
    # and guards against any future refactor that skips the in-model zeroing.
    z0_hat_masked = z0_hat * mask           # [B, L, D], pad slots = 0
    z0_masked     = z0    * mask           # [B, L, D], same for target

    per_token_mse = ((z0_hat_masked - z0_masked) ** 2).mean(dim=-1, keepdim=True)  # [B, L, 1]
    denom = (weight_t * mask).sum().clamp_min(1)
    mse = (per_token_mse * weight_t * mask).sum() / denom

    targets = input_ids.clone()
    targets[~attention_mask] = -100
    logits_denoise = model.lm_head(z0_hat_masked) / math.sqrt(model.cfg.d_model)  # [B, L, V]
    ce_denoise = F.cross_entropy(
        logits_denoise.view(-1, logits_denoise.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )

    # [FIX 8] mode_mask now genuinely selects which rows go through the decode path,
    # instead of being sampled and then ignored (previously the decode path ran on the
    # full batch regardless, and mode_mask only scaled the final scalar loss).
    mode_mask = torch.rand(B, device=device) < 0.2
    decode_frac = mode_mask.float().mean().clamp_min(0.01)
    if mode_mask.any():
        sel = mode_mask
        z0_sel = z0[sel]
        am_sel = attention_mask[sel]
        n_sel = sel.sum().item()
        snr_bin_clean = torch.full((n_sel,), model.cfg.n_snr_bins - 1,
                                    dtype=torch.long, device=device)
        z_round = model.denoise_step(z0_sel, torch.zeros(n_sel, dtype=torch.long, device=device),
                                      mode_idx=1, attention_mask=am_sel,
                                      snr_bin=snr_bin_clean)
        logits_decode = model.lm_head(z_round) / math.sqrt(model.cfg.d_model)
        targets_sel = targets[sel]
        ce_decode = F.cross_entropy(logits_decode.view(-1, logits_decode.size(-1)),
                                     targets_sel.view(-1), ignore_index=-100)
    else:
        ce_decode = torch.zeros((), device=device)

    loss = mse + lambda_ce_denoise * ce_denoise + lambda_ce * decode_frac * ce_decode
    return loss, {"mse": mse.item(), "ce_denoise": ce_denoise.item(),
                  "ce_decode": ce_decode.item(), "t_mean": t.float().mean().item()}


MAX_DDIM_STEPS = 10


@torch.no_grad()
def reverse_sample(model, z_T, attention_mask=None, n_steps=None, t_start=None,
                    snr_bin=None, temperature=0.8):
    T = model.cfg.diffusion_steps
    if t_start is None:
        t_start = T
    t_start = max(1, min(T, int(t_start)))
    n_steps = min(n_steps or t_start, MAX_DDIM_STEPS)
    B = z_T.size(0)
    z_t = z_T
    ts = torch.linspace(t_start, 1, n_steps, device=z_T.device).long()

    SIGMA_EARLY_EXIT = model.sigmas[2].item() if model.cfg.diffusion_steps >= 2 else 0.0

    for i in range(n_steps):
        t_cur = ts[i].expand(B)
        sigma_cur = model.sigmas[t_cur].view(-1, 1, 1)

        z0_hat = model.denoise_step(z_t, t_cur, mode_idx=0, attention_mask=attention_mask,
                                     snr_bin=snr_bin)

        is_last = (i == n_steps - 1)
        is_clean = (sigma_cur.mean().item() <= SIGMA_EARLY_EXIT)
        if is_last or is_clean:
            z_t = z0_hat
            break

        sigma_next = model.sigmas[ts[i + 1].expand(B)].view(-1, 1, 1)
        raw_direction = (z_t - z0_hat) / sigma_cur.clamp_min(1e-4)
        direction_norm = raw_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        max_norm = 10.0
        direction = raw_direction * (direction_norm.clamp_max(max_norm) / direction_norm)
        z_t = z0_hat + sigma_next * direction

    snr_bin_final = snr_bin if snr_bin is not None else \
        torch.full((B,), model.cfg.n_snr_bins - 1, dtype=torch.long, device=z_T.device)
    z_final = model.denoise_step(z_t, torch.zeros(B, dtype=torch.long, device=z_T.device),
                                  mode_idx=1, attention_mask=attention_mask,
                                  snr_bin=snr_bin_final)
    logits = model.lm_head(z_final) / math.sqrt(model.cfg.d_model)

    if temperature > 0 and temperature != 1.0:
        probs = F.softmax(logits / temperature, dim=-1)
        B_, L_, V_ = probs.shape
        token_ids = torch.multinomial(probs.view(-1, V_), num_samples=1).view(B_, L_)
    else:
        token_ids = logits.argmax(-1)
    return token_ids, logits


cfg = ChannelDiffusionConfig(
    vocab_size=len(hf_tokenizer),
    seq_len=512,
    d_model=D_MODEL,
    n_enc_layers=N_LAYERS // 2,
    n_dec_layers=N_LAYERS // 2,
    n_heads=N_HEADS,
    d_ff=D_FF,
    dropout=0.1,
    diffusion_steps=DIFFUSION_STEPS,
    sigma_min=0.002,
    sigma_max=1.0,
    channel_type="Rayleigh",
    snr_gamma=5.0,
    n_snr_bins=19,
)
model = ChannelDiffusionLM(cfg).to(device)


def print_model_parameters(model: torch.nn.Module):
    total_params = 0
    trainable_params = 0
    unique_params_set = set()

    print(f"{'Module Name':<45} | {'Shape':<20} | {'Parameters':<15} | {'Requires Grad'}")
    print("-" * 95)

    for name, param in model.named_parameters():
        param_id = id(param)
        is_shared = param_id in unique_params_set
        unique_params_set.add(param_id)

        num_params = param.numel()
        grad_status = str(param.requires_grad)
        shape_str = str(list(param.shape))

        display_name = f"{name} (Shared)" if is_shared else name
        print(f"{display_name:<45} | {shape_str:<20} | {num_params:<15,} | {grad_status}")

        if not is_shared:
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

    print("-" * 95)
    print(f"Total Unique Parameters:     {total_params:,}")
    print(f"Total Trainable Parameters:  {trainable_params:,}")


print_model_parameters(model)

# ----------------------------------------------------------------------
# Config for the training run itself
# ----------------------------------------------------------------------
# [FIX 2] LR / WEIGHT_DECAY / WARMUP_STEPS are intentionally NOT redefined here —
# they come from the RUN_MODE block above so the chosen profile actually applies.
TOTAL_STEPS = 200000
GRAD_CLIP = 1.0
LOG_EVERY = 50
EVAL_EVERY = 500
SAMPLE_EVERY = 1000
LAMBDA_CE = 1.0
CKPT_PATH = "channel_diffusion_fixed_ray_eos_long_ctx.pt"

device = next(model.parameters()).device


def make_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(WARMUP_STEPS, TOTAL_STEPS))


@torch.no_grad()
def evaluate(model, val_loader, n_batches=20):
    model.eval()
    total_loss, total_mse, total_ce, n = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, stats = channel_diffusion_loss(model, batch, lambda_ce=LAMBDA_CE)
        total_loss += loss.item()
        total_mse += stats['mse']
        total_ce += stats['ce_decode']
        n += 1
    model.train()
    return {"val_loss": total_loss / n, "val_mse": total_mse / n, "val_ce": total_ce / n}


@torch.no_grad()
def sanity_check_sample(model, batch, tokenizer, n_show=2, n_var=None):
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch["input_ids"][:n_show]
    attention_mask = batch["attention_mask"][:n_show]

    z0 = model.encode(input_ids, attention_mask)
    T = model.cfg.diffusion_steps
    sigma_T = model.sigmas[T].item()

    noise_std = n_var if n_var is not None else sigma_T
    t_start = sigma_to_t(model, noise_std)

    snr_bin_val = noise_to_snr_bin(noise_std, model.cfg.n_snr_bins)
    B_show = z0.size(0)
    snr_bin = torch.full((B_show,), snr_bin_val, dtype=torch.long, device=z0.device)

    z_T = model.channels.Rayleigh(z0, noise_std)
    pred_ids, _ = reverse_sample(model, z_T, attention_mask=attention_mask,
                                  t_start=t_start, snr_bin=snr_bin)

    t_zero = torch.zeros(B_show, dtype=torch.long, device=z0.device)
    snr_bin_clean = torch.full((B_show,), model.cfg.n_snr_bins - 1,
                                dtype=torch.long, device=z0.device)
    z_clean = model.denoise_step(z0, t_zero, mode_idx=1, attention_mask=attention_mask,
                                  snr_bin=snr_bin_clean)
    direct_ids = (model.lm_head(z_clean) / math.sqrt(model.cfg.d_model)).argmax(-1)

    print(f"  [channel σ={noise_std:.4f}, t_start={t_start}/{T}, SNR≈{(z0.std()/noise_std).item():.1f}dB]")
    for i in range(n_show):
        mask_i = attention_mask[i]
        gt = tokenizer.decode(input_ids[i][mask_i].tolist(), skip_special_tokens=True)
        direct = tokenizer.decode(direct_ids[i][mask_i].tolist(), skip_special_tokens=True)
        pred = tokenizer.decode(pred_ids[i][mask_i].tolist(), skip_special_tokens=True)
        print(f"  GT     : {gt}")
        print(f"  Direct : {direct}")
        print(f"  Diffuse: {pred}")
        print()
    model.train()


def train(model, train_loader, val_loader, tokenizer, total_steps, grad_accum=1):
    step = 0
    best_val_loss = float("inf")
    model.train()

    train_iter = iter(train_loader)
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < total_steps:
        # [FIX 3] Real gradient accumulation: accumulate `grad_accum` micro-batches'
        # worth of gradients (scaled so the effective loss is the mean over the
        # accumulated batches) before stepping the optimizer/scheduler, giving an
        # effective batch size of BATCH_SIZE * grad_accum as GRAD_ACCUM intended.
        accum_loss = 0.0
        accum_stats = None
        for micro in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}
            loss, stats = channel_diffusion_loss(model, batch, lambda_ce=LAMBDA_CE,
                                                  channel_type="Rayleigh")
            (loss / grad_accum).backward()
            accum_loss += loss.item()
            accum_stats = stats  # log stats from the last micro-batch

        accum_loss /= grad_accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if step % LOG_EVERY == 0:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            print(f"step {step}/{total_steps} | loss {accum_loss:.4f} "
                  f"(mse {accum_stats['mse']:.4f}, ce_d {accum_stats['ce_denoise']:.4f}, "
                  f"ce_r {accum_stats['ce_decode']:.4f}) "
                  f"| t_mean {accum_stats['t_mean']:.1f} | lr {lr_now:.2e} | {elapsed:.1f}s")

        if step % EVAL_EVERY == 0 and step > 0:
            val_stats = evaluate(model, val_loader)
            print(f"  [eval] step {step} | val_loss {val_stats['val_loss']:.4f} "
                  f"(mse {val_stats['val_mse']:.4f}, ce {val_stats['val_ce']:.4f})")

            if val_stats["val_loss"] < best_val_loss:
                best_val_loss = val_stats["val_loss"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "cfg": model.cfg,
                    "step": step,
                    "val_loss": best_val_loss,
                    "model_arch_src": inspect.getsource(ChannelDiffusionLM),
                    "model_arch_cls": "ChannelDiffusionLM",
                }, CKPT_PATH)
                print(f"  [ckpt] saved new best (val_loss={best_val_loss:.4f}) -> {CKPT_PATH}")

        if step % SAMPLE_EVERY == 0 and step > 0:
            print(f"  [sample] step {step} reverse-diffusion sanity check:")
            sample_batch = next(iter(val_loader))
            sanity_check_sample(model, sample_batch, tokenizer)

        step += 1

    print("training complete.")


train(model, train_loader, val_loader, hf_tokenizer, total_steps=TOTAL_STEPS, grad_accum=GRAD_ACCUM)