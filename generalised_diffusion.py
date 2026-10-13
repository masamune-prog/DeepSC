# -*- coding: utf-8 -*-
"""
Channel Diffusion Language Model — Training Script

Trains a ChannelDiffusionLM that encodes sentences into continuous latents,
corrupts them through a simulated wireless channel (AWGN / Rayleigh / Rician),
and learns to denoise + decode them back to token sequences using a
DDIM-style reverse diffusion process with SNR-conditioned denoising.
"""

# =============================================================================
# Imports — stdlib, numerical, PyTorch, tokenisation, data
# =============================================================================
import inspect
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass
from datasets import load_dataset, DatasetDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
# =============================================================================
# 1. Device detection
# =============================================================================
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# =============================================================================
# 2. Tokenizer initialization (T5 Vocabulary) & Dataset Loading
# =============================================================================
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.model_max_length = int(1e9)  # Prevent max_length warnings when tokenizing long raw documents prior to chunking
hf_tokenizer = tokenizer
PAD_ID = hf_tokenizer.pad_token_id
END_ID = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else PAD_ID

# =============================================================================
# 3. Load dataset, tokenize and chunk into 512-token blocks, and split
# =============================================================================
ds = load_dataset("Skylion007/openwebtext", split="train")

# 2. Tokenize and Chunk into exact 512-token blocks
def tokenize_and_chunk(examples, block_size=512):
    # Tokenize the batch without padding/truncation first
    tokenized = tokenizer(examples["text"], truncation=False, add_special_tokens=False)
    
    # Flatten/concatenate all token IDs in the batch
    concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
    total_length = len(concatenated["input_ids"])
    
    # Drop the trailing remainder tokens that don't fit into a full 512 block
    total_length = (total_length // block_size) * block_size
    
    # Slice into blocks of 512 tokens
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

# Run chunking across the whole dataset
print("Tokenizing and chunking dataset into 512-token blocks...")
chunked_ds = ds.map(
    tokenize_and_chunk,
    batched=True,
    remove_columns=ds.column_names,
    desc="Chunking into 512-token sequences"
)

# 3. Split the chunked sequences (80 / 10 / 10)
# Split off 10% for test (90% remains)
train_val_test = chunked_ds.train_test_split(test_size=0.1, seed=42)

# Split remaining 90% into 80% train and 10% val (1/9th of 90% = 10%)
train_val = train_val_test["train"].train_test_split(test_size=1/9, seed=42)

split_ds = DatasetDict({
    "train": train_val["train"],
    "validation": train_val["test"],
    "test": train_val_test["test"]
})

# 4. Results
print("\nFinal Dataset Split:")
print(split_ds)
print(f"\nSample tokenized length: {len(split_ds['train'][0]['input_ids'])} tokens")

PAD_ID = hf_tokenizer.pad_token_id
END_ID = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else PAD_ID

train_data = split_ds["train"]["input_ids"]
val_data = split_ds["validation"]["input_ids"]
test_data = split_ds["test"]["input_ids"]

print(f'Tokenized with T5Tokenizer. Vocab size: {len(hf_tokenizer)}')
print(f'Example encoding: {train_data[0][:12]}')

# =============================================================================
# 4. Run-mode hyperparameters
# =============================================================================

RUN_MODE = "budget_100"   # "quick" or "budget_100"

if RUN_MODE == "quick":
    SEQ_LEN         = 32
    D_MODEL         = 384
    N_LAYERS        = 6
    N_HEADS         = 6
    D_FF            = 4 * D_MODEL
    DIFFUSION_STEPS = 64
    BATCH_SIZE      = 32
    GRAD_ACCUM      = 1
    LR              = 3e-4
    WEIGHT_DECAY    = 0.1
    WARMUP_STEPS    = 200

elif RUN_MODE == "budget_100":
    SEQ_LEN         = 512
    D_MODEL         = 512
    N_LAYERS        = 10
    N_HEADS         = 8
    D_FF            = 4 * D_MODEL
    DIFFUSION_STEPS = 128
    BATCH_SIZE      = 32
    GRAD_ACCUM      = 2
    LR              = 2e-4
    WEIGHT_DECAY    = 0.1
    WARMUP_STEPS    = 1_000

else:
    raise ValueError("RUN_MODE must be 'quick' or 'budget_100'")

print("RUN_MODE:", RUN_MODE)

# =============================================================================
# 5. Model configuration dataclass
# =============================================================================


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
    sigma_max: float = 1.0        # Match LayerNorm output scale (std ≈ 1.0)
    channel_type: str = "Rayleigh"
    snr_gamma: float = 5.0        # Min-SNR-γ clipping value
    n_snr_bins: int = 19          # Discrete SNR bins covering 0..18 dB evaluation range


# =============================================================================
# 6. Utility functions — SNR conversion, noise schedule
# =============================================================================


def SNR_to_noise(snr):
    """Convert an SNR value (dB) to the corresponding noise standard deviation."""
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)
    return noise_std


def make_sigma_schedule(T, sigma_min=0.01, sigma_max=1.0, device="cuda"):
    """Build a geometric noise schedule: sigma[0] = sigma_min … sigma[T] = sigma_max."""
    steps = torch.arange(T + 1, device=device).float() / T
    sigmas = sigma_min * (sigma_max / sigma_min) ** steps
    return sigmas  # shape [T+1], index with integer t in [0, T]


def sigma_to_t(model, sigma: float) -> int:
    """
    Invert the geometric noise schedule sigma(t) = sigma_min * (sigma_max/sigma_min)^(t/T)
    to find the discrete timestep t that corresponds to a given noise standard deviation.

    Returns an integer in [1, T] so it is always a valid index into model.sigmas.
    """
    T = model.cfg.diffusion_steps
    sigma_clamped = max(model.cfg.sigma_min, min(model.cfg.sigma_max, sigma))
    frac = math.log(sigma_clamped / model.cfg.sigma_min) / math.log(model.cfg.sigma_max / model.cfg.sigma_min)
    t = round(frac * T)
    return max(1, min(T, t))


def noise_to_snr_bin(noise_std: float, n_snr_bins: int = 19) -> int:
    """Convert a noise standard deviation to the nearest discrete SNR bin.

    Inverse of SNR_to_noise: computes SNR_dB from noise_std, then clips to
    the integer bin index [0, n_snr_bins-1] covering 0..18 dB.
    """
    snr_linear = 1.0 / max(2.0 * noise_std ** 2, 1e-8)
    snr_db = 10.0 * math.log10(snr_linear)
    return max(0, min(n_snr_bins - 1, round(snr_db)))


# =============================================================================
# 7. Channel simulation — AWGN, Rayleigh, Rician
# =============================================================================


class Channels():
    """Simulated wireless channel models for semantic communication."""

    def AWGN(self, Tx_sig, n_var):
        """Additive White Gaussian Noise channel."""
        Rx_sig = Tx_sig + torch.randn_like(Tx_sig) * n_var
        return Rx_sig

    def Rayleigh(self, Tx_sig, n_var):
        """Rayleigh flat-fading channel with per-example fading coefficients."""
        shape = Tx_sig.shape                       # [B, L, D]
        B, device = shape[0], Tx_sig.device

        # Per-example fading coefficients
        H_real = torch.normal(0, math.sqrt(1 / 2), size=(B,), device=device)
        H_imag = torch.normal(0, math.sqrt(1 / 2), size=(B,), device=device)
        H = torch.stack([
            torch.stack([H_real, -H_imag], dim=-1),
            torch.stack([H_imag,  H_real], dim=-1),
        ], dim=-2)                                  # [B, 2, 2]

        Tx_pairs = Tx_sig.reshape(B, -1, 2)          # [B, L*D/2, 2]
        Tx_faded = torch.bmm(Tx_pairs, H)            # [B, L*D/2, 2]
        Rx_faded = self.AWGN(Tx_faded, n_var)        # n_var: [B,1,1] broadcasts fine
        Rx_sig = torch.bmm(Rx_faded, torch.inverse(H)).reshape(shape)
        return Rx_sig

    def Rician(self, Tx_sig, n_var, K=1):
        """Rician fading channel with a line-of-sight component (K factor)."""
        shape = Tx_sig.shape
        B, device = shape[0], Tx_sig.device
        mean = math.sqrt(K / (K + 1))
        std = math.sqrt(1 / (K + 1))

        H_real = torch.normal(mean, std, size=(B,), device=device)
        H_imag = torch.normal(mean, std, size=(B,), device=device)
        H = torch.stack([
            torch.stack([H_real, -H_imag], dim=-1),
            torch.stack([H_imag,  H_real], dim=-1),
        ], dim=-2)                                  # [B, 2, 2]

        Tx_pairs = Tx_sig.reshape(B, -1, 2)
        Tx_faded = torch.bmm(Tx_pairs, H)
        Rx_faded = self.AWGN(Tx_faded, n_var)
        Rx_sig = torch.bmm(Rx_faded, torch.inverse(H)).reshape(shape)
        return Rx_sig

    def forward(self, Tx_sig, n_var, channel_type="AWGN", K=1):
        """Dispatch to the appropriate channel model."""
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


# =============================================================================
# 8. Dataset — sentence-level iterable dataset with padding
# =============================================================================


class TokenSentenceDataset(IterableDataset):
    """Wraps a list of tokenized sentences into an IterableDataset with truncation/padding."""

    def __init__(self, tokenized_ds, seq_len, shuffle=False, seed=0, pad_id=0):
        """
        Args:
            tokenized_ds: A list of lists containing token IDs (e.g., train_data)
            seq_len: The desired sequence length; each sentence is truncated/padded to this length
            shuffle: Whether to shuffle the sentences
            seed: Random seed for shuffling
            pad_id: Token ID used for right-padding (should match <PAD> index)
        """
        self.tokenized_ds = tokenized_ds
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.seed = seed
        self.pad_id = pad_id

    def __iter__(self):
        indices = list(range(len(self.tokenized_ds)))
        if self.shuffle:
            epoch_seed = self.seed + random.randint(0, 1000000)
            rng = random.Random(epoch_seed)
            rng.shuffle(indices)

        for idx in indices:
            ids = self.tokenized_ds[idx][:self.seq_len]          # truncate
            pad_len = self.seq_len - len(ids)
            ids = ids + [self.pad_id] * pad_len                  # right-pad
            yield torch.tensor(ids, dtype=torch.long)


# Instantiate train / val / test datasets
train_blocks = TokenSentenceDataset(train_data, SEQ_LEN, shuffle=True, seed=10, pad_id=PAD_ID)
val_blocks = TokenSentenceDataset(val_data, SEQ_LEN, shuffle=False, pad_id=PAD_ID)
test_blocks = TokenSentenceDataset(test_data, SEQ_LEN, shuffle=False, pad_id=PAD_ID)


def collate_blocks(batch):
    """Stack a list of 1-D token tensors into a batch dict with attention mask."""
    input_ids = torch.stack(batch, dim=0)  # [B, L]
    attention_mask = (input_ids != PAD_ID)
    # content_mask excludes both PAD and END from MSE/CE so the denoiser is
    # not trained to reconstruct boundary markers through channel noise.
    content_mask = attention_mask & (input_ids != END_ID)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "content_mask": content_mask}


# Build data loaders
train_loader = DataLoader(train_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)
val_loader = DataLoader(val_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)
test_loader = DataLoader(test_blocks, batch_size=BATCH_SIZE, collate_fn=collate_blocks)

# Verify a batch
b = next(iter(train_loader))
print({k: v.shape for k, v in b.items()})
print("Decoded snippet:\n", hf_tokenizer.decode(b["input_ids"][0][:120].tolist()))
print(f"Dataset sizes: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test")

# =============================================================================
# 9. ChannelDiffusionLM — main model
# =============================================================================

#use the T5 small encoder to acheive a batch*seq_len_512 embedding
@torch.no_grad
class T5TextEncoder(nn.Module):
    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(model_name)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Wraps T5EncoderModel.

        Args:
            input_ids:            (B, L) integer token IDs.
            attention_mask:       (B, L) long/bool, T5-style (1=attend, 0=ignore).
            src_key_padding_mask: (B, L) bool, PyTorch-style (True=ignore).
                                  Converted to T5-style internally if supplied.
        Returns:
            last_hidden_state: (B, L, D)
        """
        # Resolve mask convention: prefer T5-style; convert from PyTorch-style if needed.
        if attention_mask is None and src_key_padding_mask is not None:
            attention_mask = (~src_key_padding_mask).long()
        outputs = self.t5_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

class ChannelDiffusionLM(nn.Module):
    """
    Encoder → continuous latent z0 → channel corruption → DDIM denoiser → token decode.

    Supports ELF-style dual-mode operation:
      - Mode 0 (denoise):  recover z0 from noisy z_t
      - Mode 1 (decode):   project clean z0 → token logits
    """

    def __init__(self, cfg: ChannelDiffusionConfig):
        super().__init__()
        self.cfg = cfg
        self.channels = Channels()

        # Token, positional, timestep, mode, and SNR embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.time_emb = nn.Embedding(cfg.diffusion_steps + 1, cfg.d_model)
        self.mode_emb = nn.Embedding(2, cfg.d_model)   # 0=denoise, 1=decode/round
        self.snr_emb = nn.Embedding(cfg.n_snr_bins, cfg.d_model)  # SNR conditioning

        self.drop = nn.Dropout(cfg.dropout)

        # Semantic encoder — maps token embeddings to latent z0
        self.encoder = T5TextEncoder("t5-small")
        self.z_ln = nn.LayerNorm(cfg.d_model)

        # Denoiser backbone — iteratively refines corrupted latents
        self.denoiser = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
            ), num_layers=cfg.n_dec_layers
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # LM head — weight-tied with token embedding
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        # Pre-compute the geometric noise schedule
        sigmas = make_sigma_schedule(cfg.diffusion_steps, cfg.sigma_min, cfg.sigma_max)
        self.register_buffer("sigmas", sigmas)

    def encode(self, input_ids, attention_mask=None):
        """Encode token IDs into continuous latent representations z0 via T5."""
        # T5 uses attention_mask directly (1=attend, 0=ignore).
        # Convert bool → long if needed so HF doesn't raise on dtype.
        t5_mask = attention_mask.long() if attention_mask is not None else None
        x = self.encoder(input_ids, attention_mask=t5_mask)
        z0 = self.z_ln(x)
        return z0

    def denoise_step(self, z_t, t, mode_idx=0, attention_mask=None, snr_bin=None):
        """
        Single denoising step with ELF mode + SNR conditioning.

        Re-injects fresh positional embeddings so the denoiser can recover
        sequence order even when high noise has washed out the original signal.

        Args:
            z_t:        noisy latent tensor [B, L, D]
            t:          timestep tensor [B]
            mode_idx:   0 = denoise, 1 = decode/round
            snr_bin:    optional LongTensor [B] — discrete SNR bin (0..n_snr_bins-1)
        """
        B, L, D = z_t.shape
        t_emb = self.time_emb(t).unsqueeze(1)  # [B, 1, D]

        # Mode embedding (0 = denoise, 1 = decode/round)
        modes = torch.full((B,), mode_idx, dtype=torch.long, device=z_t.device)
        m_emb = self.mode_emb(modes).unsqueeze(1)  # [B, 1, D]

        # Re-inject positional embeddings to preserve sequence order under noise
        pos = torch.arange(L, device=z_t.device).unsqueeze(0)  # [1, L]
        p_emb = self.pos_emb(pos)                               # [1, L, D]

        # Combine: noisy latent + timestep + mode + position
        x = z_t + t_emb + m_emb + p_emb

        # SNR conditioning: inject channel-quality context into every token position
        if snr_bin is not None:
            s_emb = self.snr_emb(snr_bin).unsqueeze(1)  # [B, 1, D]
            x = x + s_emb

        src_key_padding_mask = None if attention_mask is None else ~attention_mask
        x = self.denoiser(x, src_key_padding_mask=src_key_padding_mask)
        z0_hat = self.ln_f(x)
        return z0_hat

    def forward(self, input_ids, t, mode_idx=0, attention_mask=None, channel_type=None,
                snr_bin=None):
        """Full forward pass: encode → corrupt via channel → denoise → logits."""
        ctype = channel_type if channel_type is not None else self.cfg.channel_type
        z0 = self.encode(input_ids, attention_mask)

        if mode_idx == 1:
            # Decode/rounding mode: always use clean z0 (t=0, no noise)
            t_zero = torch.zeros_like(t)
            z0_hat = self.denoise_step(z0, t_zero, mode_idx=1, attention_mask=attention_mask,
                                       snr_bin=snr_bin)
        else:
            # Denoising mode: corrupt z0 through the selected physical channel, then denoise
            sigma_t = self.sigmas[t].view(-1, 1, 1)  # [B, 1, 1]
            z_t = self.channels.forward(z0, sigma_t, channel_type=ctype)
            z0_hat = self.denoise_step(z_t, t, mode_idx=0, attention_mask=attention_mask,
                                       snr_bin=snr_bin)

        # Scale logits by 1/√d_model to prevent logit explosion
        logits = self.lm_head(z0_hat) / math.sqrt(self.cfg.d_model)
        return logits, z0, z0_hat


# =============================================================================
# 10. Loss function — channel diffusion training loss
# =============================================================================


def channel_diffusion_loss(model, batch, lambda_ce=1.0, lambda_ce_denoise=0.1,
                           channel_type=None):
    """
    Compute the combined training loss:
      MSE (latent denoising) + CE denoise path + CE decode path (ELF mode=1).

    Uses Min-SNR-γ weighting so high-noise batches don't dominate MSE gradients.

    PAD and END tokens are excluded from both MSE and CE losses:
      - PAD positions carry no information and should not be reconstructed.
      - END tokens are boundary markers; training the denoiser to recover them
        through channel noise adds noise to the gradient without semantic value.
    """
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    B = input_ids.size(0)

    # content_mask: True only at real word positions (excludes PAD and END).
    # Falls back to attention_mask if the batch pre-computed content_mask is absent
    # (e.g. during standalone evaluation calls).
    if "content_mask" in batch:
        content_mask = batch["content_mask"].to(device)
    else:
        content_mask = attention_mask & (input_ids != END_ID)
    mask = content_mask.unsqueeze(-1).float()  # [B, L, 1]

    # Encode ground-truth tokens → clean latent z0
    z0 = model.encode(input_ids, attention_mask)

    # Sample random SNR values across 0–18 dB and compute noise stds + SNR bins
    snr_db_vals = torch.empty(B, device=device).uniform_(0, 18)
    snr_linear = 10 ** (snr_db_vals / 10)
    noise_stds = (1.0 / (2.0 * snr_linear).sqrt()).view(-1, 1, 1)
    snr_bin = snr_db_vals.long().clamp(0, model.cfg.n_snr_bins - 1)

    # Sample random diffusion timesteps
    t = torch.randint(1, model.cfg.diffusion_steps + 1, (B,), device=device)

    # Corrupt z0 → z_t using the selected physical channel
    ctype = channel_type if channel_type is not None else model.cfg.channel_type
    z_t = model.channels.forward(z0, noise_stds, channel_type=ctype)

    # Denoise z_t → z0_hat (mode=0)
    z0_hat = model.denoise_step(z_t, t, mode_idx=0, attention_mask=attention_mask,
                                snr_bin=snr_bin)

    # --- Min-SNR-γ loss weighting using physical noise levels ---
    # High-noise batches (large noise_stds) dominate MSE gradients without weighting;
    # this clip prevents them from drowning out the low-noise learning signal.
    snr_t = 1.0 / noise_stds.clamp_min(1e-6) ** 2           # [B, 1, 1]
    weight_t = snr_t.clamp(max=model.cfg.snr_gamma) / snr_t  # [B, 1, 1] ∈ (0, 1]

    # MSE only over real content positions (no PAD, no END)
    per_token_mse = ((z0_hat - z0) ** 2).mean(dim=-1, keepdim=True)  # [B, L, 1]
    denom = (weight_t * mask).sum().clamp_min(1)
    mse = (per_token_mse * weight_t * mask).sum() / denom

    # --- CE auxiliary loss on the denoising path ---
    # Ignore PAD positions (-100) AND END positions so the denoiser is not
    # trained to recover the EOS boundary marker through channel corruption.
    targets = input_ids.clone()
    targets[~content_mask] = -100   # mask out PAD + END
    logits_denoise = model.lm_head(z0_hat) / math.sqrt(model.cfg.d_model)  # [B, L, V]
    ce_denoise = F.cross_entropy(
        logits_denoise.view(-1, logits_denoise.size(-1)),
        targets.view(-1),
        ignore_index=-100,
    )

    # --- CE Decode Path (mode=1): learn clean latent → token mapping ---
    # Use max-SNR bin (18 dB) for the decode path since z0 is clean (no channel noise)
    mode_mask = (torch.rand(B, device=device) < 0.2).long()
    snr_bin_clean = torch.full((B,), model.cfg.n_snr_bins - 1,
                               dtype=torch.long, device=device)
    z_round = model.denoise_step(z0, torch.zeros(B, dtype=torch.long, device=device),
                                 mode_idx=1, attention_mask=attention_mask,
                                 snr_bin=snr_bin_clean)
    logits_decode = model.lm_head(z_round) / math.sqrt(model.cfg.d_model)

    ce_decode = F.cross_entropy(logits_decode.view(-1, logits_decode.size(-1)),
                                targets.view(-1), ignore_index=-100)

    # Combine losses: MSE + auxiliary CE on both denoising and decode paths
    decode_frac = mode_mask.float().mean().clamp_min(0.01)
    loss = mse + lambda_ce_denoise * ce_denoise + lambda_ce * decode_frac * ce_decode
    return loss, {"mse": mse.item(), "ce_denoise": ce_denoise.item(),
                  "ce_decode": ce_decode.item(), "t_mean": t.float().mean().item()}


# =============================================================================
# 11. DDIM reverse sampling
# =============================================================================

# Maximum DDIM denoising iterations. DDIM can skip timesteps, so only a small
# number of steps are needed to traverse t_start → 1. Running too many steps
# (e.g. 100+) causes cumulative drift that collapses token predictions.
MAX_DDIM_STEPS = 10


@torch.no_grad()
def reverse_sample(model, z_T, attention_mask=None, n_steps=None, t_start=None,
                   snr_bin=None, temperature=0.8):
    """
    DDIM-style deterministic reverse sampling.

    Args:
        z_T:         noisy latent from the channel [B, L, D]
        attention_mask: padding mask [B, L]
        t_start:     discrete timestep matching the actual channel noise (from sigma_to_t)
        n_steps:     DDIM steps, capped at MAX_DDIM_STEPS=10 to prevent drift/collapse
        snr_bin:     LongTensor [B] — discrete SNR bin for denoiser conditioning
        temperature: final decode temperature; values < 1 sharpen the token distribution
    """
    T = model.cfg.diffusion_steps
    if t_start is None:
        t_start = T
    t_start = max(1, min(T, int(t_start)))

    # Cap at MAX_DDIM_STEPS: DDIM can jump from t_start to t=1 in few steps by
    # skipping intermediate timesteps — no need to run every one of the 128 steps.
    n_steps = min(n_steps or t_start, MAX_DDIM_STEPS)
    B = z_T.size(0)
    z_t = z_T
    ts = torch.linspace(t_start, 1, n_steps, device=z_T.device).long()

    SIGMA_EARLY_EXIT = model.sigmas[2].item() if model.cfg.diffusion_steps >= 2 else 0.0

    # Iterative DDIM denoising loop
    for i in range(n_steps):
        t_cur = ts[i].expand(B)
        sigma_cur = model.sigmas[t_cur].view(-1, 1, 1)

        z0_hat = model.denoise_step(z_t, t_cur, mode_idx=0, attention_mask=attention_mask,
                                    snr_bin=snr_bin)

        # Early exit if sigma is small enough or this is the last step
        is_last = (i == n_steps - 1)
        is_clean = (sigma_cur.mean().item() <= SIGMA_EARLY_EXIT)
        if is_last or is_clean:
            z_t = z0_hat
            break

        # DDIM update: interpolate between z0_hat and the noise direction
        sigma_next = model.sigmas[ts[i + 1].expand(B)].view(-1, 1, 1)
        raw_direction = (z_t - z0_hat) / sigma_cur.clamp_min(1e-4)
        direction_norm = raw_direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        max_norm = 10.0
        direction = raw_direction * (direction_norm.clamp_max(max_norm) / direction_norm)
        z_t = z0_hat + sigma_next * direction

    # Last-Mile Decode (mode=1): project clean latent → tokens with SNR conditioning
    snr_bin_final = snr_bin if snr_bin is not None else \
                    torch.full((B,), model.cfg.n_snr_bins - 1, dtype=torch.long, device=z_T.device)
    z_final = model.denoise_step(z_t, torch.zeros(B, dtype=torch.long, device=z_T.device),
                                 mode_idx=1, attention_mask=attention_mask,
                                 snr_bin=snr_bin_final)
    logits = model.lm_head(z_final) / math.sqrt(model.cfg.d_model)

    # Temperature-scaled sampling reduces token-collapse vs. greedy argmax
    if temperature > 0 and temperature != 1.0:
        probs = F.softmax(logits / temperature, dim=-1)
        B_, L_, V_ = probs.shape
        token_ids = torch.multinomial(probs.view(-1, V_), num_samples=1).view(B_, L_)
    else:
        token_ids = logits.argmax(-1)

    # Mask out every position after the first <END> token so downstream
    # decoding / metrics are not polluted by tokens predicted past EOS.
    end_mask = (token_ids == END_ID).cumsum(dim=-1)   # 0 before END, ≥1 after
    # Positions where cumsum > 1 are strictly after the first END — replace with PAD.
    # (The END token itself sits where cumsum == 1, which we preserve.)
    token_ids = token_ids.masked_fill(end_mask > 1, PAD_ID)

    return token_ids, logits


# =============================================================================
# 12. Model instantiation
# =============================================================================

cfg = ChannelDiffusionConfig(
    vocab_size=len(hf_tokenizer),
    seq_len=32,
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


# =============================================================================
# 12b. Phased T5 freezing helpers
# =============================================================================
# Training is split into three phases:
#
#   Phase 1  (steps 0 … T5_WARMUP_STEPS)
#     Only the bottom N_T5_OPEN_LAYERS encoder blocks of T5 are trainable;
#     everything else in T5 (embeddings, upper layers, final LN) is frozen.
#     The denoiser / LM-head are always fully trainable.
#
#   Phase 2  (steps T5_WARMUP_STEPS … ENCODER_FREEZE_UNTIL)
#     The entire T5 encoder is frozen.  Let the denoiser and LM-head adapt
#     to the now-stable latent geometry without encoder drift.
#
#   Phase 3  (steps > ENCODER_FREEZE_UNTIL)
#     All parameters unfrozen — full fine-tuning.

N_T5_OPEN_LAYERS    = 2       # number of bottom T5 blocks left trainable in phase 1
T5_WARMUP_STEPS     = 2000   # steps before moving to phase 2
ENCODER_FREEZE_UNTIL = 10000  # steps before unfreezing encoder (phase 3)


def _set_t5_grad(model: ChannelDiffusionLM, requires_grad: bool):
    """Toggle requires_grad for every T5 parameter."""
    for p in model.encoder.t5_encoder.parameters():
        p.requires_grad_(requires_grad)


def freeze_t5_except_bottom(model: ChannelDiffusionLM, n_open: int = N_T5_OPEN_LAYERS):
    """
    Phase 1: freeze all T5 weights except the bottom `n_open` encoder blocks.
    The shared token-embedding table inside T5 is also frozen.
    """
    _set_t5_grad(model, False)          # freeze everything in T5 first
    t5_enc = model.encoder.t5_encoder.encoder
    for i in range(min(n_open, len(t5_enc.block))):
        for p in t5_enc.block[i].parameters():
            p.requires_grad_(True)      # unfreeze the bottom N blocks
    print(f"[freeze] Phase 1 — T5 frozen except bottom {n_open} encoder block(s).")


def freeze_encoder(model: ChannelDiffusionLM):
    """Phase 2: freeze the entire T5 encoder (latent space clean-up phase)."""
    _set_t5_grad(model, False)
    print("[freeze] Phase 2 — T5 encoder fully frozen.")


def unfreeze_all(model: ChannelDiffusionLM):
    """Phase 3: restore full fine-tuning for all parameters."""
    _set_t5_grad(model, True)
    print("[freeze] Phase 3 — T5 encoder unfrozen (full fine-tuning).")


# Apply phase-1 freezing immediately after model creation
freeze_t5_except_bottom(model, N_T5_OPEN_LAYERS)


def print_model_parameters(model: torch.nn.Module):
    """
    Prints a detailed breakdown of the model's parameters,
    taking into account weight sharing (like the tied LM head).
    """
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

        # Flag shared weights (like lm_head.weight tied to tok_emb.weight)
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

# =============================================================================
# 13. Training configuration
# =============================================================================

TOTAL_STEPS  = 200000
LR           = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
GRAD_CLIP    = 1.0
LOG_EVERY    = 50
EVAL_EVERY   = 500
SAMPLE_EVERY = 1000
LAMBDA_CE    = 1.0
CKPT_PATH    = "channel_diffusion_generalised_ckpt_Rayleigh_owt.pt"

device = next(model.parameters()).device


def make_lr_lambda(warmup_steps, total_steps):
    """Create a cosine-decay LR lambda with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return lr_lambda


optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(WARMUP_STEPS, TOTAL_STEPS))

# =============================================================================
# 14. Evaluation and sampling helpers
# =============================================================================


@torch.no_grad()
def evaluate(model, val_loader, n_batches=20):
    """Validate by calling channel_diffusion_loss directly to exactly mirror training."""
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
    """
    Evaluate the model as a semantic channel codec:
      source → Encoder → z0 → Channel(z0, sigma_T) → Reverse Diffusion → pred tokens

    This is the CORRECT evaluation protocol. The model was trained to denoise
    corrupted encodings, NOT to generate from pure zero-mean Gaussian noise.
    """
    model.eval()
    batch = {k: v.to(device) for k, v in batch.items()}
    input_ids = batch["input_ids"][:n_show]
    attention_mask = batch["attention_mask"][:n_show]

    # Encode ground-truth tokens to the clean latent
    z0 = model.encode(input_ids, attention_mask)
    T = model.cfg.diffusion_steps
    sigma_T = model.sigmas[T].item()

    # Use the provided noise std or default to sigma_max (worst-case channel)
    noise_std = n_var if n_var is not None else sigma_T

    # Invert the noise schedule to find the correct starting timestep
    t_start = sigma_to_t(model, noise_std)

    # Compute SNR bin from noise_std for denoiser conditioning
    snr_bin_val = noise_to_snr_bin(noise_std, model.cfg.n_snr_bins)
    B_show = z0.size(0)
    snr_bin = torch.full((B_show,), snr_bin_val, dtype=torch.long, device=z0.device)

    # Simulate the physical channel: z_T = Rayleigh(z0, noise_std)
    z_T = model.channels.Rayleigh(z0, noise_std)

    # Run DDIM reverse diffusion (capped at MAX_DDIM_STEPS to prevent drift)
    pred_ids, _ = reverse_sample(model, z_T, attention_mask=attention_mask,
                                 t_start=t_start, snr_bin=snr_bin)

    # Baseline: direct decode from clean z0 (upper bound, no channel noise)
    t_zero = torch.zeros(B_show, dtype=torch.long, device=z0.device)
    snr_bin_clean = torch.full((B_show,), model.cfg.n_snr_bins - 1,
                               dtype=torch.long, device=z0.device)
    z_clean = model.denoise_step(z0, t_zero, mode_idx=1, attention_mask=attention_mask,
                                 snr_bin=snr_bin_clean)
    direct_ids = (model.lm_head(z_clean) / math.sqrt(model.cfg.d_model)).argmax(-1)

    # Print comparison: ground truth vs direct decode vs diffusion recovery
    print(f"  [channel σ={noise_std:.4f}, t_start={t_start}/{T}, SNR≈{(z0.std()/noise_std).item():.1f}dB]")
    for i in range(n_show):
        mask_i = attention_mask[i]
        gt = tokenizer.decode(input_ids[i][mask_i].tolist(), skip_special_tokens=True)
        direct = tokenizer.decode(direct_ids[i][mask_i].tolist(), skip_special_tokens=True)
        pred = tokenizer.decode(pred_ids[i][mask_i].tolist(), skip_special_tokens=True)
        print(f"  GT     : {gt}")
        print(f"  Direct : {direct}")   # should be near-perfect if CE head is trained
        print(f"  Diffuse: {pred}")     # DDIM recovery through the noisy channel
        print()
    model.train()


# =============================================================================
# 15. Training loop
# =============================================================================


def train(model, train_loader, val_loader, tokenizer, total_steps):
    """Main training loop with periodic evaluation, sampling, and checkpointing."""
    step = 0
    best_val_loss = float("inf")
    model.train()

    train_iter = iter(train_loader)
    start_time = time.time()

    while step < total_steps:
        # Fetch next batch (restart iterator on exhaustion)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}

        # ── Phased T5 freezing ──────────────────────────────────────────────
        if step == T5_WARMUP_STEPS:
            freeze_encoder(model)          # phase 2: freeze whole encoder
        elif step == ENCODER_FREEZE_UNTIL:
            unfreeze_all(model)            # phase 3: full fine-tuning
        # ────────────────────────────────────────────────────────────────────

        # Forward + loss computation
        loss, stats = channel_diffusion_loss(model, batch, lambda_ce=LAMBDA_CE, channel_type="Rayleigh")

        # Backward + gradient clipping + optimizer step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        # Periodic logging
        if step % LOG_EVERY == 0:
            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - start_time
            print(f"step {step}/{total_steps} | loss {loss.item():.4f} "
                  f"(mse {stats['mse']:.4f}, ce_d {stats['ce_denoise']:.4f}, ce_r {stats['ce_decode']:.4f}) "
                  f"| t_mean {stats['t_mean']:.1f} | lr {lr_now:.2e} | {elapsed:.1f}s")

        # Periodic validation + checkpointing
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

        # Periodic sanity-check sampling
        if step % SAMPLE_EVERY == 0 and step > 0:
            print(f"  [sample] step {step} reverse-diffusion sanity check:")
            sample_batch = next(iter(val_loader))
            sanity_check_sample(model, sample_batch, tokenizer)

        step += 1

    print("training complete.")


# =============================================================================
# 16. Launch training
# =============================================================================

train(model, train_loader, val_loader, hf_tokenizer, total_steps=TOTAL_STEPS)
