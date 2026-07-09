# -*- coding: utf-8 -*-
"""
Diffusion-based decoder for DeepSC.

Architecture
------------
Training  : DDPM x0-prediction loss (MSE in embedding space).
Inference : Deterministic DDIM reverse sampling (eta = 0),
            conditioned on channel_dec_output via cross-attention.

Shapes (all paths)
------------------
  memory  : [B, S, d_model]   channel-decoder output (conditioning signal)
  x0      : [B, S, d_model]   clean target token embeddings
  x_t     : [B, S, d_model]   noisy embedding at step t
  x0_pred : [B, S, d_model]   denoiser prediction of x0
  logits  : [B, S, vocab_size] produced by DeepSC.dense(x0_pred)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule
# ---------------------------------------------------------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (Nichol & Dhariwal, 2021).

    Returns beta values of shape [T].
    """
    steps = T + 1
    t = torch.linspace(0, T, steps, dtype=torch.float64)
    f = torch.cos(((t / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 0.0, 0.999).float()


# ---------------------------------------------------------------------------
# Positional encoding (self-contained copy to avoid circular imports)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding for diffusion timestep t, projected to d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer('freq', freq)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B] integer timestep indices
        Returns:
            emb: [B, d_model]
        """
        args = t.float().unsqueeze(-1) * self.freq.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, d_model]
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Denoiser network
# ---------------------------------------------------------------------------

class DiffusionDenoiser(nn.Module):
    """Cross-attention Transformer denoiser conditioned on timestep and memory.

    Args:
        d_model:    hidden dimension.
        num_heads:  attention heads.
        dff:        feed-forward inner dimension.
        num_layers: number of TransformerDecoderLayer stacks.
        dropout:    dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dff: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.t_emb = TimestepEmbedding(d_model)
        self.input_proj = nn.Linear(d_model, d_model)
        self.dec_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=dff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,   # Pre-LN for training stability
            )
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Predict clean x0 from noisy x_t, conditioned on t and memory.

        Args:
            x_t:    [B, S, d_model]  noisy sequence at diffusion step t
            t:      [B]              integer timestep indices
            memory: [B, S, d_model]  cross-attention key/value (channel-dec output)
        Returns:
            x0_pred: [B, S, d_model]
        """
        t_emb = self.t_emb(t).unsqueeze(1)        # [B, 1, d_model]
        h = self.input_proj(x_t) + t_emb          # [B, S, d_model]
        for layer in self.dec_layers:
            h = layer(h, memory)                   # cross-attend to memory
        return self.out_proj(self.out_norm(h))     # [B, S, d_model]


# ---------------------------------------------------------------------------
# Channel-aware forward corruption
# ---------------------------------------------------------------------------

def _apply_channel_forward(
    x_scaled: torch.Tensor,
    channel: str,
    n_var_t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Corrupt a (schedule-scaled) embedding using the given physical channel.

    This replaces the standard Gaussian noise in the DDPM forward process
    with channel-typed noise, making the denoiser channel-aware.

    The noise magnitude at diffusion step t is:
        n_var_t = n_var * sqrt(1 - ᾱ_t)
    so the corruption fades from 0 (at t=0) to n_var (at t=T-1).

    For Rayleigh / Rician: pairs of embedding dimensions are treated as
    complex-valued symbols (same convention as ``utils.Channels``). The
    fading matrix H is applied, AWGN is added, then H is inverted (perfect
    equalization).  A single H realization is shared across the batch,
    consistent with ``utils.Channels``.

    Args:
        x_scaled: [B, S, d_model]  signal already scaled by sqrt(ᾱ_t).
        channel:  one of 'AWGN', 'Rayleigh', 'Rician'.
        n_var_t:  [B, 1, 1] per-sample noise std at the current timestep.
        device:   torch device.

    Returns:
        x_t: [B, S, d_model]  channel-corrupted signal.
    """
    shape = x_scaled.shape
    B, S, d = shape

    if channel == 'AWGN':
        # Standard Gaussian noise, supports per-sample variance via broadcasting
        return x_scaled + torch.randn(*shape, device=device) * n_var_t

    elif channel in ('Rayleigh', 'Rician'):
        # Single channel realisation per forward call (batch-shared H)
        n_var_scalar: float = n_var_t.mean().item()

        if channel == 'Rayleigh':
            H_real = torch.normal(torch.zeros(1), math.sqrt(0.5)).item()
            H_imag = torch.normal(torch.zeros(1), math.sqrt(0.5)).item()
        else:  # Rician, K = 1
            K = 1
            mean_h = math.sqrt(K / (K + 1))
            std_h  = math.sqrt(1.0 / (K + 1))
            H_real = torch.normal(torch.full((1,), mean_h), std_h).item()
            H_imag = torch.normal(torch.full((1,), mean_h), std_h).item()

        H = torch.tensor(
            [[H_real, -H_imag],
             [H_imag,  H_real]], dtype=x_scaled.dtype, device=device
        )  # [2, 2]

        # Treat pairs of embedding dims as complex symbols: [B, S*d//2, 2]
        x_pairs  = x_scaled.reshape(B, -1, 2)
        x_faded  = torch.matmul(x_pairs, H)                               # fading
        awgn     = torch.normal(0, n_var_scalar, size=x_faded.shape).to(device)
        x_eq     = torch.matmul(x_faded + awgn, torch.inverse(H))         # equalize
        return x_eq.reshape(shape)

    else:
        raise ValueError(
            f"Unknown channel '{channel}'. Choose one of: AWGN, Rayleigh, Rician."
        )



class DiffusionDecoder(nn.Module):
    """DDPM training + DDIM inference decoder that replaces the autoregressive
    Transformer decoder in DeepSC.

    During **training**, `forward()` draws a random timestep, corrupts the
    clean target embedding with Gaussian noise, and returns the MSE loss
    between the denoiser's x0 prediction and the true clean embedding.

    During **inference**, `sample()` runs a deterministic DDIM reverse
    diffusion (eta = 0) starting from pure Gaussian noise, conditioned on
    the channel-decoder output (memory).

    Args:
        trg_vocab_size: vocabulary size for target token embedding.
        max_len:        maximum sequence length.
        d_model:        hidden dimension (must match the rest of DeepSC).
        num_heads:      attention heads for the denoiser layers.
        dff:            feed-forward inner dimension for denoiser layers.
        num_layers:     number of denoiser cross-attention layers.
        diff_steps:     T — total number of forward-process timesteps.
        sampling_steps: number of DDIM reverse steps at inference.
        dropout:        dropout probability.
    """

    def __init__(
        self,
        trg_vocab_size: int,
        max_len: int,
        d_model: int,
        num_heads: int,
        dff: int,
        num_layers: int,
        diff_steps: int = 100,
        sampling_steps: int = 50,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.T = diff_steps
        self.sampling_steps = sampling_steps

        # Embedding for x0 construction during training
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.denoiser = DiffusionDenoiser(d_model, num_heads, dff, num_layers, dropout)

        # Dedicated vocab projection: maps denoised embedding → logits.
        # Trained in the same embedding space as the denoiser, decoupled
        # from the external DeepSC.dense (which lives in the AR decoder space).
        self.vocab_proj = nn.Linear(d_model, trg_vocab_size)

        # Pre-compute and register noise-schedule buffers
        betas = cosine_beta_schedule(diff_steps)            # [T]
        alphas = 1.0 - betas                                # [T]
        alpha_bar = torch.cumprod(alphas, dim=0)            # [T]

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bar', alpha_bar)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        trg_tokens: torch.Tensor,
        memory: torch.Tensor,
        padding_idx: int = None,
        channel: str = 'AWGN',
        n_var: float = 0.1,
    ) -> torch.Tensor:
        """Compute the DDPM x0-prediction MSE loss with channel-typed noise.

        Instead of using pure Gaussian noise in the forward process, the
        corruption q(x_t | x0) is simulated using the specified physical
        channel (AWGN, Rayleigh, or Rician).  The noise variance at step t
        is ``n_var * sqrt(1 - ᾱ_t)``, matching the diffusion schedule.

        Args:
            trg_tokens:  [B, S]           target token IDs.
            memory:      [B, S, d_model]  channel-decoder output (conditioning).
            padding_idx: if given, pad positions are masked out of the loss.
            channel:     physical channel type — 'AWGN' (default), 'Rayleigh',
                         or 'Rician'.  Should match the channel used in the
                         encoder path during the same training step.
            n_var:       channel noise std (output of ``SNR_to_noise(snr_db)``).
        Returns:
            loss: scalar MSE diffusion loss.
        """
        B, S = trg_tokens.shape
        device = trg_tokens.device

        # Build clean embedding x0 ∈ R^{B × S × d_model}
        x0 = self.embedding(trg_tokens) * math.sqrt(self.d_model)
        x0 = self.pos_encoding(x0)   # [B, S, d_model]

        # Sample a random diffusion timestep for each item in the batch
        t = torch.randint(0, self.T, (B,), device=device)   # [B]

        # Forward process: apply physical channel noise scaled by the schedule
        #   n_var_t = n_var * sqrt(1 − ᾱ_t)  →  0 at t=0, n_var at t=T-1
        alpha_bar_t = self.alpha_bar[t].view(B, 1, 1)       # [B, 1, 1]
        x_scaled    = alpha_bar_t.sqrt() * x0               # [B, S, d_model]
        n_var_t     = n_var * (1.0 - alpha_bar_t).sqrt()    # [B, 1, 1]
        x_t = _apply_channel_forward(x_scaled, channel, n_var_t, device)

        # Denoiser predicts clean x0
        x0_pred = self.denoiser(x_t, t, memory)             # [B, S, d_model]

        # --- MSE loss in embedding space ---
        mse = F.mse_loss(x0_pred, x0, reduction='none')     # [B, S, d_model]
        if padding_idx is not None:
            mask = (trg_tokens != padding_idx).float().unsqueeze(-1)  # [B, S, 1]
            denom = mask.sum() * self.d_model + 1e-8
            mse_loss = (mse * mask).sum() / denom
        else:
            mse_loss = mse.mean()

        # --- Cross-entropy loss via vocab_proj (trains the projection head) ---
        logits = self.vocab_proj(x0_pred)                   # [B, S, vocab_size]
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            trg_tokens.reshape(-1),
            ignore_index=padding_idx if padding_idx is not None else -100,
        )

        # Combined loss: MSE keeps the denoiser in embedding space,
        # CE trains the projection head to map embeddings → correct tokens.
        return mse_loss + ce_loss

    # ------------------------------------------------------------------
    # Inference — DDIM (eta = 0, fully deterministic)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        memory: torch.Tensor,
        seq_len: int = None,
    ) -> torch.Tensor:
        """DDIM reverse sampling to produce a denoised embedding.

        The result should be passed through ``DeepSC.dense`` and argmax'd
        to obtain predicted token IDs.

        Args:
            memory:  [B, S, d_model]  channel-decoder output
            seq_len: output sequence length (defaults to memory sequence length)
        Returns:
            x0_pred: [B, seq_len, d_model]  denoised sequence embedding
        """
        B, S_mem, d = memory.shape
        seq_len = seq_len if seq_len is not None else S_mem
        device = memory.device

        # Linearly spaced timestep subset for DDIM
        # indices[0] = 0  (clean),  indices[-1] = T-1  (most noisy)
        indices = torch.linspace(
            0, self.T - 1, self.sampling_steps + 1, dtype=torch.long, device=device
        )

        # Start from pure Gaussian noise
        x = torch.randn(B, seq_len, d, device=device)

        # Reverse: from t = T-1 → 0
        for i in reversed(range(self.sampling_steps)):
            t_curr = indices[i + 1]   # larger t (noisier)
            t_prev = indices[i]       # smaller t (cleaner)

            t_batch = t_curr.expand(B)   # [B]

            # Predict x0
            x0_pred = self.denoiser(x, t_batch, memory)   # [B, S, d_model]

            a_t    = self.alpha_bar[t_curr]
            a_prev = self.alpha_bar[t_prev]

            # Clamp to avoid division by zero when a_t ≈ 1 (near t=0)
            denom = (1.0 - a_t).sqrt().clamp(min=1e-6)

            # Estimated noise direction from x0 prediction
            # ε_pred = (x_t − √ᾱ_t · x0_pred) / √(1 − ᾱ_t)
            eps_pred = (x - a_t.sqrt() * x0_pred) / denom

            # DDIM update (eta = 0: no stochasticity)
            # x_{t-1} = √ᾱ_{t-1} · x0_pred + √(1 − ᾱ_{t-1}) · ε_pred
            x = a_prev.sqrt() * x0_pred + (1.0 - a_prev).sqrt() * eps_pred

        # Return logits directly from the diffusion decoder's own projection head.
        # Do NOT pass through DeepSC.dense — that layer lives in the AR decoder
        # embedding space and was never trained against the diffusion x0 space.
        logits = self.vocab_proj(x)    # [B, seq_len, vocab_size]
        return logits
