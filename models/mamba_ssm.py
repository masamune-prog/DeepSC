# -*- coding: utf-8 -*-
"""
models/mamba_ssm.py

Self-contained Mamba-style Selective State Space Model (SSM) implemented in
pure PyTorch — no custom CUDA extensions required.

Architecture follows Mamba (Gu & Dao, 2023):
  - Selective (input-dependent) ∆, B, C projections
  - Parallel associative scan for training; simple recurrence for inference
  - Depthwise conv1d before SSM for local mixing
  - Gated output projection (SiLU gate × SSM output)

Reference: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
           https://arxiv.org/abs/2312.00752
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sequential scan (works on any device; drop-in swap for a fused CUDA kernel)
# ---------------------------------------------------------------------------

def _scan(A_bar: torch.Tensor, dBu: torch.Tensor) -> torch.Tensor:
    """Sequential SSM scan: h_t = A_bar_t * h_{t-1} + dBu_t.

    Args:
        A_bar: (B*d_inner, L, d_state) — discrete per-step decay factors.
        dBu:   (B*d_inner, L, d_state) — discretised B · u input.

    Returns:
        h: (B*d_inner, L, d_state) — hidden states at every time step.
    """
    Bd, L, N = dBu.shape
    h = torch.zeros(Bd, N, device=dBu.device, dtype=dBu.dtype)
    hs = []
    for t in range(L):
        h = A_bar[:, t, :] * h + dBu[:, t, :]
        hs.append(h)
    return torch.stack(hs, dim=1)  # (Bd, L, N)


# ---------------------------------------------------------------------------
# Core selective SSM
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """Selective State Space layer (the inner SSM of a Mamba block).

    Args:
        d_model:  Input/output feature dimension (after expansion inside MambaLayer).
        d_state:  SSM state dimension (N). Default: 16.
        dt_rank:  Rank of the ∆ projection. Default: ceil(d_model / 16).
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # Input-dependent projections for ∆, B, C  (the "selective" mechanism)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)

        # ∆ projection: dt_rank → d_model
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        # Initialise so ∆ starts small
        nn.init.uniform_(self.dt_proj.weight,
                         -(self.dt_rank ** -0.5), self.dt_rank ** -0.5)
        nn.init.constant_(self.dt_proj.bias, 0.0)

        # Fixed (learnable) A in log space: (d_model, d_state)
        # HiPPO-inspired initialisation: A_n = n+1
        A = torch.arange(1, d_state + 1, dtype=torch.float
                         ).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip / direct-through connection, one per feature
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B, L, d = x.shape
        N = self.d_state

        # --- Selective projections -------------------------------------------
        xbc = self.x_proj(x)                                   # (B, L, dt_rank + 2N)
        dt_raw, B_proj, C_proj = xbc.split(
            [self.dt_rank, N, N], dim=-1)                      # (B,L,·)

        # ∆: ensure positive via softplus
        delta = F.softplus(self.dt_proj(dt_raw))               # (B, L, d)

        # --- Discretise A (ZOH) → Ā = exp(∆ · A) --------------------------
        A = -torch.exp(self.A_log.float())                     # (d, N)
        # dA[b,l,d,n] = delta[b,l,d] * A[d,n]
        dA = torch.einsum('bld,dn->bldn', delta, A)            # (B, L, d, N)
        A_bar = torch.exp(dA)                                  # (B, L, d, N) ∈ (0,1)

        # --- Discretise B (ZOH) → B̄u = ∆ · B · u -------------------------
        # dB[b,l,d,n] = delta[b,l,d] * B_proj[b,l,n]
        dB = torch.einsum('bld,bln->bldn', delta, B_proj)      # (B, L, d, N)
        dBu = dB * x.unsqueeze(-1)                             # (B, L, d, N)

        # --- Sequential scan -------------------------------------------------
        # Reshape to (B*d, L, N) for the scan helper
        A_bar_r = A_bar.permute(0, 2, 1, 3).reshape(B * d, L, N)
        dBu_r   = dBu.permute(0, 2, 1, 3).reshape(B * d, L, N)

        h = _scan(A_bar_r, dBu_r)                             # (B*d, L, N)
        h = h.reshape(B, d, L, N).permute(0, 2, 1, 3)        # (B, L, d, N)

        # --- Output: y = Σ_n C_n * h_n + D * x ------------------------------
        y = torch.einsum('bldn,bln->bld', h, C_proj)          # (B, L, d)
        y = y + self.D * x                                     # skip connection

        return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Full Mamba block
# ---------------------------------------------------------------------------

class MambaLayer(nn.Module):
    """One Mamba block with pre-norm and residual connection.

    Structure (per original paper):
        x  →  LayerNorm  →  in_proj (×2)
                           │             │
                      x branch       gate branch
                           │
                      Conv1d → SiLU
                           │
                       SelectiveSSM
                           │
                      y × SiLU(gate)
                           │
                       out_proj
                           │
                    +  residual (x)

    Args:
        d_model:  Model (token) dimension.
        d_state:  SSM state dimension N. Default: 16.
        d_conv:   Depthwise conv kernel size. Default: 4.
        expand:   Inner expansion factor. Default: 2.
        dropout:  Dropout on output. Default: 0.0.
        causal:   True → right-padding only (for causal decoder).
                  False → symmetric padding (for bidirectional encoder).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        self.causal = causal

        self.norm = nn.LayerNorm(d_model, eps=1e-6)

        # Project to 2×d_inner: one for x, one for the SiLU gate
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise conv along the sequence axis
        # padding = d_conv-1 so the output length stays ≥ L; we trim in forward
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )

        self.act = nn.SiLU()

        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)  — residual already applied.
        """
        residual = x
        x = self.norm(x)

        # Dual projection
        xz = self.in_proj(x)                                   # (B, L, 2*d_inner)
        x_b, z = xz.chunk(2, dim=-1)                           # each (B, L, d_inner)

        # Depthwise conv  (operates on sequence dim)
        xc = x_b.transpose(1, 2)                               # (B, d_inner, L)
        xc = self.conv1d(xc)                                   # (B, d_inner, L + d_conv-1)

        L = x_b.size(1)
        if self.causal:
            # Keep only the first L outputs (no future leakage)
            xc = xc[:, :, :L]
        else:
            # Symmetric trim for bidirectional use
            pad = self.d_conv - 1
            left  = pad // 2
            right = pad - left
            xc = xc[:, :, left: xc.size(-1) - right] if right > 0 else xc[:, :, left:]

        x_b = self.act(xc).transpose(1, 2)                    # (B, L, d_inner)

        # Selective SSM
        y = self.ssm(x_b)                                      # (B, L, d_inner)

        # Gated output
        y = y * self.act(z)

        # Project back and add residual
        return self.dropout(self.out_proj(y)) + residual
