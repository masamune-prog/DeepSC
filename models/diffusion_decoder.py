import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Exact ports of DeepSC's channel primitives (utils.py), so both models are
# corrupted by the identical physical process. Do not "improve" these without
# also re-running DeepSC with the same change, or the comparison breaks.
# ---------------------------------------------------------------------------

def power_normalize(x):
    """Matches utils.PowerNormalize exactly: ONE global scalar power over the
    whole tensor (batch, seq, and feature dims combined), only ever scales
    down (never up). This is intentionally not per-sample."""
    power = torch.sqrt(torch.mean(x * x))
    if power > 1:
        x = x / power
    return x


def snr_to_noise(snr_db):
    """Matches utils.SNR_to_noise exactly. Returns the AWGN standard
    deviation (not variance) assuming a unit-power transmitted signal."""
    snr = 10 ** (snr_db / 10.0)
    return 1.0 / math.sqrt(2 * snr)


def rayleigh_channel(tx_sig, n_var):
    """Exact port of utils.Channels.Rayleigh.

    - tx_sig: (..., d) with d even (real/imag pairs).
    - A SINGLE fading coefficient H is drawn per call and applied to every
      position and every batch element (matches the original, which draws
      H once per train_step / val_step, i.e. once per batch).
    - AWGN with std n_var is added post-fade.
    - Zero-forcing "channel estimation" (H^-1) is applied after noise,
      assuming perfect CSI -- matches the original exactly.
    """
    shape = tx_sig.shape
    device = tx_sig.device
    assert shape[-1] % 2 == 0, "channel dim must be even (real/imag pairs)"

    H_real = torch.normal(0, math.sqrt(1 / 2), size=[1], device=device)
    H_imag = torch.normal(0, math.sqrt(1 / 2), size=[1], device=device)
    H = torch.stack([
        torch.cat([H_real, -H_imag]),
        torch.cat([H_imag, H_real]),
    ], dim=0)  # (2, 2)

    tx_pairs = tx_sig.reshape(shape[0], -1, 2)
    faded = torch.matmul(tx_pairs, H)

    rx = faded + torch.randn_like(faded) * n_var  # AWGN, std = n_var

    rx = torch.matmul(rx, torch.inverse(H)).reshape(shape)
    return rx


class ChannelEncoder(nn.Module):
    """Matches DeepSC's channel_encoder exactly: d_model -> 256 -> bandwidth."""
    def __init__(self, d_model, bandwidth=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, bandwidth),
        )

    def forward(self, x):
        return self.net(x)


class ChannelDecoder(nn.Module):
    """Exact port of DeepSC's ChannelDecoder(in_features=bandwidth, size1=d_model, size2=512)."""
    def __init__(self, in_features, size1, size2):
        super().__init__()
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        return self.layernorm(x1 + x5)


class RayleighTextDecoderDiffusion(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_steps=500,
                 channel_bandwidth=16, channel_hidden=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_steps = num_steps

        # Shared vocabulary embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Timestep conditioning MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Channel encoder/decoder, dimension-matched to DeepSC so both models
        # transmit the same bandwidth through the Rayleigh channel.
        self.channel_encoder = ChannelEncoder(embed_dim, bandwidth=channel_bandwidth)
        self.channel_decoder = ChannelDecoder(channel_bandwidth, embed_dim, channel_hidden)

        # Transformer Denoiser: Processes [Denoising State + Channel Output]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim * 2,
            nhead=8,
            dim_feedforward=embed_dim * 8,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.output_projection = nn.Linear(embed_dim * 2, embed_dim)

        beta = torch.linspace(1e-4, 0.02, num_steps)
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', 1.0 - beta)
        self.register_buffer('alpha_bar', torch.cumprod(1.0 - beta, dim=0))

    def apply_rayleigh_channel(self, x_0, snr_db=15):
        """
        Transmits x_0 through the same physical pipeline as DeepSC:
        channel_encoder -> PowerNormalize -> Rayleigh(H) + AWGN -> zero-forcing
        equalization -> channel_decoder.
        """
        tx = self.channel_encoder(x_0)          # (B, S, bandwidth)
        tx = power_normalize(tx)
        n_var = snr_to_noise(snr_db)
        rx = rayleigh_channel(tx, n_var)         # (B, S, bandwidth)
        y_corrupted = self.channel_decoder(rx)   # (B, S, embed_dim)
        return y_corrupted

    def get_time_embedding(self, timesteps):
        half_dim = self.embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return self.time_mlp(emb)

    def forward(self, input_ids, snr_db=15, pad_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        x_0 = self.embedding(input_ids)

        y_corrupted = self.apply_rayleigh_channel(x_0, snr_db=snr_db)

        t = torch.randint(0, self.num_steps, (batch_size,), device=device)
        a_bar = self.alpha_bar[t].view(batch_size, 1, 1)

        inner_noise = torch.randn_like(x_0)
        x_t = torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * inner_noise

        t_emb = self.get_time_embedding(t).unsqueeze(1)
        x_t_prime = x_t + self.pos_embedding[:, :seq_len, :] + t_emb

        model_input = torch.cat([x_t_prime, y_corrupted], dim=-1)

        transformer_out = self.transformer(model_input)
        predicted_x_0 = self.output_projection(transformer_out)

        # Loss -- masked by pad_mask if provided (channel itself is NOT
        # masked, matching DeepSC: the channel corrupts padding too, only
        # the loss ignores it, via loss_function's mask-after-criterion).
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).float()
            loss_mse = ((predicted_x_0 - x_0).pow(2) * mask).sum() / mask.sum().clamp(min=1) / x_0.shape[-1]
        else:
            loss_mse = F.mse_loss(predicted_x_0, x_0)

        logits = torch.matmul(predicted_x_0, self.embedding.weight.T)
        if pad_mask is not None:
            loss_round = F.cross_entropy(
                logits.reshape(-1, self.vocab_size), input_ids.reshape(-1), reduction='none'
            )
            flat_mask = pad_mask.reshape(-1).float()
            loss_round = (loss_round * flat_mask).sum() / flat_mask.sum().clamp(min=1)
        else:
            loss_round = F.cross_entropy(logits.reshape(-1, self.vocab_size), input_ids.reshape(-1))

        return loss_mse + 0.1 * loss_round

    @torch.no_grad()
    def decode_channel_output(self, y_corrupted, use_clamping=True):
        """
        Inference sampling: recovers text embeddings from an already
        channel-decoded observation (i.e. the output of apply_rayleigh_channel,
        or an equivalent real transmission decoded the same way).
        """
        batch_size, seq_len, d_dim = y_corrupted.shape
        device = y_corrupted.device

        x_t = torch.randn(batch_size, seq_len, d_dim, device=device)

        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_emb = self.get_time_embedding(t).unsqueeze(1)

            x_t_prime = x_t + self.pos_embedding[:, :seq_len, :] + t_emb
            model_input = torch.cat([x_t_prime, y_corrupted], dim=-1)

            pred_x_0 = self.output_projection(self.transformer(model_input))

            if use_clamping and i < self.num_steps * 0.5:
                logits = torch.matmul(pred_x_0, self.embedding.weight.T)
                pred_x_0 = self.embedding(logits.argmax(dim=-1))

            a_t = self.alpha[i]
            a_bar_t = self.alpha_bar[i]

            if i > 0:
                a_bar_t_prev = self.alpha_bar[i - 1]
                mean = (torch.sqrt(a_bar_t_prev) * self.beta[i] / (1 - a_bar_t)) * pred_x_0 + \
                       (torch.sqrt(a_t) * (1 - a_bar_t_prev) / (1 - a_bar_t)) * x_t
                var = (1 - a_bar_t_prev) / (1 - a_bar_t) * self.beta[i]
                x_t = mean + torch.sqrt(var) * torch.randn_like(x_t)
            else:
                x_t = pred_x_0

        final_logits = torch.matmul(x_t, self.embedding.weight.T)
        return final_logits.argmax(dim=-1)