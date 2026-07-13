# -*- coding: utf-8 -*-
"""
models/transceiver.py

DeepSC transceiver with Mamba (Selective SSM) encoder and decoder layers.

Encoder:
    Token Embedding → Positional Encoding → [MambaEncoderLayer × N]
    Each MambaEncoderLayer uses a *bidirectional* Mamba block (forward + backward
    scan merged) so the encoder sees full context — analogous to a BERT-style
    non-causal Transformer encoder.

Decoder:
    Token Embedding → Positional Encoding → [MambaDecoderLayer × N]
    Each MambaDecoderLayer uses:
        1. Causal Mamba self-SSM   (replaces masked self-attention)
        2. Multi-head cross-attention to encoder memory  (preserved)
        3. PositionwiseFeedForward + LayerNorm

Original author: HQ Xie
Mamba refactor: 2026-07
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mamba_ssm import MambaLayer


# ---------------------------------------------------------------------------
# Positional Encoding (unchanged)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Multi-Head Attention (kept for decoder cross-attention)
# ---------------------------------------------------------------------------

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attn = self.attention(query, key, value, mask=mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)
        x = self.dense(x)
        x = self.dropout(x)
        return x

    def attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores += (mask * -1e9)
        p_attn = F.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


# ---------------------------------------------------------------------------
# Feed-Forward Network (kept for decoder layers)
# ---------------------------------------------------------------------------

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Mamba Encoder Layer  (bidirectional)
# ---------------------------------------------------------------------------

class MambaEncoderLayer(nn.Module):
    """Encoder layer using a bidirectional Mamba SSM block.

    Bidirectional operation: run one forward-scan Mamba and one reversed-scan
    Mamba, then merge via learned linear combination.  This gives the encoder
    full-sequence context without the O(L²) cost of self-attention.

    Structure:
        x  →  MambaLayer(causal=False) [forward]   ─┐
           →  flip → MambaLayer(causal=False) → flip ─┤  concat → linear → out
    """

    def __init__(self, d_model: int, dff: int, dropout: float = 0.1,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        # Forward and backward Mamba scans (each non-causal for clean padding)
        self.mamba_fwd = MambaLayer(d_model, d_state=d_state, d_conv=d_conv,
                                    expand=expand, dropout=dropout, causal=False)
        self.mamba_bwd = MambaLayer(d_model, d_state=d_state, d_conv=d_conv,
                                    expand=expand, dropout=dropout, causal=False)

        # Merge: 2*d_model → d_model
        self.merge = nn.Linear(2 * d_model, d_model, bias=False)
        self.norm  = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        """
        Args:
            x:    (B, L, d_model)
            mask: accepted but ignored (Mamba processes all positions).
        Returns:
            (B, L, d_model)
        """
        fwd = self.mamba_fwd(x)                  # forward scan; residual inside
        bwd = self.mamba_bwd(x.flip(1)).flip(1)  # backward scan; residual inside

        # Merge and normalise
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))
        return self.norm(merged + x)


# ---------------------------------------------------------------------------
# Mamba Decoder Layer  (causal self-SSM + cross-attention + FFN)
# ---------------------------------------------------------------------------

class MambaDecoderLayer(nn.Module):
    """Decoder layer: causal Mamba self-SSM + MHA cross-attention + FFN.

    Structure:
        x  →  Mamba(causal=True)  →  [+x]  →  LN
           →  CrossAttn(q=x, kv=memory)  →  [+]  →  LN
           →  FFN  →  [+]  →  LN
    """

    def __init__(self, d_model: int, num_heads: int, dff: int, dropout: float = 0.1,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()

        # 1) Causal Mamba self-sequence block (replaces masked self-attention)
        self.mamba_self = MambaLayer(d_model, d_state=d_state, d_conv=d_conv,
                                     expand=expand, dropout=dropout, causal=True)

        # 2) Cross-attention to encoder memory (preserved)
        self.cross_attn = MultiHeadedAttention(num_heads, d_model, dropout=dropout)
        self.layernorm_cross = nn.LayerNorm(d_model, eps=1e-6)

        # 3) Feed-forward
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=dropout)
        self.layernorm_ffn = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        """
        Args:
            x:                (B, T, d_model) — decoder input (target sequence so far)
            memory:           (B, S, d_model) — encoder output (channel-decoded)
            look_ahead_mask:  accepted for API compatibility; Mamba is inherently causal
            trg_padding_mask: (B, 1, S) — cross-attention padding mask
        Returns:
            (B, T, d_model)
        """
        # 1) Causal Mamba self-SSM (residual applied inside MambaLayer)
        x = self.mamba_self(x)

        # 2) Cross-attention to encoder memory
        src_out = self.cross_attn(x, memory, memory, trg_padding_mask)
        x = self.layernorm_cross(x + src_out)

        # 3) FFN
        ffn_out = self.ffn(x)
        x = self.layernorm_ffn(x + ffn_out)

        return x


# ---------------------------------------------------------------------------
# Encoder stack
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    "Mamba Encoder: Embedding → PE → [MambaEncoderLayer × N]"

    def __init__(self, num_layers, src_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([
            MambaEncoderLayer(d_model, dff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, src_mask):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)

        return x


# ---------------------------------------------------------------------------
# BERT Semantic Encoder (unchanged)
# ---------------------------------------------------------------------------

class BertSemanticEncoder(nn.Module):
    """Pretrained multilingual BERT encoder with a learned projection to d_model.

    Args:
        bert_model_name: HuggingFace model identifier
            (e.g. 'bert-base-multilingual-cased').
        d_model: Target hidden dimension for the downstream channel encoder.
        freeze: If True, all BERT parameters are frozen (no gradients).
            Only the projection layer is trained.
    """
    def __init__(self, bert_model_name: str, d_model: int, freeze: bool = True):
        super(BertSemanticEncoder, self).__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(bert_model_name)
        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.proj = nn.Linear(self.bert.config.hidden_size, d_model)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids:      LongTensor [batch, seq_len]
            attention_mask: FloatTensor [batch, seq_len] (1=real, 0=pad)
        Returns:
            Tensor [batch, seq_len, d_model]
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return self.proj(hidden)


# ---------------------------------------------------------------------------
# Decoder stack
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    "Mamba Decoder: Embedding → PE → [MambaDecoderLayer × N]"

    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([
            MambaDecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)

        return x


# ---------------------------------------------------------------------------
# Channel Decoder (unchanged)
# ---------------------------------------------------------------------------

class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

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
        output = self.layernorm(x1 + x5)
        return output


# ---------------------------------------------------------------------------
# DeepSC transceiver (constructor signature unchanged)
# ---------------------------------------------------------------------------

class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1,
                 use_bert_encoder=False,
                 bert_model_name='bert-base-multilingual-cased',
                 freeze_bert=True):
        """DeepSC transceiver with Mamba encoder/decoder.

        When use_bert_encoder=True:
          - src_vocab_size is ignored (BERT has its own embeddings).
          - trg_vocab_size should be the BERT tokenizer vocab size.

        Constructor signature is identical to the original Transformer version.
        """
        super(DeepSC, self).__init__()

        self.use_bert_encoder = use_bert_encoder

        if use_bert_encoder:
            self.encoder = BertSemanticEncoder(
                bert_model_name, d_model, freeze=freeze_bert
            )
        else:
            self.encoder = Encoder(
                num_layers, src_vocab_size, src_max_len, d_model, num_heads, dff, dropout
            )

        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 16)
        )
        self.channel_decoder = ChannelDecoder(16, d_model, 512)

        self.decoder = Decoder(
            num_layers, trg_vocab_size, trg_max_len, d_model, num_heads, dff, dropout
        )

        self.dense = nn.Linear(d_model, trg_vocab_size)

    def encode(self, src, src_mask=None, attention_mask=None):
        """Unified encode interface.

        For the BERT encoder path, pass attention_mask (BERT convention).
        For the custom encoder path, pass src_mask (padding mask convention).
        """
        if self.use_bert_encoder:
            return self.encoder(src, attention_mask=attention_mask)
        else:
            return self.encoder(src, src_mask)
