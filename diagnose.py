# -*- coding: utf-8 -*-
import torch
import math
import torch.nn.functional as F
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from functools import partial
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data, collate_data_bert
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.transceiver import ChannelDecoder
# 1. Initialize your model architecture matching your exact training params
model = DeepSC(
    num_layers=4,          
    src_vocab_size=119547, 
    trg_vocab_size=119547, 
    src_max_len=128, 
    trg_max_len=128, 
    d_model=128,           
    num_heads=4, 
    dff=512,
    use_bert_encoder=True
)


# 2. Load your checkpoint
checkpoint = torch.load("checkpoints/deepsc-Rayleigh/checkpoint_67.pth", map_location="cpu")

if isinstance(checkpoint, dict):
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    state_dict = checkpoint

# Robust filtering block (Should report no skips now!)
model_state = model.state_dict()
filtered_state_dict = {}
skipped_keys = []
for k, v in state_dict.items():
    if k in model_state and model_state[k].shape != v.shape:
        skipped_keys.append((k, v.shape, model_state[k].shape))
    else:
        filtered_state_dict[k] = v

if skipped_keys:
    for k, ckpt_shape, model_shape in skipped_keys:
        print(f"[SKIP] Shape mismatch for '{k}': checkpoint={ckpt_shape}, model={model_shape}")

load_result = model.load_state_dict(filtered_state_dict, strict=False)

if load_result.missing_keys:
    print(f"[WARN] Missing keys (not in checkpoint): {load_result.missing_keys}")
if load_result.unexpected_keys:
    print(f"[INFO] Unexpected keys (ignored):        {load_result.unexpected_keys}")

print("Successfully loaded model state dict!")
model.eval()

# 3. Setup hooks to catch cross-attention from ALL decoder layers
num_decoder_layers = len(model.decoder.dec_layers)
captured_attn = {i: [] for i in range(num_decoder_layers)}
handles = []

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        if module.attn is not None:
            captured_attn[layer_idx].append(module.attn.detach().cpu())
    return hook_fn

for i, dec_layer in enumerate(model.decoder.dec_layers):
    handles.append(dec_layer.src_mha.register_forward_hook(make_hook(i)))

# 4. Synthesize a dummy evaluation sequence
batch_size = 1
src_seq_len = 20
tgt_seq_len = 15

dummy_src = torch.randint(0, 119547, (batch_size, src_seq_len))
dummy_tgt = torch.randint(0, 119547, (batch_size, tgt_seq_len))
dummy_attn_mask = torch.ones(batch_size, src_seq_len)

# Build a proper causal (look-ahead) mask so the decoder cannot cheat by
# attending to future tokens — without this, self-attention can reconstruct
# the output without using memory, making cross-attention artificially collapse.
# Shape must be [batch, tgt, tgt] — MHA.forward does unsqueeze(1) internally
# to get [batch, 1, tgt, tgt] which broadcasts across heads.
look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1)
look_ahead_mask = look_ahead_mask.unsqueeze(0)  # [1, tgt, tgt]

# 5. Run a clean forward pass
with torch.no_grad():
    enc_out = model.encode(dummy_src, attention_mask=dummy_attn_mask)
    ch_enc = model.channel_encoder(enc_out)
    memory = model.channel_decoder(ch_enc)
    _ = model.decoder(dummy_tgt, memory, look_ahead_mask=look_ahead_mask, trg_padding_mask=None)

for h in handles:
    h.remove()

# 6. Cross-Attention Collapse Diagnosis
# Collapse modes:
#   (A) Uniform collapse  → entropy ≈ max_entropy, utilisation ≈ 0%
#   (B) Peaked collapse   → entropy ≈ 0, max_weight ≈ 1.0
#   (C) Healthy           → entropy in a middle range, reasonable max_weight

max_entropy = math.log(src_seq_len)
UNIFORM_THRESHOLD = 0.90   # entropy/max_entropy > this → uniform collapse
PEAKED_THRESHOLD  = 0.15   # entropy/max_entropy < this → peaked collapse

print("\n" + "=" * 65)
print("         CROSS-ATTENTION COLLAPSE DIAGNOSIS")
print("=" * 65)
print(f"{'Layer':<8} {'Avg Entropy':>14} {'Max Weight':>12} {'Eff Rank':>10}  Status")
print("-" * 65)

all_verdicts = []
for i in range(num_decoder_layers):
    if not captured_attn[i]:
        print(f"Layer {i:<3} | NO ATTENTION CAPTURED")
        continue

    # attn shape: (batch, heads, tgt_len, src_len)
    attn = captured_attn[i][0]          # first (only) batch item
    # Average over heads → (tgt_len, src_len)
    mean_attn = attn[0].mean(dim=0)

    # --- Entropy (per target token, averaged) ---
    entropy = -torch.sum(mean_attn * torch.log(mean_attn + 1e-9), dim=-1)
    avg_entropy = entropy.mean().item()
    ratio = avg_entropy / max_entropy

    # --- Max attention weight (avg over tgt tokens) ---
    max_weight = mean_attn.max(dim=-1).values.mean().item()

    # --- Effective rank via entropy of singular values ---
    # Treat mean_attn as a matrix and compute its SVD-based effective rank
    try:
        sv = torch.linalg.svdvals(mean_attn.float())
        sv_prob = sv / sv.sum()
        eff_rank = torch.exp(-torch.sum(sv_prob * torch.log(sv_prob + 1e-9))).item()
    except Exception:
        eff_rank = float('nan')

    # --- Verdict ---
    if ratio > UNIFORM_THRESHOLD:
        verdict = "⚠ UNIFORM COLLAPSE"
    elif ratio < PEAKED_THRESHOLD:
        verdict = "⚠ PEAKED COLLAPSE"
    else:
        verdict = "✓ Healthy"

    all_verdicts.append(verdict)
    print(f"  {i:<6} {avg_entropy:>12.4f}   {max_weight:>10.4f}   {eff_rank:>8.2f}  {verdict}")

print("-" * 65)
print(f"Max possible entropy (uniform over {src_seq_len} src tokens): {max_entropy:.4f}")
print()

# Summary
collapsed = [v for v in all_verdicts if "COLLAPSE" in v]
if not collapsed:
    print("SUMMARY: All decoder layers show healthy cross-attention diversity.")
else:
    print(f"SUMMARY: {len(collapsed)}/{len(all_verdicts)} layer(s) show attention collapse.")
    print("         This explains poor generation quality / repetition.")
print("=" * 65)