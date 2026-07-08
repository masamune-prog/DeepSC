#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_curves.py  —  Plot training and validation loss curves from the CSV
log produced by main.py.

Usage:
    python plot_curves.py                            # uses training_log.csv
    python plot_curves.py --log training_log.csv
    python plot_curves.py --log run1.csv run2.csv --labels "BERT" "Custom"
    python plot_curves.py --log training_log.csv --out curves.png
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless — safe on cluster nodes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Colour palette ────────────────────────────────────────────────────────────
TRAIN_COLOURS = ['#4C9BE8', '#E8834C', '#4CE87A', '#E84C8B']
VAL_COLOURS   = ['#1A5FA8', '#A84F1A', '#1AA84E', '#A81A5F']

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Plot DeepSC training curves')
parser.add_argument('--log', nargs='+', default=['training_log.csv'],
                    help='Path(s) to CSV log file(s) produced by main.py')
parser.add_argument('--labels', nargs='+', default=None,
                    help='Legend labels for each log file (must match --log count)')
parser.add_argument('--out', default='training_curves.png',
                    help='Output image path (PNG/PDF/SVG)')
parser.add_argument('--smooth', type=int, default=1,
                    help='Gaussian smoothing window (epochs). 1 = no smoothing.')
parser.add_argument('--no-val', action='store_true',
                    help='Omit validation loss curves')
parser.add_argument('--no-train', action='store_true',
                    help='Omit training loss curves')


def load_log(path):
    """Load a CSV log into (epochs, train_losses, val_losses, elapsed)."""
    epochs, train_losses, val_losses, elapsed = [], [], [], []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            train_losses.append(float(row['train_loss']))
            val_losses.append(float(row['val_loss']))
            elapsed.append(float(row['elapsed_s']))
    return (np.array(epochs),
            np.array(train_losses),
            np.array(val_losses),
            np.array(elapsed))


def gaussian_smooth(y, window):
    """Apply a simple Gaussian-weighted moving average."""
    if window <= 1 or len(y) < window:
        return y
    kernel = np.exp(-0.5 * np.linspace(-(window - 1) / 2,
                                        (window - 1) / 2, window) ** 2)
    kernel /= kernel.sum()
    pad = window // 2
    y_padded = np.concatenate([y[:pad][::-1], y, y[-pad:][::-1]])
    return np.convolve(y_padded, kernel, mode='valid')[:len(y)]


def annotate_best(ax, epochs, vals, colour):
    """Mark the best (minimum) val loss with a dashed vertical line."""
    best_idx = int(np.argmin(vals))
    best_epoch = epochs[best_idx]
    best_val = vals[best_idx]
    ax.axvline(best_epoch, color=colour, linestyle='--', linewidth=0.8, alpha=0.6)
    ax.annotate(f'best\n{best_val:.4f}',
                xy=(best_epoch, best_val),
                xytext=(best_epoch + 0.5, best_val * 1.05),
                fontsize=7, color=colour,
                arrowprops=dict(arrowstyle='->', color=colour, lw=0.8))


def plot(args):
    log_paths = args.log
    labels = args.labels or [os.path.splitext(os.path.basename(p))[0]
                             for p in log_paths]
    if len(labels) != len(log_paths):
        sys.exit('--labels count must match --log count')

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax_loss, ax_time = axes

    fig.patch.set_facecolor('#0F1117')
    for ax in axes:
        ax.set_facecolor('#1A1D27')
        ax.tick_params(colors='#CCCCCC', which='both')
        ax.xaxis.label.set_color('#CCCCCC')
        ax.yaxis.label.set_color('#CCCCCC')
        ax.title.set_color('#FFFFFF')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333344')
        ax.grid(True, color='#2A2D3A', linewidth=0.5, linestyle='--')

    # ── Per-run plots ─────────────────────────────────────────────────────────
    for i, (path, label) in enumerate(zip(log_paths, labels)):
        if not os.path.exists(path):
            print(f'WARNING: log file not found: {path}')
            continue

        epochs, train_l, val_l, elapsed = load_log(path)
        tc = TRAIN_COLOURS[i % len(TRAIN_COLOURS)]
        vc = VAL_COLOURS[i % len(VAL_COLOURS)]

        train_s = gaussian_smooth(train_l, args.smooth)
        val_s   = gaussian_smooth(val_l,   args.smooth)

        if not args.no_train:
            ax_loss.plot(epochs, train_s, color=tc, linewidth=1.5,
                         label=f'{label} — train')
            if args.smooth > 1:           # show raw as faint background
                ax_loss.plot(epochs, train_l, color=tc, linewidth=0.5,
                             alpha=0.25)

        if not args.no_val:
            ax_loss.plot(epochs, val_s, color=vc, linewidth=1.8,
                         label=f'{label} — val', linestyle='-.')
            annotate_best(ax_loss, epochs, val_s, vc)

        # ── Time subplot ──────────────────────────────────────────────────────
        cumtime = np.cumsum(elapsed) / 3600   # hours
        ax_time.plot(epochs, cumtime, color=tc, linewidth=1.5, label=label)

    # ── Loss axis ─────────────────────────────────────────────────────────────
    ax_loss.set_title('Loss Curves', fontsize=13, fontweight='bold', pad=8)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.legend(fontsize=8, facecolor='#1A1D27', edgecolor='#333344',
                   labelcolor='#CCCCCC')
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Time axis ─────────────────────────────────────────────────────────────
    ax_time.set_title('Cumulative Training Time', fontsize=13,
                      fontweight='bold', pad=8)
    ax_time.set_xlabel('Epoch')
    ax_time.set_ylabel('Wall-clock time (hours)')
    ax_time.legend(fontsize=8, facecolor='#1A1D27', edgecolor='#333344',
                   labelcolor='#CCCCCC')
    ax_time.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.suptitle('DeepSC — Training Curves', fontsize=15, fontweight='bold',
                 color='#FFFFFF', y=1.01)
    plt.savefig(args.out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    args = parser.parse_args()
    plot(args)
