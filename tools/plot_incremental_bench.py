#!/usr/bin/env python3
"""Plot the IncBench CSV (fixed vs incremental reference).

    SEMPER_RUN_INCBENCH=1 SEMPER_INCBENCH_CSV=incbench.csv ./dic_tests IncBench
    python tools/plot_incremental_bench.py incbench.csv docs/images/incremental

Writes, per noise level that has data:
  error_vs_frame_sigma<s>.png     rms displacement error against frame, one panel per scenario
  coverage_vs_frame_sigma<s>.png  share of points within 0.5 px against frame
  accuracy_vs_cost.png            final-frame rms error against ms per frame
Values are averaged over noise seeds. Needs numpy and matplotlib.
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODES = ["fixed", "N=1", "N=2", "N=5", "N=10"]
COLORS = {"fixed": "#222222", "N=1": "#d62728", "N=2": "#ff7f0e", "N=5": "#2ca02c", "N=10": "#1f77b4"}


def load(path):
    """rows[(scenario, variant, sigma)][mode][frame] -> list of row dicts (one per seed)."""
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["scenario"], r["variant"], float(r["sigma"]))
            rows[key][r["mode"]][int(r["frame"])].append(r)
    return rows


def series(by_frame, field, base=None):
    frames = sorted(by_frame)
    vals = []
    for t in frames:
        rs = by_frame[t]
        v = np.mean([float(r[field]) for r in rs])
        if base is not None:
            v = v / base
        vals.append(v)
    return np.array(frames), np.array(vals)


def panel_grid(n):
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, axes.flat


def per_frame_plots(rows, out_dir, sigma, field, ylabel, fname, coverage=False):
    keys = [k for k in rows if k[2] == sigma]
    if not keys:
        return
    keys.sort(key=lambda k: (int(k[0][1:]), k[1]))
    fig, axes = panel_grid(len(keys))
    for ax, key in zip(axes, keys):
        modes = rows[key]
        base = None
        if coverage and "fixed" in modes and 1 in modes["fixed"]:
            base = max(1.0, np.mean([float(r["rc"]) for r in modes["fixed"][1]]))
        for m in MODES:
            if m not in modes:
                continue
            x, y = series(modes[m], "good" if coverage else field, base)
            if coverage:
                y = 100.0 * y
            ax.plot(x, y, color=COLORS[m], lw=1.2, label=m)
        ax.set_title(f"{key[0]} {key[1]}", fontsize=9)
        ax.set_xlabel("frame", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        if not coverage:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{ylabel}, sigma = {sigma:g} GL", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=110)
    plt.close(fig)


def accuracy_vs_cost(rows, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, modes in rows.items():
        if key[2] == 0:
            continue
        for m in MODES:
            if m not in modes:
                continue
            last = max(modes[m])
            rms = np.mean([float(r["rms_u"]) for r in modes[m][last] if int(r["rc"]) > 0] or [np.nan])
            ms = np.mean([float(r["wall_ms"]) for t in modes[m] for r in modes[m][t]])
            ax.scatter(ms, rms, color=COLORS[m], s=14, alpha=0.7)
    for m in MODES:
        ax.scatter([], [], color=COLORS[m], label=m)
    ax.set_xlabel("wall ms per frame")
    ax.set_ylabel("rms |U - U_true| at the last frame (px)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Accuracy against cost, every scenario with noise", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_cost.png"), dpi=110)
    plt.close(fig)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    out_dir = argv[2] if len(argv) > 2 else "."
    os.makedirs(out_dir, exist_ok=True)
    rows = load(argv[1])
    for sigma in sorted({k[2] for k in rows}):
        tag = f"{sigma:g}"
        per_frame_plots(rows, out_dir, sigma, "rms_u", "rms error (px)", f"error_vs_frame_sigma{tag}.png")
        per_frame_plots(rows, out_dir, sigma, "good", "% within 0.5 px", f"coverage_vs_frame_sigma{tag}.png",
                        coverage=True)
    accuracy_vs_cost(rows, out_dir)
    print(f"wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
