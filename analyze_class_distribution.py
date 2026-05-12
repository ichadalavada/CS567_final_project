#!/usr/bin/env python3
"""
Analyze and plot class distribution for every 1% pkl file in merged_datasets/.
Produces one grouped bar chart comparing all datasets side-by-side.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter
from pathlib import Path

MERGED_DIR  = Path("/Users/ishachadalavada/Desktop/ml_final_project/merged_datasets")
CLASS_NAMES = ["vehicle", "pedestrian", "cyclist", "misc"]
NUM_CLASSES = 4
OUT_FILE    = "class_distribution_1pct.png"


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_labels(data):
    """Return flat array of dominant class per image, dropping class >= NUM_CLASSES."""
    labels = []
    for anns in data["labels"]:
        if len(anns) == 0:
            continue
        class_ids = [ann[0] for ann in anns]
        most_common = Counter(class_ids).most_common(1)[0][0]
        if 0 <= most_common < NUM_CLASSES:
            labels.append(most_common)
    return np.array(labels, dtype=int)


def class_counts(labels):
    """Return list of counts for classes 0..NUM_CLASSES-1."""
    c = Counter(labels.tolist())
    return [c.get(i, 0) for i in range(NUM_CLASSES)]


def main():
    pkl_files = sorted(MERGED_DIR.glob("*_1%_train.pkl"))
    if not pkl_files:
        print(f"No *_1%_train.pkl files found in {MERGED_DIR}")
        return

    dataset_names = []
    all_counts    = []

    print(f"{'Dataset':35s}  " + "  ".join(f"{c:>10s}" for c in CLASS_NAMES) + "  {'total':>8s}")
    print("-" * 90)

    for pkl_path in pkl_files:
        data   = load_pkl(pkl_path)
        labels = extract_labels(data)
        counts = class_counts(labels)

        # Friendly name: strip path / suffix, keep stem
        name = pkl_path.stem.replace("_1%_train", "").replace("_", " + ").replace("idd", "IDD")
        dataset_names.append(name)
        all_counts.append(counts)

        row = "  ".join(f"{n:10d}" for n in counts)
        print(f"{name:35s}  {row}  {sum(counts):8d}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    n_datasets = len(dataset_names)
    n_classes  = NUM_CLASSES
    x          = np.arange(n_datasets)
    bar_width  = 0.18
    offsets    = np.linspace(-(n_classes - 1) / 2, (n_classes - 1) / 2, n_classes) * bar_width

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 1]})

    # ── Left: grouped bar chart (absolute counts) ────────────────────────────
    ax = axes[0]
    for ci, (cls_name, color, offset) in enumerate(zip(CLASS_NAMES, colors, offsets)):
        vals = [all_counts[di][ci] for di in range(n_datasets)]
        bars = ax.bar(x + offset, vals, bar_width, label=cls_name, color=color, alpha=0.85)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                        str(v), ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Sample count")
    ax.set_title("Class Distribution — 1% Merged Datasets (absolute)", fontweight="bold")
    ax.legend(title="Class", loc="upper right")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="y", alpha=0.3)

    # ── Right: stacked percentage bar chart ──────────────────────────────────
    ax2 = axes[1]
    totals = [sum(c) for c in all_counts]
    bottoms = np.zeros(n_datasets)

    for ci, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        pcts = np.array([
            (all_counts[di][ci] / totals[di] * 100) if totals[di] > 0 else 0
            for di in range(n_datasets)
        ])
        ax2.bar(x, pcts, label=cls_name, color=color, alpha=0.85, bottom=bottoms)
        for di in range(n_datasets):
            if pcts[di] > 3:
                ax2.text(di, bottoms[di] + pcts[di] / 2,
                         f"{pcts[di]:.0f}%", ha="center", va="center",
                         fontsize=7.5, color="white", fontweight="bold")
        bottoms += pcts

    ax2.set_xticks(x)
    ax2.set_xticklabels(dataset_names, rotation=35, ha="right", fontsize=9)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("Class Distribution — 1% Merged Datasets (relative)", fontweight="bold")
    ax2.legend(title="Class", loc="upper right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
