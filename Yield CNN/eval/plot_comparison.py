"""
Name: Reynaldo Gomez
Last Edited: 3/29/2026

Description:
    Reads the classification reports produced by baseline_eval.py and builds
    a grouped bar chart comparing per-class F1 score for the baseline and
    final model across all 9 classes.

    Inputs:
        eval_outputs/baseline_report.txt  — from baseline_eval.py
        eval_outputs/final_report.txt     — from baseline_eval.py

    Saves:
        eval_outputs/comparison_chart.png

Usage:
    cd eval/
    python baseline_eval.py        # produces the two .txt files
    python plot_comparison.py      # produces comparison_chart.png
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUT_DIR      = Path(__file__).parent / "eval_outputs"
BASELINE_TXT = OUT_DIR / "baseline_report.txt"
FINAL_TXT    = OUT_DIR / "final_report.txt"


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_f1(report_path: Path) -> dict[str, float]:
    """
    Extracts per-class F1 scores from a sklearn classification_report text file.
    Skips summary rows (accuracy, macro avg, weighted avg).

    Returns {class_name: f1_score}.
    """
    skip = {"accuracy", "macro avg", "weighted avg"}
    f1   = {}
    for line in report_path.read_text().splitlines():
        # sklearn rows: "  ClassName    prec  recall  f1  support"
        parts = line.split()
        if len(parts) == 5 and parts[0] not in skip:
            try:
                f1[parts[0]] = float(parts[3])
            except ValueError:
                pass
        # two-word class names e.g. "Near-full" handled — they are hyphenated
        # single tokens so len==5 still holds
    return f1


# ── Load ──────────────────────────────────────────────────────────────────────

baseline_f1 = parse_f1(BASELINE_TXT)
final_f1    = parse_f1(FINAL_TXT)

# Align on shared classes, sorted alphabetically to match report order
classes = sorted(set(baseline_f1) & set(final_f1))

b_vals = [baseline_f1[c] for c in classes]
f_vals = [final_f1[c]    for c in classes]

# ── Plot ──────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="deep")
palette        = sns.color_palette("deep")
BASELINE_COLOR = palette[0]
FINAL_COLOR    = palette[1]

x     = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 5.5))

bars_b = ax.bar(x - width / 2, b_vals, width, color=BASELINE_COLOR,
                label="Baseline (plain CNN)")
bars_f = ax.bar(x + width / 2, f_vals, width, color=FINAL_COLOR,
                label="Final (SE-ResNet + Focal + LS)")

# Value labels
for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
            f"{bar.get_height():.2f}", ha="center", va="bottom",
            fontsize=7.5, color=BASELINE_COLOR, fontweight="bold")

for bar in bars_f:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
            f"{bar.get_height():.2f}", ha="center", va="bottom",
            fontsize=7.5, color=FINAL_COLOR, fontweight="bold")

# Delta labels above each pair
for i, (b, f) in enumerate(zip(b_vals, f_vals)):
    delta = f - b
    sign  = "+" if delta >= 0 else ""
    color = "#1a7a1a" if delta > 0.01 else ("#b30000" if delta < -0.01 else "#666666")
    ax.text(x[i], max(b, f) + 0.055,
            f"{sign}{delta:.2f}",
            ha="center", va="bottom", fontsize=7.5, color=color, fontweight="bold")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10, rotation=30, ha="right")
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_ylim(0, 1.15)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

ax.set_title("Per-Class F1: Baseline vs Final Model", fontsize=13,
             fontweight="bold", pad=12)
ax.legend(fontsize=9, framealpha=0.9, loc="lower right")

plt.tight_layout()
save_path = OUT_DIR / "comparison_chart.png"
plt.savefig(save_path, dpi=180, bbox_inches="tight")
print(f"Saved: {save_path}")
plt.show()
