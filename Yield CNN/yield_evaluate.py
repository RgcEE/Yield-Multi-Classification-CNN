"""
Name: Reynaldo Gomez
Last Edited: 2/19/2026

Description:

This script loads the trained WaferCNN checkpoint (best_multiclass.pt) produced
by yield_multi_classifier.py and evaluates it against the held-out validation set.
It produces four visualizations saved as PNG files:

    1. confusion_matrix.png     — predicted vs actual class counts
    2. per_class_metrics.png    — precision, recall, F1 per class as a bar chart
    3. confidence_histogram.png — distribution of model confidence on correct vs wrong predictions
    4. sample_predictions.png   — grid of wafer map images with predicted and true labels

Notes:
    - Requires best_multiclass.pt to exist (run yield_multi_classifier.py first)
    - Uses the same LSWMD.pkl loading + preprocessing as the training script
    - Val split is recreated with the same random_state=42 so it matches training exactly
    - All plots saved to ./eval_outputs/

Usage:
    python yield_evaluate.py
    - prints classification report to terminal
    - saves 4 PNG files to ./eval_outputs/
"""

import sys
import types
import pickle as pkl
from pathlib import Path

import numpy as np
import pandas
import pandas.core.indexes.base
import pandas.core.indexes.range
import pandas.core.indexes.multi

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from skimage.transform import resize

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── pandas 2.x compatibility patch (same as training script) ─────────────────
_pi         = types.ModuleType("pandas.indexes")
_pi.base    = pandas.core.indexes.base
_pi.range   = pandas.core.indexes.range
_pi.multi   = pandas.core.indexes.multi
_pi.numeric = pandas.core.indexes.base

sys.modules["pandas.indexes"]         = _pi
sys.modules["pandas.indexes.base"]    = pandas.core.indexes.base
sys.modules["pandas.indexes.range"]   = pandas.core.indexes.range
sys.modules["pandas.indexes.multi"]   = pandas.core.indexes.multi
sys.modules["pandas.indexes.numeric"] = pandas.core.indexes.base
# ─────────────────────────────────────────────────────────────────────────────

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64
OUT_DIR  = Path("eval_outputs")
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# Must match the architecture in yield_multi_classifier.py exactly.
# PyTorch saves weights but not architecture — if these don't match,
# load_state_dict() will raise a size mismatch error.
# ─────────────────────────────────────────────────────────────────────────────

class WaferCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (identical to training script)
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_label(val):
    while isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None
        val = val[0]
    return str(val).strip()


def load_val_set(pkl_path: str, random_state: int = 42):
    """
    Rebuilds the exact validation split used during training.
    random_state=42 and test_size=0.2 must match yield_multi_classifier.py
    or you will be evaluating on data the model trained on.

    Returns:
        X_va     : np.ndarray (N, 64, 64) float32
        y_va     : np.ndarray (N,) int64
        classes  : list of str, index → class name
        X_raw_va : list of raw wafer map arrays (for visualization, before resize)
    """
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        df = pkl.load(f, encoding="latin-1")

    raw_labels = df["failureType"].apply(_unwrap_label)
    mask       = raw_labels.notna()
    df         = df[mask].reset_index(drop=True)
    raw_labels = raw_labels[mask].reset_index(drop=True)

    maps = []
    raw_maps = []
    for wm in df["waferMap"]:
        arr     = np.array(wm, dtype=np.float32)
        raw_maps.append(arr)
        arr_res = resize(arr, (IMG_SIZE, IMG_SIZE),
                         anti_aliasing=False, preserve_range=True)
        maps.append(arr_res)

    X   = np.stack(maps)
    y   = raw_labels.values

    classes    = sorted(set(y))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx      = np.array([cls_to_idx[v] for v in y], dtype=np.int64)

    # Recreate the exact same train/val split as during training
    _, X_va, _, y_va, _, raw_va = train_test_split(
        X, y_idx, raw_maps,
        test_size=0.2,
        random_state=random_state,
        stratify=y_idx
    )

    return X_va, y_va, classes, raw_va


class WaferDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        if x.max() > 1.5:
            x = x / 255.0
        return x, int(self.y[idx])


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader):
    """
    Runs the model over the entire validation set.

    Returns:
        all_preds   : np.ndarray int, predicted class indices
        all_targets : np.ndarray int, true class indices
        all_probs   : np.ndarray float, shape (N, K), softmax probabilities
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)                              # (B, K) raw scores
        probs  = torch.softmax(logits, dim=1)          # (B, K) probabilities
        preds  = logits.argmax(dim=1)                  # (B,) predicted class index

        all_preds.append(preds.cpu().numpy())
        all_targets.append(np.array(y))
        all_probs.append(probs.cpu().numpy())

    return (np.concatenate(all_preds),
            np.concatenate(all_targets),
            np.concatenate(all_probs))


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Normalized confusion matrix (row = true class, col = predicted class).
    Normalized by row so each cell shows recall per class (0.0 to 1.0).
    Useful for spotting which classes get confused with each other.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                linewidths=0.5, ax=ax, vmin=0, vmax=1)

    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class",      fontsize=12)
    ax.set_title("Confusion Matrix (row-normalized recall)", fontsize=14, pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    out = OUT_DIR / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_per_class_metrics(y_true, y_pred, classes):
    """
    Bar chart of precision, recall, and F1 score for each class.

    Why these three metrics matter for imbalanced data:
        Precision = of all the times we predicted class X, how often were we right?
        Recall    = of all the actual class X samples, how many did we find?
        F1        = harmonic mean of precision and recall — penalizes if either is low

    Accuracy alone is misleading here: a model that always predicts 'none'
    gets 85% accuracy but 0% recall on every defect class.
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)

    precision = [report[c]["precision"] for c in classes]
    recall    = [report[c]["recall"]    for c in classes]
    f1        = [report[c]["f1-score"]  for c in classes]

    x    = np.arange(len(classes))
    w    = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x - w,   precision, w, label="Precision", color="#4C72B0")
    ax.bar(x,       recall,    w, label="Recall",    color="#DD8452")
    ax.bar(x + w,   f1,        w, label="F1 Score",  color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision, Recall, and F1 Score", fontsize=14, pad=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / "per_class_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_confidence_histogram(y_true, y_pred, all_probs):
    """
    Distribution of model confidence (max softmax probability) split by
    whether the prediction was correct or incorrect.

    A well-calibrated model should be:
        - High confidence on correct predictions
        - Lower / more spread confidence on incorrect predictions

    If wrong predictions are also high confidence, the model is overconfident
    and you may need more regularization or calibration.
    """
    # Confidence = probability assigned to the predicted class
    confidence = all_probs[np.arange(len(y_pred)), y_pred]
    correct    = y_pred == y_true

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(confidence[correct],  bins=40, alpha=0.7, label="Correct",   color="#55A868")
    ax.hist(confidence[~correct], bins=40, alpha=0.7, label="Incorrect", color="#C44E52")
    ax.set_xlabel("Model Confidence (max softmax probability)", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence: Correct vs Incorrect", fontsize=14, pad=15)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = OUT_DIR / "confidence_histogram.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_sample_predictions(X_raw_va, y_va, y_pred, all_probs, classes, n=24):
    """
    Grid of wafer map images showing what the model predicted vs the true label.
    Green title = correct prediction. Red title = wrong prediction.

    Samples are selected to show a mix of correct and incorrect predictions
    so the grid is informative rather than just showing easy wins.
    """
    correct_idx   = np.where(y_pred == y_va)[0]
    incorrect_idx = np.where(y_pred != y_va)[0]

    # Take half correct, half incorrect (if enough incorrect exist)
    n_wrong   = min(n // 2, len(incorrect_idx))
    n_correct = n - n_wrong

    rng      = np.random.default_rng(seed=0)
    sel_c    = rng.choice(correct_idx,   size=n_correct, replace=False)
    sel_w    = rng.choice(incorrect_idx, size=n_wrong,   replace=False) if n_wrong > 0 else []
    indices  = np.concatenate([sel_c, sel_w]).astype(int)
    rng.shuffle(indices)

    cols = 6
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax    = axes[i]
        wm    = np.array(X_raw_va[idx], dtype=np.float32)
        conf  = all_probs[idx][y_pred[idx]]
        pred_name = classes[y_pred[idx]]
        true_name = classes[y_va[idx]]
        correct   = y_pred[idx] == y_va[idx]

        ax.imshow(wm, cmap="RdYlBu_r", interpolation="nearest")
        ax.axis("off")

        color = "#1a7a1a" if correct else "#b30000"
        label = f"P: {pred_name}\nT: {true_name}\n{conf:.0%}"
        ax.set_title(label, fontsize=7.5, color=color, pad=3)

    # Hide any unused subplots
    for j in range(len(indices), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Sample Predictions  (Green = correct, Red = wrong)",
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out = OUT_DIR / "sample_predictions.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    checkpoint_path = "best_multiclass.pt"
    dataset_path    = "Dataset/LSWMD.pkl"

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    classes    = checkpoint["classes"]
    K          = len(classes)
    print(f"Classes ({K}): {classes}")

    model = WaferCNN(num_classes=K).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    # load_state_dict copies saved weights into the model.
    # map_location=DEVICE handles the case where the model was trained on GPU
    # but you are evaluating on CPU (or vice versa).

    # ── Rebuild val set ───────────────────────────────────────────────────────
    print("\nRebuilding validation set from dataset...")
    X_va, y_va, _, raw_va = load_val_set(dataset_path)
    print(f"Validation samples: {len(X_va):,}")

    val_ds     = WaferDataset(X_va, y_va)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Run inference ─────────────────────────────────────────────────────────
    print("\nRunning inference...")
    y_pred, y_true, all_probs = run_inference(model, val_loader)

    overall_acc = (y_pred == y_true).mean()
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")

    # ── Classification report (terminal) ─────────────────────────────────────
    # Shows precision, recall, F1, and support (sample count) per class.
    # This is your single most important diagnostic table.
    print("─" * 60)
    print(classification_report(y_true, y_pred, target_names=classes))
    print("─" * 60)

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nSaving plots to ./{OUT_DIR}/")
    plot_confusion_matrix(y_true, y_pred, classes)
    plot_per_class_metrics(y_true, y_pred, classes)
    plot_confidence_histogram(y_true, y_pred, all_probs)
    plot_sample_predictions(raw_va, y_true, y_pred, all_probs, classes)

    print("\nDone.")


if __name__ == "__main__":
    main()