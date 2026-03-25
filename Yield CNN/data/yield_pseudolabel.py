"""
Author: Reynaldo Gomez

Pseudo-Labeling (Self-Training)

Plan:
    1. Load trained checkpoint (best_se_only.pt)
    2. Load the unlabeled rows from LSWMD.pkl
    3. Run inference on each unlabeled wafer map
    4. Accept predictions where confidence >= threshold per class
       - Default: 0.95 for all classes
       - Higher threshold for weak classes (Donut, Near-full): 0.98
    5. Append accepted pseudo-labeled samples to the training set
    6. Retrain on original labeled + accepted pseudo-labeled data
    7. Save new checkpoint and log to results.csv

Expected outcome:
    - Meaningful improvement on well-predicted classes (none, Edge-Ring)
    - Run only after Phase 2 so pseudo-labels come from the strongest model

Depends on: tracker.py, checkpoints/best_se_only.pt
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize

# tracker.py lives one level up (Yield CNN/)
sys.path.insert(0, str(Path(__file__).parent.parent))

# --------------------------------------------------------------------------- #
# pandas 2.x compat patch
# --------------------------------------------------------------------------- #
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
# --------------------------------------------------------------------------- #

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 64

PKL_PATH        = Path(__file__).parent.parent / "Dataset" / "LSWMD.pkl"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best_se_only.pt"
PSEUDO_OUT_PATH = Path(__file__).parent.parent / "data" / "pseudo_labeled.pkl"

# Per-class confidence thresholds.
# Donut and Near-full are weak classes for this model — require higher confidence
# before accepting a pseudo-label to avoid polluting the training set.
DEFAULT_THRESHOLD = 0.95
# Per-class overrides. Donut at 0.999: first run showed mean confidence 1.0000,
# indicating saturated softmax on ambiguous patterns, not genuine Donut signal.
# Near-full at 0.98: weak class, require extra certainty before accepting.
PER_CLASS_THRESHOLD = {
    "Donut":     0.999,
    "Near-full": 0.98,
}

# Cap pseudo-labels at this multiple of the original labeled count per class.
# Prevents any single class from dominating the combined dataset.
MAX_PSEUDO_MULTIPLIER = 3
LABELED_CLASS_COUNTS = {
    "none":      117945,
    "Edge-Ring":   7744,
    "Center":      3435,
    "Edge-Loc":    4151,
    "Loc":         2874,
    "Scratch":      954,
    "Random":       693,
    "Donut":        555,
    "Near-full":    119,
}


# --------------------------------------------------------------------------- #
# MODEL — exact copy from yield_se_only.py so load_state_dict() succeeds
# --------------------------------------------------------------------------- #

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block (Hu et al., 2018).

    Globally pools the feature map (squeeze), passes through a small FC
    bottleneck (excitation), then scales each channel by its learned attention
    weight. This lets the network suppress uninformative channels and amplify
    useful ones without adding many parameters.

    reduction=16 is the standard bottleneck ratio from the original paper.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection and SE channel attention.

    The skip adds the block's input directly to its output before the final
    activation: out = act(SE(F(x)) + x). SE attention is applied after the
    conv path and before the residual add.

    stride=2 handles spatial downsampling (replaces MaxPool from the baseline).
    When stride != 1 or channels change, the skip uses a 1x1 conv to match dims.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.SiLU(inplace=True)

        self.se = SEBlock(out_ch)

        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.act(out + identity)


class WaferResNet(nn.Module):
    """
    WaferResNet with SE attention in every block, no CoordConv.

    Channel progression: 32→64→128→256. Each stage uses stride=2 for
    downsampling. First block takes in_ch=1 (raw wafer map only).

    Input:  (N, 1, 64, 64)
    Output: (N, K) logits
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(1,   32,  stride=2),
            ResidualBlock(32,  64,  stride=2),
            ResidualBlock(64,  128, stride=2),
            ResidualBlock(128, 256, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# DATA
# --------------------------------------------------------------------------- #

def _unwrap_label(val):
    while isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None
        val = val[0]
    return str(val).strip()


def load_unlabeled(pkl_path: Path):
    """
    Returns (X_unlabeled, indices) — wafer maps whose failureType is empty.
    Resizes each map to IMG_SIZE x IMG_SIZE to match the training preprocessing.
    """
    with pkl_path.open("rb") as f:
        df = pkl.load(f, encoding="latin-1")

    raw_labels = df["failureType"].apply(_unwrap_label)
    mask_unlabeled = raw_labels.isna()
    df_unlabeled   = df[mask_unlabeled].reset_index(drop=True)

    print(f"Unlabeled rows: {len(df_unlabeled):,}")

    maps = []
    for wm in df_unlabeled["waferMap"]:
        arr = np.array(wm, dtype=np.float32)
        arr = resize(arr, (IMG_SIZE, IMG_SIZE),
                     anti_aliasing=False, preserve_range=True)
        maps.append(arr)

    return np.stack(maps)


class UnlabeledDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        if x.max() > 1.5:
            x = x / 255.0
        return x, idx   # return idx so we can map back to the original array


# --------------------------------------------------------------------------- #
# INFERENCE
# --------------------------------------------------------------------------- #

@torch.no_grad()
def run_inference(model, loader):
    """Returns (probs, pred_classes, original_indices) as numpy arrays."""
    model.eval()
    all_probs, all_preds, all_idx = [], [], []

    for x, idx in loader:
        logits = model(x.to(DEVICE))
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.append(preds)
        all_idx.append(idx.numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_preds),
        np.concatenate(all_idx),
    )


def apply_thresholds(probs, preds, classes):
    """
    For each prediction, check if max confidence >= the class-specific threshold.
    Returns a boolean mask of accepted samples.
    """
    thresholds = np.array([
        PER_CLASS_THRESHOLD.get(classes[i], DEFAULT_THRESHOLD)
        for i in range(len(classes))
    ])
    conf = probs.max(axis=1)
    thr  = thresholds[preds]
    return conf >= thr


def apply_class_caps(X, y_preds, conf, classes):
    """
    Cap accepted pseudo-labels at MAX_PSEUDO_MULTIPLIER * original labeled count per class.
    Samples are taken in order of decreasing confidence so the highest-confidence
    pseudo-labels survive the cap.
    """
    caps = {cls: LABELED_CLASS_COUNTS.get(cls, 0) * MAX_PSEUDO_MULTIPLIER
            for cls in classes}

    # sort by confidence descending so the cap keeps the most confident samples
    order = np.argsort(-conf)
    X_sorted    = X[order]
    y_sorted    = y_preds[order]
    conf_sorted = conf[order]

    per_class_count = {cls: 0 for cls in classes}
    keep = []
    for i in range(len(y_sorted)):
        cls = classes[y_sorted[i]]
        if per_class_count[cls] < caps[cls]:
            keep.append(i)
            per_class_count[cls] += 1

    keep = np.array(keep)
    return X_sorted[keep], y_sorted[keep], conf_sorted[keep]


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt    = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    classes = ckpt["classes"]
    K       = len(classes)
    print(f"Classes ({K}): {classes}")

    model = WaferResNet(num_classes=K).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    print("Weights loaded successfully.\n")

    # Load unlabeled data
    X_unlabeled = load_unlabeled(PKL_PATH)

    loader = DataLoader(
        UnlabeledDataset(X_unlabeled),
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    # Inference
    print(f"Running inference on {len(X_unlabeled):,} unlabeled wafers...")
    probs, preds, orig_idx = run_inference(model, loader)

    # Apply thresholds
    conf_all    = probs.max(axis=1)
    accept_mask = apply_thresholds(probs, preds, classes)
    X_thresh    = X_unlabeled[orig_idx[accept_mask]]
    y_thresh    = preds[accept_mask]
    conf_thresh = conf_all[accept_mask]

    print(f"\nPost-threshold: {accept_mask.sum():,} / {len(X_unlabeled):,} "
          f"({100 * accept_mask.mean():.1f}%)\n")

    # Apply per-class cap
    X_accepted, y_accepted, conf_accepted = apply_class_caps(
        X_thresh, y_thresh, conf_thresh, classes
    )

    print(f"Post-cap:       {len(X_accepted):,} samples\n")

    # Class distribution of accepted samples
    print("Class distribution of accepted pseudo-labels (post-cap):")
    print(f"  {'Class':<20} {'Count':>8}  {'Mean conf':>10}  {'Threshold':>10}  {'Cap':>8}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*8}")
    for i, cls in enumerate(classes):
        cls_mask  = y_accepted == i
        count     = cls_mask.sum()
        mean_c    = conf_accepted[cls_mask].mean() if count > 0 else float("nan")
        thr       = PER_CLASS_THRESHOLD.get(cls, DEFAULT_THRESHOLD)
        cap       = LABELED_CLASS_COUNTS.get(cls, 0) * MAX_PSEUDO_MULTIPLIER
        print(f"  {cls:<20} {count:>8,}  {mean_c:>10.4f}  {thr:>10.3f}  {cap:>8,}")

    # Save
    PSEUDO_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PSEUDO_OUT_PATH.open("wb") as f:
        pkl.dump({"X": X_accepted, "y": y_accepted, "classes": classes}, f)
    print(f"\nSaved {len(X_accepted):,} pseudo-labels to {PSEUDO_OUT_PATH}")


if __name__ == "__main__":
    main()
