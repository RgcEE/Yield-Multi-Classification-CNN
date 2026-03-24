"""
Author: Reynaldo Gomez
Last Edited: 3/18/2026

Description:

This script trains a multiclass CNN to classify semiconductor wafer defect patterns
from the LSWMD dataset. It extends yield_resnet_focal.py with two targeted additions:

    1. SEBlock (Squeeze-and-Excitation) — added inside every ResidualBlock.
       After the two conv layers, SE globally pools each channel, passes through a
       small FC bottleneck, and outputs per-channel attention weights. This lets the
       network suppress uninformative channels and amplify useful ones without adding
       many parameters.

    2. CoordConv channels — two spatial coordinate maps (row and column, normalized
       to [0, 1]) are concatenated to the wafer image before entering the network.
       The first ResidualBlock now takes in_ch=3 instead of 1. Explicit position
       information helps distinguish position-dependent defects (e.g. Edge-Loc vs.
       Center) that look identical once spatial context is removed.

Everything else, data loading, augmentation, sampler, optimizer, and loss, is
identical to yield_resnet_focal.py.

Usage:
    python yield_se_coord.py
    - trains for 40 epochs, saves best checkpoint when val loss improves
    - prints per-class classification report at the end
    - appends one row to ../results.csv
"""

import sys
import time
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.transform import resize

# tracker.py lives one level up (Yield CNN/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from tracker import log_run

# --------------------------------------------------------------------------- #
# pandas 2.x compat patch
# LSWMD.pkl was saved with Python 2 + pandas ~0.19. Inject fake modules so
# pickle's import machinery can find the old paths.
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

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 64
BATCH_SIZE = 128
EPOCHS     = 20
LR         = 3e-4

PKL_PATH        = Path(__file__).parent.parent / "Dataset" / "LSWMD.pkl"
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best_se_coord.pt"


# --------------------------------------------------------------------------- #
# 1. DATA
# --------------------------------------------------------------------------- #

def _unwrap_label(val):
    """
    LSWMD stores labels as nested arrays: [['Edge-Loc']] or [[]] for unlabeled.
    Peels nesting until it reaches the scalar string.
    Returns None for empty arrays so they can be filtered out.
    """
    while isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None
        val = val[0]
    return str(val).strip()


def load_pkl(pkl_path: Path):
    with pkl_path.open("rb") as f:
        df = pkl.load(f, encoding="latin-1")

    raw_labels = df["failureType"].apply(_unwrap_label)
    mask       = raw_labels.notna()
    df         = df[mask].reset_index(drop=True)
    raw_labels = raw_labels[mask].reset_index(drop=True)

    maps = []
    for wm in df["waferMap"]:
        arr = np.array(wm, dtype=np.float32)
        arr = resize(arr, (IMG_SIZE, IMG_SIZE),
                     anti_aliasing=False, preserve_range=True)
        maps.append(arr)

    X = np.stack(maps)
    y = raw_labels.values

    print(f"\nLoaded {len(X):,} labeled wafers across {len(set(y))} classes")
    print(pandas.Series(y).value_counts().to_string())
    return X, y


def encode_labels(y_raw):
    """
    Converts string class names to integers 0..K-1.
    Sorted alphabetically so the mapping is reproducible across runs.
    """
    classes    = sorted(set(y_raw))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx      = np.array([cls_to_idx[v] for v in y_raw], dtype=np.int64)
    return y_idx, classes


class WaferDataset(Dataset):
    """
    PyTorch Dataset for LSWMD wafer maps.
    Augmentation (rot90 + hflip) is applied only during training.
    Defect patterns are rotationally symmetric so these are valid transforms.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = X
        self.y       = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        k = torch.randint(0, 4, (1,)).item()
        x = torch.rot90(x, k, dims=(1, 2))
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=(2,))
        return x

    def _add_coord_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Appends two coordinate channels (row and column normalized to [0, 1])
        to the single-channel wafer map.  Output shape: (3, H, W).
        Coord channels give the network explicit spatial awareness so it can
        distinguish position-dependent defects (e.g. Edge-Loc vs. Center).
        """
        _, h, w = x.shape
        row_coords = torch.linspace(0, 1, h, device=x.device).view(h, 1).expand(h, w)
        col_coords = torch.linspace(0, 1, w, device=x.device).view(1, w).expand(h, w)
        return torch.cat([x, row_coords.unsqueeze(0), col_coords.unsqueeze(0)], dim=0)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        if x.max() > 1.5:
            x = x / 255.0
        x = self._add_coord_channels(x)
        if self.augment:
            x = self._augment(x)
        return x, int(self.y[idx])


# --------------------------------------------------------------------------- #
# 2. MODEL
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
    Two conv layers with a skip connection.

    The skip adds the block's input directly to its output before the final
    activation: out = act(F(x) + x). Gradients flow backward through both the
    conv path and the skip path, which prevents them from vanishing in deeper nets.

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

        self.se   = SEBlock(out_ch)

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
        out = self.se(out)                       # channel attention after conv path
        return self.act(out + identity)


class WaferResNet(nn.Module):
    """
    WaferCNN rebuilt with ResidualBlocks.

    Channel progression matches the baseline (32→64→128→256). Each stage
    uses stride=2 for downsampling instead of MaxPool2d.

    Input:  (N, 3, 64, 64)  — 1 wafer channel + 2 CoordConv channels
    Output: (N, K) logits
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(3,   32,  stride=2),   # → (32, 32, 32)
            ResidualBlock(32,  64,  stride=2),   # → (64, 16, 16)
            ResidualBlock(64,  128, stride=2),   # → (128, 8, 8)
            ResidualBlock(128, 256, stride=2),   # → (256, 4, 4)
            nn.AdaptiveAvgPool2d((1, 1)),         # → (256, 1, 1)
        )
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 3. LOSS
# --------------------------------------------------------------------------- #

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017).

    focal_weight = (1 - pt)^gamma
        pt    = probability assigned to the correct class
        gamma = 2.0 (standard starting point)

    When pt ≈ 1.0 (easy correct prediction): weight ≈ 0 → near-zero contribution
    When pt ≈ 0.0 (hard wrong prediction):   weight ≈ 1 → full cross-entropy

    The ~29k "none" wafers the model already predicts correctly contribute almost
    nothing. The optimizer spends its capacity on Donut, Scratch, and other
    hard classes instead.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce           = F.cross_entropy(logits, targets, reduction="none")
        pt           = torch.exp(-ce)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * ce).mean()


# --------------------------------------------------------------------------- #
# 4. TRAINING & EVALUATION
# --------------------------------------------------------------------------- #

def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss   = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    """
    @torch.no_grad() disables gradient tracking — faster and uses less memory.
    model.eval() disables dropout and uses running stats for batchnorm.
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        logits = model(x)
        loss   = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model, loader):
    """Run inference over the full val set, return (y_pred, y_true) arrays."""
    model.eval()
    all_preds, all_targets = [], []

    for x, y in loader:
        logits = model(x.to(DEVICE))
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(np.array(y))

    return np.concatenate(all_preds), np.concatenate(all_targets)


# --------------------------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------------------------- #

def main():
    t_start = time.time()

    X_raw, y_raw   = load_pkl(PKL_PATH)
    y_idx, classes = encode_labels(y_raw)
    K              = len(classes)
    print(f"\nClasses ({K}):", classes)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_raw, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    train_ds = WaferDataset(X_tr, y_tr, augment=True)
    val_ds   = WaferDataset(X_va, y_va, augment=False)

    # WeightedRandomSampler: rare classes get sampled more often so every
    # training batch sees a roughly balanced class distribution
    class_counts   = np.bincount(y_tr, minlength=K)
    class_weights  = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # num_workers=0 on Windows: multiprocessing workers require pickling the full
    # dataset through a named pipe whose buffer is ~64 KB — far smaller than
    # 138 K wafer maps — causing OSError [Errno 22] / truncated-pickle crashes.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=0, pin_memory=True)

    model     = WaferResNet(num_classes=K).to(DEVICE)
    loss_fn   = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print(f"\nTraining on {DEVICE}")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}\n")

    best_val_loss = float("inf")
    best_epoch    = 1

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        va_loss, va_acc = eval_one_epoch(model, val_loader,   loss_fn)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_epoch    = epoch
            torch.save(
                {"model": model.state_dict(), "classes": classes},
                CHECKPOINT_PATH,
            )
            print(f"  Saved best model (val loss {va_loss:.4f})")

    train_time = time.time() - t_start
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}  (epoch {best_epoch})")

    # Per-class metrics
    y_pred, y_true = collect_predictions(model, val_loader)
    report         = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    per_class_f1   = {cls: report[cls]["f1-score"] for cls in classes}
    macro_f1       = report["macro avg"]["f1-score"]
    val_acc        = float((y_pred == y_true).mean())

    print("\n" + classification_report(y_true, y_pred, target_names=classes))

    log_run(
        exp_id       = "se-coord-v1",
        model        = "WaferResNet",
        loss_fn      = "FocalLoss(gamma=2)",
        epochs       = EPOCHS,
        lr           = LR,
        batch_size   = BATCH_SIZE,
        augmentation = "rot90+hflip",
        val_accuracy = val_acc,
        val_loss     = best_val_loss,
        macro_f1     = macro_f1,
        per_class_f1 = per_class_f1,
        best_epoch   = best_epoch,
        train_time_s = train_time,
        checkpoint   = CHECKPOINT_PATH.relative_to(Path(__file__).parent.parent).as_posix(),
        notes        = "ResidualBlocks + SiLU + FocalLoss + SEBlock + CoordConv",
    )


if __name__ == "__main__":
    main()
