"""
Author: Reynaldo Gomez
Last Edited: 3/18/2026

Multiclass CNN classifier for semiconductor wafer defect patterns (LSWMD dataset).
Trains on 172,950 labeled wafer maps across 9 classes (8 defect types + 'none').
Saves best_multiclass.pt to checkpoints/ whenever validation loss improves.

Usage:
    python yield_multi_classifier.py

Inference:
    checkpoint = torch.load("checkpoints/best_multiclass.pt", map_location="cpu")
    model = WaferCNN(num_classes=len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from skimage.transform import resize

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
CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best_multiclass.pt"


# --------------------------------------------------------------------------- #
# 1. DATA
# --------------------------------------------------------------------------- #

def _unwrap_label(val):
    # Labels stored as nested arrays [[label]] or [[]] for unlabeled. Unwrap to scalar.
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
        arr = resize(arr, (IMG_SIZE, IMG_SIZE), anti_aliasing=False, preserve_range=True)
        maps.append(arr)

    X = np.stack(maps)
    y = raw_labels.values

    print(f"\nLoaded {len(X):,} labeled wafers across {len(set(y))} classes")
    print(pandas.Series(y).value_counts().to_string())
    return X, y


def encode_labels(y_raw):
    # Sorted alphabetically so index → class mapping is reproducible across runs.
    classes    = sorted(set(y_raw))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx      = np.array([cls_to_idx[v] for v in y_raw], dtype=np.int64)
    return y_idx, classes


# --------------------------------------------------------------------------- #
# 2. DATASET
# --------------------------------------------------------------------------- #

class WaferDataset(Dataset):
    # Augmentation (rot90 + hflip) applied only during training.
    # Valid because defect patterns are rotationally symmetric.
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

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)
        if x.max() > 1.5:
            x = x / 255.0
        if self.augment:
            x = self._augment(x)
        return x, int(self.y[idx])


# --------------------------------------------------------------------------- #
# 3. MODEL
# --------------------------------------------------------------------------- #

class WaferCNN(nn.Module):
    """
    4x Conv->BN->ReLU->MaxPool blocks, AdaptiveAvgPool, Dropout, Linear.
    Output: raw logits (N, K). Do not add softmax; CrossEntropyLoss handles it.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,   32,  3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32,  64,  3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64,  128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
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


# --------------------------------------------------------------------------- #
# 5. MAIN
# --------------------------------------------------------------------------- #

def main():
    # 1. Load & resize wafer maps
    # 2. Encode string labels to integers
    # 3. Stratified 80/20 train/val split
    # 4. Build datasets, weighted sampler, and dataloaders
    # 5. Build model, loss, optimizer, scheduler
    # 6. Train for EPOCHS, saving checkpoint on val loss improvement
    
    X_raw, y_raw   = load_pkl(PKL_PATH)
    y_idx, classes = encode_labels(y_raw)
    K              = len(classes)
    print(f"\nClasses ({K}):", classes)

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_raw, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    train_ds = WaferDataset(X_tr, y_tr, augment=True)
    val_ds   = WaferDataset(X_va, y_va, augment=False)

    # WeightedRandomSampler ensures rare classes appear in every batch
    class_counts   = np.bincount(y_tr, minlength=K)
    class_weights  = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[y_tr]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = WaferCNN(num_classes=K).to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print(f"\nTraining on {DEVICE}")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}\n")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        va_loss, va_acc = eval_one_epoch(model, val_loader,   loss_fn)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save({"model": model.state_dict(), "classes": classes}, CHECKPOINT_PATH)
            print(f"  Saved best model (val loss {va_loss:.4f})")

    print("\nTraining complete. Best val loss:", round(best_val_loss, 4))


if __name__ == "__main__":
    main()
