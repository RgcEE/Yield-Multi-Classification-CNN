"""
Author: Reynaldo Gomez
Last Edited: 2/18/2026

Description:

This is a generalized multiclass classifier trained to identify wafer maps into K defect pattern classes.

The general pipeline is:
1. Loading raw data
2. Encode labels as integers
3. Split into train/validation
4. Wrap in dataset and dataloader
5. Define model
6. Define loss function
7. Define optimizer 
8. Training loop
9. Evaluation loop
10. Save best model

Description:

This script trains a multiclass CNN to classify semiconductor wafer defect patterns
from the LSWMD dataset. It loads the dataset from a Python 2-era pickle file, applies
compatibility patches to handle deprecated pandas module paths, preprocesses the wafer
maps into uniform 64x64 arrays, and trains a convolutional neural network to predict
one of 9 defect classes.

The trained model is saved as best_multiclass.pt whenever validation loss improves.
This checkpoint contains both the model weights and the class name mapping, so it can
be loaded later to make predictions on new wafer maps without retraining.

Notes:
From the LSWMD.pkl dataset we know that:
    - 811,457 total rows but 638,507 are unlabeled ([]), leaving 172,950 usable labeled rows
    - 9 classes: 8 defect types + 'none' (no defect)
    - Heavy class imbalance: 'none' has 147k samples but 'Near-full' only has 149
      - WeightedRandomSampler is crucial to prevent the model from ignoring rare classes
    - Labels are stored as nested arrays [[label]] and must be unwrapped before use
    - Wafer maps are variable size → resized to 64x64 so batches have uniform shape

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

"""
Usage:

    TRAINING:
        python yield_multi_classifier.py
        - trains for 20 epochs, saves best_multiclass.pt when val loss improves

    INFERENCE (after training):
        checkpoint = torch.load("best_multiclass.pt", map_location="cpu")
        model = WaferCNN(num_classes=len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["model"])
        model.eval()

        # Prepare a single wafer map (numpy array, any size)
        from skimage.transform import resize
        wm = resize(your_wafer_map, (64, 64), anti_aliasing=False, preserve_range=True)
        x  = torch.tensor(wm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        if x.max() > 1.5:
            x = x / 255.0

        with torch.no_grad():
            logits = model(x)                          # raw scores per class
            probs  = torch.softmax(logits, dim=1)      # convert to probabilities
            pred   = logits.argmax(dim=1).item()       # index of predicted class

        print("Predicted class:", checkpoint["classes"][pred])
        print("Confidence:     ", round(probs[0][pred].item(), 4))

"""


# ----------------------------------------------------------------------------- pandas 2.x compat patch -----------------------------------------------------------------------------
"""
 LSWMD.pkl was saved with Python 2 + pandas ~0.19.
 Pickle stores class paths like "pandas.indexes.base.Index" which no longer
 exist in pandas 2.x. We inject fake modules into sys.modules under the old
 paths so pickle's import machinery can find and redirect them.

"""
_pi = types.ModuleType("pandas.indexes")
_pi.base    = pandas.core.indexes.base
_pi.range   = pandas.core.indexes.range
_pi.multi   = pandas.core.indexes.multi
_pi.numeric = pandas.core.indexes.base   # removed in pandas 2.0, redirect to base

sys.modules["pandas.indexes"]         = _pi
sys.modules["pandas.indexes.base"]    = pandas.core.indexes.base
sys.modules["pandas.indexes.range"]   = pandas.core.indexes.range
sys.modules["pandas.indexes.multi"]   = pandas.core.indexes.multi
sys.modules["pandas.indexes.numeric"] = pandas.core.indexes.base
# -----------------------------------------------------------------------------

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE  = 64      # all wafer maps resized to 64x64
BATCH_SIZE = 128
EPOCHS     = 20
LR         = 3e-4


# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------

def _unwrap_label(val):
    """
    LSWMD stores labels as nested arrays: [['Edge-Loc']] or [[]] for unlabeled.
    This peels off nesting layers until it reaches the scalar string.
    Returns None for empty arrays (unlabeled rows) so they can be dropped.
    """
    while isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return None          # unlabeled — will be filtered out
        val = val[0]
    return str(val).strip()


def load_pkl(pkl_path: str):
    """
    Loads LSWMD.pkl and returns:
        X : np.ndarray, shape (N, 64, 64), float32, values in [0, 1]
        y : np.ndarray, shape (N,), dtype str  e.g. 'Edge-Loc', 'none', ...

    Steps:
        1. Apply pandas compat patch (done at module level above)
        2. Load with latin-1 encoding (Python 2 pickle → Python 3)
        3. Unwrap nested labels [[label]] → label
        4. Drop the 638k unlabeled rows where label == []
        5. Resize each variable-size wafer map to IMG_SIZE x IMG_SIZE
    """
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        df = pkl.load(f, encoding="latin-1")

    # Unwrap labels and drop unlabeled rows ([] entries)
    raw_labels = df["failureType"].apply(_unwrap_label)
    mask       = raw_labels.notna()                    # False for the 638k [] rows
    df         = df[mask].reset_index(drop=True)
    raw_labels = raw_labels[mask].reset_index(drop=True)

    # Convert variable-size nested-list wafer maps → uniform numpy arrays
    maps = []
    for wm in df["waferMap"]:
        arr = np.array(wm, dtype=np.float32)           # nested list: 2D array
        arr = resize(arr, (IMG_SIZE, IMG_SIZE),
                     anti_aliasing=False,
                     preserve_range=True)              # keep original pixel values
        maps.append(arr)

    X = np.stack(maps)                                 # (N, 64, 64)
    y = raw_labels.values                              # (N,) string labels

    print(f"\nLoaded {len(X):,} labeled wafers across {len(set(y))} classes")
    print(pandas.Series(y).value_counts().to_string())
    return X, y


# -----------------------------------------------------------------------------
# 2. LABEL ENCODING
# -----------------------------------------------------------------------------

def encode_labels(y_raw):
    """
    Converts string class names to integers 0..K-1.
    Sorted alphabetically so the mapping is reproducible across runs.

    Returns:
        y_idx   : np.ndarray int64, shape (N,)
        classes : list of str, index → class name  e.g. classes[3] == 'Edge-Ring'
    """
    classes    = sorted(set(y_raw))                    # deterministic ordering
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx      = np.array([cls_to_idx[v] for v in y_raw], dtype=np.int64)
    return y_idx, classes


# -----------------------------------------------------------------------------
# 3. DATASET
# -----------------------------------------------------------------------------

class WaferDataset(Dataset):
    """
    PyTorch Dataset for LSWMD wafer maps.

    __getitem__ contract:
        x : float32 tensor, shape (1, 64, 64), values in [0, 1]
        y : int, class index

    augment=True applies random rotation (0/90/180/270°) and horizontal flip.
    These are valid augmentations because most wafer defect patterns are
    rotationally symmetric — a rotated scratch is still a scratch.
    Augmentation is only applied during training, never during validation.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False):
        self.X       = X
        self.y       = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, H, W)
        k = torch.randint(0, 4, (1,)).item()           # random 0/90/180/270° rotation
        x = torch.rot90(x, k, dims=(1, 2))
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=(2,))              # random horizontal flip
        return x

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        # Normalize to [0, 1] if values are in 0-255 range
        if x.max() > 1.5:
            x = x / 255.0

        if self.augment:
            x = self._augment(x)

        return x, int(self.y[idx])


# -----------------------------------------------------------------------------
# 4. MODEL
# -----------------------------------------------------------------------------

class WaferCNN(nn.Module):
    """
    Simple but effective CNN for wafer defect classification.

    Architecture:
        4 × (Conv - ReLU - MaxPool) blocks with growing channel depth
        AdaptiveAvgPool collapses any spatial size to (1,1) — handles variable input sizes
        Linear layer maps 256 features - K class logits

    Output: raw logits (NOT softmax probabilities).
        CrossEntropyLoss handles softmax internally in a numerically stable way.
        Never add softmax here if using CrossEntropyLoss.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: (1, 64, 64) → (32, 32, 32)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: (32, 32, 32) → (64, 16, 16)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: (128, 8, 8) → (256, 1, 1)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),   # collapses spatial dims → always (256, 1, 1)
        )
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)      # (N, 256, 1, 1)
        x = x.flatten(1)          # (N, 256)
        x = self.dropout(x)
        return self.classifier(x) # (N, K)  ← logits


# -----------------------------------------------------------------------------
# 5. TRAINING & EVALUATION LOOPS
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn):
    """
    One full pass over the training set.
    The core 4-step training loop per batch:
        1. Forward pass  - logits
        2. Compute loss
        3. Backward pass - gradients
        4. Optimizer step - update weights
    """
    model.train()   # enables dropout and batchnorm training mode
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        optimizer.zero_grad(set_to_none=True)  # clear gradients from previous batch
        logits = model(x)                       # forward pass
        loss   = loss_fn(logits, y)             # cross-entropy loss
        loss.backward()                         # compute gradients via backprop
        optimizer.step()                        # update model weights

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    """
    One full pass over the validation set.
    @torch.no_grad() disables gradient tracking — faster and uses less memory.
    model.eval() disables dropout and uses running stats for batchnorm.
    No optimizer.step() — we are measuring, not learning.
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


# -----------------------------------------------------------------------------
# 6. MAIN
# -----------------------------------------------------------------------------

def main():
    # ── Load and encode --------------------------
    X_raw, y_raw    = load_pkl("Dataset/LSWMD.pkl")
    y_idx, classes  = encode_labels(y_raw)
    K               = len(classes)
    print(f"\nClasses ({K}):", classes)

    # ── Train / val split (stratified so every class appears in both) --------------------------
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_raw, y_idx,
        test_size=0.2,
        random_state=42,
        stratify=y_idx         # preserves class distribution in both splits
    )

    # ── Datasets -----------------------------------------------------------------------------
    train_ds = WaferDataset(X_tr, y_tr, augment=True)
    val_ds   = WaferDataset(X_va, y_va, augment=False)

    # ── Imbalance handling: WeightedRandomSampler --------------------------
    # Rare classes (Near-full: 149 samples) get sampled more often so every
    # training batch sees a roughly balanced class distribution.
    # This is different from class weights in the loss function:
    #   sampler  - changes which samples appear in batches
    #   loss weights - changes how hard the loss pushes on each class
    class_counts   = np.bincount(y_tr, minlength=K)
    class_weights  = 1.0 / np.maximum(class_counts, 1)  # rare - high weight
    sample_weights = class_weights[y_tr]                  # per-sample weight
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model, loss, optimizer -----------------------------------------------------------------------------
    model     = WaferCNN(num_classes=K).to(DEVICE)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Reduce LR by 0.5 if val loss doesn't improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    print(f"\nTraining on {DEVICE}")
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}\n")

    # ── Training loop -----------------------------------------------------------------------------
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, loss_fn)
        va_loss, va_acc = eval_one_epoch(model, val_loader,   loss_fn)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

        # Save checkpoint whenever val loss improves
        # We save classes alongside weights so we can map output index - class name later
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {"model": model.state_dict(), "classes": classes},
                "best_multiclass.pt"
            )
            print(f" Saved best model (val loss {va_loss:.4f})")

    print("\nTraining complete. Best val loss:", round(best_val_loss, 4))


if __name__ == "__main__":
    main()
