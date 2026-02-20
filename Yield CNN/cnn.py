"""
SimpleCNN Reference (PyTorch)
Purpose:
  - Easy-to-read CNN for wafer map multiclass classification.
  - Designed to be a clean reference you can reuse.

Assumptions:
  - Input x is a single-channel wafer map: shape (N, 1, H, W)
  - Labels y are integers: 0..(K-1)

Key idea:
  - Feature extractor: Conv -> ReLU -> Pool (repeat)
  - Classifier head: Global Average Pool -> Linear -> logits
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()

        # ----------------------------
        # Feature extractor (CNN body)
        # ----------------------------
        # Each "block" does:
        #   Conv2d: learns local patterns
        #   ReLU: non-linearity
        #   MaxPool: downsamples (H,W) by 2 -> reduces compute, increases receptive field
        self.features = nn.Sequential(
            # Block 1: in_channels -> 32 feature maps
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (H,W) -> (H/2, W/2)

            # Block 2: 32 -> 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (H/2,W/2) -> (H/4, W/4)

            # Block 3: 64 -> 128 feature maps
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # (H/4,W/4) -> (H/8, W/8)

            # Block 4: 128 -> 256 feature maps
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            # Global Average Pool:
            # Converts (N, 256, h, w) -> (N, 256, 1, 1)
            # This avoids a huge flatten layer and reduces overfitting.
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # ----------------------------
        # Classifier (CNN head)
        # ----------------------------
        # After global average pooling, we have 256 numbers per sample.
        # Linear maps 256 -> num_classes (logits).
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          x: (N, 1, H, W)
          returns logits: (N, num_classes)
        """
        x = self.features(x)         # (N, 256, 1, 1)
        x = x.flatten(start_dim=1)   # (N, 256)
        logits = self.classifier(x)  # (N, num_classes)
        return logits


# ----------------------------
# Minimal usage example
# ----------------------------
if __name__ == "__main__":
    # Example: K=9 defect classes, wafer maps 64x64
    K = 9
    model = SimpleCNN(num_classes=K, in_channels=1)

    # Fake batch: N=8 samples
    x = torch.randn(8, 1, 64, 64)
    logits = model(x)

    print("logits shape:", logits.shape)  # (8, 9)

    # Training note:
    # Use CrossEntropyLoss for multiclass classification.
    # CrossEntropyLoss expects:
    #   logits: (N, K)
    #   labels: (N,) with integer class indices
    y = torch.randint(0, K, (8,))
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)

    print("loss:", float(loss))
