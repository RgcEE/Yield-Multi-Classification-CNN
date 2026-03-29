"""
Author: Reynaldo Gomez

Grad-CAM Visualization and Confidence Thresholding + Abstention

Plan:
    Grad-CAM:
        Gradient-weighted Class Activation Mapping.
        Backpropagates the class score gradient through the final conv layer
        to produce a heatmap showing which spatial regions the model attended to.
        Overlaid on the original wafer map to visually verify the model is looking
        at defect regions and not background noise.

        Steps:
            1. Register a forward hook on the last conv layer to capture feature maps
            2. Register a backward hook to capture gradients w.r.t. those feature maps
            3. Global-average-pool the gradients → per-channel importance weights
            4. Weighted sum of feature maps → coarse heatmap
            5. Upsample heatmap to 64x64 and overlay on wafer image

    Confidence Thresholding:
        Instead of always forcing a class prediction, flag low-confidence samples
        for human review rather than auto-classifying them.
        Threshold: if max softmax probability < 0.70, output "uncertain".
        Especially important for Donut and Random which currently have poor precision.

        Expected outcome:
            Precision on all classes jumps when low-confidence predictions
            are routed to human review instead of auto-accepted.

Depends on: checkpoints/best_resnet_focal.pt, tracker.py
"""

# TODO: implement after Phase 2 experiments — Grad-CAM requires access to
#       the final conv layer of whatever architecture is best at that point
