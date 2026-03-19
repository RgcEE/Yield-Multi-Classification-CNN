# Yield CNN Documentation Index
**Author: Reynaldo Gomez**
**Repo: Semiconductor-Engineering / Yield CNN / docs**

Reference notes written during development of the LSWMD wafer defect
classifier. Written to be understood, not just referenced; each doc
explains the why behind the implementation, not just what the code does.

---

## Documents

### 01_cnn_fundamentals.md
What a CNN is mathematically. The forward pass layer by layer: convolution
as a dot product, BatchNorm, SiLU vs ReLU, spatial downsampling, AdaptiveAvgPool,
dropout, the linear classifier. The skip connection explained via gradient flow.
Read this first.

### 02_training_mechanics.md
The five-line training loop. CrossEntropyLoss vs FocalLoss with the math.
Backpropagation and the chain rule. AdamW update equation explained term by term.
Learning rate schedulers: ReduceLROnPlateau vs CosineAnnealingLR vs warm restarts.
WeightedRandomSampler and why it is not the same as loss class weights.
What overfitting looks like in the epoch output.

### 03_reading_results.md
How to read the classification report column by column. The difference between
precision, recall, F1, accuracy, macro avg, and weighted avg. Why accuracy is
misleading for LSWMD. Statistical uncertainty per class based on sample count.
How to compare experiments. The confusion matrix and what to look for.
Why val loss is saved over accuracy.

### 04_dynamic_training.md
The "differential score analyzer" concept mapped to real techniques: dynamic
class weighting, learning rate warm restarts (SGDR), checkpoint-and-branch.
How to open the training black box: per-batch loss logging, gradient norm
monitoring, activation statistics, Grad-CAM. Research papers for each technique
with direct links.

### 05_batch_size_ablation.md
Batch size and LR ablation across 7 runs on yield_resnet_focal.py.
Why 256 batch degraded Scratch performance, why 64 batch did not improve on 128,
and why the macro F1 ceiling (0.888) is an architectural limit, not a
hyperparameter limit. Confirmed optimal config: batch=128, LR=3e-4, epochs=40.

---

## Experiment Results Summary

| Experiment | Epochs | Macro F1 | Donut F1 | Scratch F1 | Notes |
|---|---|---|---|---|---|
| Baseline (plain CNN) | 20 | 0.80 | 0.56 | 0.76 | CrossEntropyLoss, MaxPool |
| ResNet+Focal | 20 | 0.87 | 0.89 | 0.71 | ResNet + SiLU + FocalLoss |
| ResNet+Focal | 40 | 0.89 | 0.87 | 0.80 | +20 epochs, CosineAnnealingLR |
| Batch ablation best | 40 | 0.888 | 0.867 | 0.803 | batch=128, LR=3e-4, confirmed optimal |

---

## Key References

**ResNet: skip connections**
He et al., 2015. Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385

**BatchNorm: activation normalization**
Ioffe & Szegedy, 2015. Batch Normalization: Accelerating Deep Network Training.
https://arxiv.org/abs/1502.03167

**SiLU / Swish: smooth activation**
Ramachandran et al., 2017. Searching for Activation Functions.
https://arxiv.org/abs/1710.05941

**Focal Loss: imbalanced classification**
Lin et al., 2017. Focal Loss for Dense Object Detection.
https://arxiv.org/abs/1708.02002

**AdamW: decoupled weight decay**
Loshchilov & Hutter, 2017. Decoupled Weight Decay Regularization.
https://arxiv.org/abs/1711.05101

**SGDR: cosine annealing with warm restarts**
Loshchilov & Hutter, 2016. SGDR: Stochastic Gradient Descent with Warm Restarts.
https://arxiv.org/abs/1608.03983

**Grad-CAM: visualizing CNN decisions**
Selvaraju et al., 2016. Grad-CAM: Visual Explanations from Deep Networks.
https://arxiv.org/abs/1610.02391

**Linear scaling rule for batch size**
Goyal et al., 2017. Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.
https://arxiv.org/abs/1706.02677

**Implicit regularization of small batch training**
Keskar et al., 2017. On Large-Batch Training for Deep Learning.
https://arxiv.org/abs/1609.04836

**Class-Balanced Loss: effective number reweighting**
Cui et al., 2019. Class-Balanced Loss Based on Effective Number of Samples.
https://arxiv.org/abs/1901.05555

**OHEM: online hard example mining**
Shrivastava et al., 2016. Training Region-based Object Detectors with Online Hard Example Mining.
https://arxiv.org/abs/1604.03540

**Meta-Weight-Net: learned loss weighting**
Shu et al., 2019. Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting.
https://arxiv.org/abs/1902.07379

**Population Based Training**
Jaderberg et al., 2017. Population Based Training of Neural Networks.
https://arxiv.org/abs/1711.09846

**Deep Learning textbook, free online**
Goodfellow, Bengio, Courville. MIT Press.
https://www.deeplearningbook.org/

**PyTorch documentation**
https://pytorch.org/docs/stable/index.html
