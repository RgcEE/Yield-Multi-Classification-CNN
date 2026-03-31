# Yield CNN documentation index

Author: Reynaldo Gomez
---

## Documents

**01_cnn_fundamentals.md** establishes the mathematical identity of the network as $f_\theta(x) = \hat{y}$, where $x$ is a 64×64 wafer map, $\hat{y}$ is a 9-class score vector, and $\theta$ is the full parameter set, then traces the forward pass through each transformation. Convolution is derived as a sliding dot product producing spatially localized feature maps; BatchNorm2d is explained through the normalization formula and why the learnable $\gamma$ and $\beta$ parameters recover expressive capacity after normalization. The SiLU activation, $x \cdot \sigma(x)$, is contrasted with ReLU through its derivative structure and the non-zero gradient it provides for negative activations. AdaptiveAvgPool2d, the skip connection, and the $+1$ term in the residual gradient flow are covered in full. Read this first.

**02_training_mechanics.md** covers the five-line optimization loop and the full chain from forward pass to weight update. CrossEntropyLoss and FocalLoss are derived side by side; the focal weighting term $(1 - p_t)^\gamma$ with gamma=2.0 is shown numerically to suppress the gradient contribution of the 29,486 *none* samples while preserving full-magnitude signal from hard classes such as Donut (111 samples) and Scratch (1,193). The AdamW update is unpacked by first moment, second moment, and the weight decay correction that distinguishes AdamW from standard Adam. The three learning rate schedules in use (ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts) are compared on stability and convergence behavior. WeightedRandomSampler, which adjusts sampling probabilities inversely by class count, is distinguished from loss class weighting, which adjusts the gradient magnitude post-sample.

**03_reading_results.md** establishes how to interpret the classification report produced by `sklearn.metrics.classification_report` after each evaluation pass. Precision, recall, and F1 are defined formulaically and mapped to manufacturing-relevant failure modes: low precision wastes engineering time on false alarms, low recall allows defective wafers to escape downstream. The accuracy statistic is shown to be dominated by the *none* class at 85.2% of the validation set and unsuitable as a headline metric. Macro F1, the unweighted mean across all nine classes, is established as the correct optimization target. Statistical uncertainty of each metric is derived from the binomial standard error $\sqrt{p(1-p)/n}$ and applied to each class to distinguish real improvements from measurement noise given the available support counts.

**04_dynamic_training.md** maps the "differential score analyzer" concept, which monitors per-class convergence, detects stalls, applies targeted interventions, and reverts if they cause harm, to three concrete implementations: dynamic class weighting via a per-class F1 improvement rate $\Delta F_{1,c}(t) = F_{1,c}(t) - F_{1,c}(t - W)$ with a multiplicative boost when the rate falls below a convergence threshold; CosineAnnealingWarmRestarts (SGDR) with `T_0=10, T_mult=2` to escape local minima by periodically resetting the learning rate to `eta_max`; and checkpoint-and-branch using `copy.deepcopy` on both model and optimizer state to preserve Adam's momentum estimates across saved snapshots. A second section covers instrumentation of the training loop itself via per-batch loss logging, gradient norm monitoring with `p.grad.data.norm(2)`, activation statistics via `register_forward_hook`, and Grad-CAM for spatial attribution of model decisions.

**05_batch_size_ablation.md** documents the batch size and learning rate ablation across seven runs on `yield_resnet_focal.py`. Batch sizes of 64, 128, and 256 are tested at LR values derived from the linear scaling rule $\text{LR}_\text{new} = \text{LR}_\text{old} \cdot B_\text{new} / B_\text{old}$. The 256-batch configuration consistently collapsed Scratch precision to 0.44-0.60 across two confirming runs; the mechanism, reduced gradient noise removing the implicit regularization that rare classes depend on, is derived from the relationship between batch gradient variance and generalization to sharp versus flat minima. The 64-batch configuration failed to improve on 128 due to BatchNorm instability at approximately seven samples per class per batch under WeightedRandomSampler. The macro F1 ceiling at 0.888 across all seven runs is interpreted as evidence that the model is constrained by missing representational capacity, not by hyperparameter configuration.

**06_se_coord.md** documents the SE attention and CoordConv ablation that followed the batch size experiments. Three implementation bugs are identified with their effect on result validity traced run by run: SEBlock initialized but not wired into `ResidualBlock.forward`, `in_ch=1` on a three-channel input silently discarding two channels, and `num_workers=2` on Windows. The root cause of CoordConv's macro F1 regression from 0.888 to 0.773 is derived: variable-size LSWMD source wafers rescaled to a fixed 64×64 grid produce coordinate channels whose normalized values do not correspond to consistent physical positions across the dataset, making the coordinate signal geometrically incoherent. The full ablation across `coord_only`, `se_only`, and `se_coord` variants is compared on macro F1, val loss, and per-class F1 for the three weakest classes. The decision to adopt SE attention as the new base architecture and discard CoordConv pending a data pipeline change is recorded with the supporting experimental evidence.

**07_pseudo_labeling.md** documents the pseudo-labeling run using `best_se_only.pt` on 638,507 unlabeled LSWMD wafers. 340,568 samples (53.3%) met the per-class confidence threshold before capping. The Donut class produced 59,578 pseudo-labels at mean confidence 1.0000; the saturated softmax mechanism is derived and shown to indicate extrapolation onto out-of-distribution patterns rather than genuine Donut signal. Two mitigations are applied: Donut threshold raised from 0.98 to 0.999, and a `MAX_PSEUDO_MULTIPLIER = 3` cap enforced per class. Post-cap combined dataset is approximately 416,513 samples; the three weakest classes gain directly: Scratch 954→3,320, Loc 2,874→6,414, Near-full 119→476.

**08_pseudo_labeling_experiments.md** documents three retrain experiments on the combined labeled + pseudo-labeled dataset (EXP-11, EXP-12, EXP-13). EXP-11 failed: 248,215 none pseudo-labels were not truncated by the 3x cap, making the combined dataset 87% none and causing model collapse to predicting only none. EXP-12 excluded none and strong classes (F1 > 0.90) from pseudo-labeling; Donut, Edge-Loc, and Loc improved materially but Scratch collapsed to F1 0.540 from overconfident pseudo-labels. EXP-13 excluded Scratch pseudo-labels; Scratch partially recovered to 0.620 while Donut (0.975), Edge-Loc (0.910), and Loc (0.844) held. Net result: macro F1 0.884 versus se_only baseline 0.886, statistically indistinguishable at the macro level. Best checkpoint: epoch 25, val_loss=0.0446.

---

## Experiment results summary

| Experiment | Epochs | Macro F1 | Donut F1 | Scratch F1 | Notes |
|---|---|---|---|---|---|
| Baseline (plain CNN) | 20 | 0.800 | 0.56 | 0.76 | CrossEntropyLoss, MaxPool |
| ResNet+Focal | 20 | 0.870 | 0.89 | 0.71 | ResNet + SiLU + FocalLoss |
| ResNet+Focal | 40 | 0.890 | 0.87 | 0.80 | +20 epochs, CosineAnnealingLR |
| Batch ablation best | 40 | 0.888 | 0.867 | 0.803 | batch=128, LR=3e-4, confirmed optimal |
| SE only | 40 | 0.886 | 0.873 | 0.802 | SE attention, ReduceLROnPlateau |
| SE pseudo v3 | 40 | 0.884 | 0.975 | 0.620 | pseudo-labels: Donut, Edge-Loc, Loc, Near-full only |

---

## Key references

He et al., 2015. *Deep Residual Learning for Image Recognition.*
https://arxiv.org/abs/1512.03385

Ioffe & Szegedy, 2015. *Batch Normalization: Accelerating Deep Network Training.*
https://arxiv.org/abs/1502.03167

Ramachandran et al., 2017. *Searching for Activation Functions.*
https://arxiv.org/abs/1710.05941

Lin et al., 2017. *Focal Loss for Dense Object Detection.*
https://arxiv.org/abs/1708.02002

Loshchilov & Hutter, 2017. *Decoupled Weight Decay Regularization.*
https://arxiv.org/abs/1711.05101

Loshchilov & Hutter, 2016. *SGDR: Stochastic Gradient Descent with Warm Restarts.*
https://arxiv.org/abs/1608.03983

Selvaraju et al., 2016. *Grad-CAM: Visual Explanations from Deep Networks.*
https://arxiv.org/abs/1610.02391

Goyal et al., 2017. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
https://arxiv.org/abs/1706.02677

Keskar et al., 2017. *On Large-Batch Training for Deep Learning.*
https://arxiv.org/abs/1609.04836

Cui et al., 2019. *Class-Balanced Loss Based on Effective Number of Samples.*
https://arxiv.org/abs/1901.05555

Shrivastava et al., 2016. *Training Region-based Object Detectors with Online Hard Example Mining.*
https://arxiv.org/abs/1604.03540

Shu et al., 2019. *Meta-Weight-Net: Learning an Explicit Mapping for Sample Weighting.*
https://arxiv.org/abs/1902.07379

Jaderberg et al., 2017. *Population Based Training of Neural Networks.*
https://arxiv.org/abs/1711.09846

Hu et al., 2018. *Squeeze-and-Excitation Networks.*
https://arxiv.org/abs/1709.01507

Liu et al., 2018. *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.*
https://arxiv.org/abs/1807.03247

Goodfellow, Bengio, Courville. *Deep Learning.* MIT Press.
https://www.deeplearningbook.org/

PyTorch documentation.
https://pytorch.org/docs/stable/index.html