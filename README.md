# Semiconductor-Engineering

Author: Reynaldo Gomez
Semiconductor-Engineering

This repository is dedicated to semiconductor engineering work.

---

## Yield CNN

A multiclass CNN classifier for semiconductor wafer defect pattern recognition using the [WM-811K (LSWMD) dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/code). Trains a 4-block convolutional network across 9 defect classes (8 defect types + none) and saves the best checkpoint based on validation loss.

Reference notes written during development. Written to be understood, not just referenced; each doc explains the why behind the implementation, not just what the code does.

### Documentation

#### [01_cnn_fundamentals.md](Yield%20CNN/docs/01_cnn_fundamentals.md)
What a CNN is mathematically. The forward pass layer by layer: convolution as a dot product, BatchNorm, SiLU vs ReLU, spatial downsampling, AdaptiveAvgPool, dropout, the linear classifier. The skip connection explained via gradient flow. Read this first.

#### [02_training_mechanics.md](Yield%20CNN/docs/02_training_mechanics.md)
The five-line training loop. CrossEntropyLoss vs FocalLoss with the math. Backpropagation and the chain rule. AdamW update equation explained term by term. Learning rate schedulers: ReduceLROnPlateau vs CosineAnnealingLR vs warm restarts. WeightedRandomSampler and why it is not the same as loss class weights. What overfitting looks like in the epoch output.

#### [03_reading_results.md](Yield%20CNN/docs/03_reading_results.md)
How to read the classification report column by column. The difference between precision, recall, F1, accuracy, macro avg, and weighted avg. Why accuracy is misleading for LSWMD. Statistical uncertainty per class based on sample count. How to compare experiments. The confusion matrix and what to look for. Why val loss is saved over accuracy.

#### [04_dynamic_training.md](Yield%20CNN/docs/04_dynamic_training.md)
The "differential score analyzer" concept mapped to real techniques: dynamic class weighting, learning rate warm restarts (SGDR), checkpoint-and-branch. How to open the training black box: per-batch loss logging, gradient norm monitoring, activation statistics, Grad-CAM. Research papers for each technique with direct links.

#### [05_batch_size_ablation.md](Yield%20CNN/docs/05_batch_size_ablation.md)
Batch size and learning rate ablation across seven runs. The mechanism behind batch=128 as the optimal configuration: implicit regularization, BatchNorm stability, and the linear scaling rule. The macro F1 ceiling at 0.888 interpreted as a representational capacity limit, not a hyperparameter problem.

#### [06_se_coord.md](Yield%20CNN/docs/06_se_coord.md)
SE attention and CoordConv ablation. Three implementation bugs documented with their effect on result validity traced run by run. Root cause of CoordConv's F1 regression derived. Decision to adopt SE attention as the new base architecture recorded with supporting experimental evidence.

#### [07_pseudo_labeling.md](Yield%20CNN/docs/07_pseudo_labeling.md)
Pseudo-labeling run on 638,507 unlabeled LSWMD wafers using `best_se_only.pt`. 340,568 samples accepted at per-class confidence thresholds before capping. Donut softmax saturation at mean confidence 1.0000 derived as extrapolation onto out-of-distribution patterns, not genuine signal. Two mitigations applied: Donut threshold raised to 0.999 and a 3x per-class cap enforced. Post-cap combined dataset approximately 416,513 samples; Scratch, Loc, and Near-full gain directly.

#### [08_pseudo_labeling_experiments.md](Yield%20CNN/docs/08_pseudo_labeling.md)
Three retrain experiments on the combined labeled + pseudo-labeled dataset. The full pseudo-label retrain collapsed to predicting only none: uncapped none pseudo-labels made the combined dataset 87% none. Excluding none and strong-F1 classes from pseudo-labeling recovered Donut (0.975), Edge-Loc (0.910), and Loc (0.844) but collapsed Scratch to 0.540 from overconfident pseudo-labels. Excluding Scratch pseudo-labels as well recovered Scratch partially to 0.620 while Donut and Edge-Loc held. Net macro F1 0.884 versus se_only baseline 0.886, statistically indistinguishable.

---

### Experiment results

| Experiment | Epochs | Macro F1 | Donut F1 | Scratch F1 | Notes |
|---|---|---|---|---|---|
| Baseline (plain CNN) | 20 | 0.800 | 0.56 | 0.76 | CrossEntropyLoss, MaxPool |
| ResNet+Focal | 20 | 0.870 | 0.89 | 0.71 | ResNet + SiLU + FocalLoss |
| ResNet+Focal | 40 | 0.890 | 0.87 | 0.80 | +20 epochs, CosineAnnealingLR |
| Batch ablation best | 40 | 0.888 | 0.867 | 0.803 | batch=128, LR=3e-4, confirmed optimal |
| SE only | 40 | 0.886 | 0.873 | 0.802 | SE attention, ReduceLROnPlateau |
| SE pseudo v3 | 40 | 0.884 | 0.975 | 0.620 | pseudo-labels: Donut, Edge-Loc, Loc, Near-full only |
| SE pseudo v3 + threshold | — | **0.930** | 0.940 | 0.820 | confidence ≥ 0.7; 95.2% auto-classified, 4.8% abstained |

---

### Confidence thresholding

Evaluating the SE pseudo v3 checkpoint with a confidence threshold of 0.70 on the 34,590-sample validation set produces macro F1 0.930 on the 32,928 samples the model classifies (95.2%). The 1,662 samples below threshold (4.8%) are abstained rather than forced to a low-confidence prediction.

Per-class results on auto-classified samples:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Center | 0.97 | 0.97 | 0.97 | 811 |
| Donut | 0.92 | 0.96 | 0.94 | 96 |
| Edge-Loc | 0.84 | 0.96 | 0.90 | 899 |
| Edge-Ring | 0.99 | 0.98 | 0.99 | 1,818 |
| Loc | 0.84 | 0.91 | 0.88 | 584 |
| Near-full | 0.96 | 1.00 | 0.98 | 23 |
| Random | 0.90 | 0.96 | 0.93 | 163 |
| Scratch | 0.74 | 0.93 | 0.82 | 204 |
| none | 1.00 | 0.99 | 0.99 | 28,330 |

Scratch precision at 0.74 remains the weakest figure; low-confidence Scratch predictions are the primary driver of the abstention pool. Macro F1 gain from 0.884 to 0.930 reflects selective abstention removing the hardest cases, not an improvement in the underlying model.

---

### Evaluation visualizations

All visualizations below are generated by `yield_evaluate.py` and `plot_comparison.py` against the held-out 34,590-sample validation split (random_state=42) using the `best_se_only_ls.pt` checkpoint.

#### Model comparison

![Per-class F1: Baseline vs Final](Yield%20CNN/eval/eval_outputs/comparison_chart.png)

Grouped bar chart of per-class F1 for the plain CNN baseline (macro F1 0.800) and the final SE-ResNet + FocalLoss + label smoothing model (macro F1 0.870 uncapped, 0.930 with threshold). Delta labels show the gain or loss per class. Donut improves by +0.31; Scratch and Loc show the largest remaining gaps.

#### Confusion matrix

![Confusion matrix](Yield%20CNN/eval/eval_outputs/confusion_matrix.png)

Predicted vs. actual class counts on the full validation set. The diagonal is correct predictions. Off-diagonal entries show where the model confuses classes: Edge-Loc and Loc are the primary sources of cross-class error; the none class (29,486 samples) is nearly diagonal.

#### Per-class precision, recall, and F1

![Per-class metrics](Yield%20CNN/eval/eval_outputs/per_class_metrics.png)

Bar chart of precision, recall, and F1 for each of the 9 classes. Scratch has the lowest precision (0.60) due to over-prediction; Donut recall (0.90) is recovered relative to baseline. Near-full F1 (0.92) is strong despite its 30-sample support.

#### Confidence distribution

![Confidence histogram](Yield%20CNN/eval/eval_outputs/confidence_histogram.png)

Softmax confidence distribution separated by correct and incorrect predictions. Correct predictions concentrate near 1.0; incorrect predictions spread across the full range with a secondary peak at low confidence. The gap between the two distributions is what the 0.70 threshold exploits to reduce forced low-confidence classifications.

#### Confidence threshold analysis

![Threshold analysis](Yield%20CNN/eval/eval_outputs/threshold_analysis.png)

Macro F1 (left axis) and abstention rate (right axis) as a function of confidence threshold swept from 0 to 1. F1 rises as the threshold increases because hard cases are abstained rather than misclassified. At threshold 0.70 the model classifies 95.2% of samples at macro F1 0.930; tightening further gains diminishing F1 return at accelerating abstention cost.

#### Sample predictions

![Sample predictions](Yield%20CNN/eval/eval_outputs/sample_predictions.png)

Grid of wafer map images drawn from the validation set. Each cell shows the predicted label (top) and true label (bottom); correct predictions are labeled in green, errors in red. Edge cases such as ambiguous Scratch and Loc patterns are overrepresented in the error rows.

#### Grad-CAM attention maps

![Grad-CAM](Yield%20CNN/eval/eval_outputs/gradcam.png)

Gradient-weighted Class Activation Maps overlaid on wafer map images, one row per defect class. Warm regions indicate the spatial areas that most influenced the model's class prediction. Center activations concentrate at the die center; Edge-Ring activations trace the wafer perimeter; Scratch activations follow the linear defect streak, confirming the model attends to geometrically meaningful features.

---

### Key references

Wu et al., 2014. *WM-811K Wafer Map Dataset (LSWMD).*
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map/code

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