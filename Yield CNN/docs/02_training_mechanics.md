# Training Mechanics
**Author: Reynaldo Gomez**
**Repo: Semiconductor-Engineering / Yield CNN**

---

## The Training Loop in Plain English

Every epoch, for every batch of 128 wafer maps:

```
1. zero_grad   : clear gradients from the previous batch
2. forward     : pass data through the model, get logits
3. loss        : measure how wrong the predictions are
4. backward    : compute how much each weight contributed to that wrongness
5. step        : move each weight slightly in the direction that reduces loss
```

These five lines are the entire engine. Everything else in the training script
is scaffolding around them.

```python
optimizer.zero_grad(set_to_none=True)   # step 1
logits = model(x)                        # step 2
loss   = loss_fn(logits, y)              # step 3
loss.backward()                          # step 4
optimizer.step()                         # step 5
```

---

## Loss Functions

### Cross-Entropy Loss (baseline)

For a single sample:
```
L_CE = -log(p_t)
```

p_t is the probability assigned to the correct class.
- Model says 95% confident, correct:   -log(0.95) = 0.05  (small loss)
- Model says 5% confident, wrong:      -log(0.05) = 3.0   (large loss)

The problem for LSWMD: the model gets good at none (29,486 samples) early.
Every none sample contributes 0.05 loss and tiny gradient. But 29,486 tiny
gradients still dominates the update signal over 111 Donut samples at 3.0 each.
The model allocates its learning capacity to what it sees most.

### Focal Loss (resnet_focal model)

```
L_FL = -(1 - p_t)^gamma * log(p_t)
```

The (1 - p_t)^gamma term is the focal weight. It shrinks when the model
is already confident and correct.

With gamma=2.0:
- p_t = 0.95 (easy correct):  (1 - 0.95)^2 = 0.0025  (near zero)
- p_t = 0.50 (uncertain):     (1 - 0.50)^2 = 0.25    (moderate)
- p_t = 0.05 (hard wrong):    (1 - 0.05)^2 = 0.9025  (near full)

Effect: the 29k easy none samples contribute almost nothing. The optimizer
spends its entire gradient budget on Donut, Scratch, and other hard classes.
This is why Donut F1 went from 0.56 to 0.89 in one experiment.

gamma=2.0 is empirical from the original paper. Lower gamma approaches
standard cross-entropy. Higher gamma over-suppresses easy samples and
can destabilize training.

Original paper: https://arxiv.org/abs/1708.02002

---

## Backpropagation

After the forward pass computes a loss L, backprop computes the gradient
of L with respect to every parameter in the network using the chain rule.

For a weight W in layer l:
```
dL/dW_l = dL/dz_L * dz_L/dz_(L-1) * ... * dz_(l+1)/dz_l * dz_l/dW_l
```

Each dz_(k+1)/dz_k is the Jacobian of one layer's output with respect to its
input. PyTorch computes all of this automatically through its autograd engine
when loss.backward() is called.

The product of Jacobians is the vanishing gradient problem: if each term is
slightly less than 1, multiplying many together drives the gradient toward zero.
Early layers stop receiving useful signal and stop learning.
ResidualBlocks fix this by adding the +1 term (see 01_cnn_fundamentals.md).

---

## Optimizer: AdamW

After backprop computes gradients, AdamW updates each weight:

```
theta_(t+1) = theta_t - lr * m_hat_t / (sqrt(v_hat_t) + epsilon) - lr * lambda * theta_t
```

Where:
- lr (alpha) = learning rate = 3e-4 in the scripts
- m_hat_t = bias-corrected running mean of gradients (first moment)
- v_hat_t = bias-corrected running mean of squared gradients (second moment)
- epsilon = 1e-8, prevents division by zero
- lambda = weight decay = 1e-4, the "W" in AdamW

What the moments do:
- m_hat (mean): smooths the update direction across batches
- v_hat (variance): gives parameters with noisy gradients smaller updates

What weight decay does: adds -lr * lambda * theta to every update.
This pulls weights toward zero slightly every step, preventing any weight
from becoming extremely large. Equivalent to L2 regularization but applied
correctly (standard Adam applies it wrong; AdamW fixes this).

Adam maintains a running estimate of the mean and variance of each gradient.
Parameters with high gradient variance get smaller effective learning rates.
This stabilizes training when different parts of the network are learning at
different speeds, which is common with class imbalance.

AdamW paper: https://arxiv.org/abs/1711.05101

---

## Learning Rate Schedulers

The learning rate controls how large each weight update step is.
Too large: overshoots the minimum, training diverges.
Too small: never converges, training is extremely slow.
Schedulers decay the LR over training so early steps are large
(fast progress) and late steps are small (fine-tuning).

### ReduceLROnPlateau (both current models)

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)
scheduler.step(va_loss)   # called every epoch
```

Monitors val_loss. If it hasn't improved for 3 consecutive epochs,
multiplies LR by 0.5. Reactive, responds to what's actually happening.

Problem: the step-down is abrupt. A sudden LR halving can disturb
the optimizer's momentum estimates and cause instability.

### CosineAnnealingLR

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)
scheduler.step()   # no argument, steps on epoch count
```

Decays LR following a cosine curve from LR to eta_min over T_max epochs:
```
lr_t = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * t / T_max))
```

Smooth decay. No abrupt drops. Generally more stable than ReduceLROnPlateau
for longer training runs.

### CosineAnnealingWarmRestarts (for dynamic training)

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

Runs cosine decay for T_0 epochs, then resets LR to the original value.
Each restart, T_0 is multiplied by T_mult. So: 10 epochs, reset, 20 epochs,
reset, 40 epochs, reset.

The restart jolts the optimizer out of local minima. Used in SGDR:
https://arxiv.org/abs/1608.03983

---

## WeightedRandomSampler

Class imbalance in the dataset:
- none:      147,431 samples
- Near-full:     149 samples
- Ratio:          ~990:1

Without intervention, the training loader gives the model mostly none batches.
WeightedRandomSampler assigns each sample a probability of being drawn:

```python
class_weights  = 1.0 / class_counts        # rare class -> high weight
sample_weights = class_weights[y_train]    # per-sample weight
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

Near-full gets weight 1/149 = 0.0067.
none gets weight 1/117,945 = 0.0000085.
Near-full is sampled ~790x more often per sample than none.

Result: every batch sees roughly equal representation of all classes.

This is different from class weights in the loss function:
- Sampler: changes which samples appear in batches
- Loss weights: changes how hard the loss pushes on each class
Both can be used simultaneously. The sampler is already implemented.
Loss weights are an option to add to FocalLoss as an additional term.

---

## Dropout

```python
self.dropout = nn.Dropout(0.4)
```

During training: randomly zeroes 40% of the 256 features before the classifier.
During eval (model.eval()): disabled, all features active.

Forces redundant representations. If the model learns that feature 47
predicts Donut with 99% accuracy, dropout randomly removes it. The model
must learn backup features. This prevents overfitting to training set patterns
that don't generalize.

The 0.4 rate means on average 102 of the 256 features are zeroed each
forward pass. Increasing it (e.g. 0.5) adds more regularization.
Decreasing it (e.g. 0.2) allows the model more capacity.

---

## What Overfitting Looks Like

Compare epoch-by-epoch output:

```
Epoch 01 | train loss 0.45 acc 0.85 | val loss 0.38 acc 0.88   <- underfitting
Epoch 10 | train loss 0.12 acc 0.96 | val loss 0.22 acc 0.96   <- healthy
Epoch 18 | train loss 0.01 acc 0.99 | val loss 0.25 acc 0.97   <- slight overfit
Epoch 40 | train loss 0.00 acc 1.00 | val loss 0.45 acc 0.94   <- overfit
```

Signs of overfitting:
- Train loss keeps falling but val loss starts rising
- Train accuracy approaches 100% but val accuracy is lower and falling
- Large gap between train and val metrics

At epoch 20: train loss 0.0093, val loss 0.2774. The gap is large
but val accuracy is still good (0.97). Slight overfit zone but not
critical; the model generalizes well despite the train/val gap.
This is common with dropout + BatchNorm keeping things stable.
