# Dynamic Training Techniques
**Author: Reynaldo Gomez**
**Repo: Semiconductor-Engineering / Yield CNN**

This document covers adaptive training (the "differential score analyzer"
concept) and connects it to real implemented techniques. It also addresses
the training black box problem and how to open it up.

---

## The Idea, Formally Stated

The goal: monitor training signals, detect when classes are not converging,
intervene by adjusting parameters, save state, test the intervention,
revert if it makes things worse.

This maps to three real techniques:

1. Dynamic class weighting: adjust loss weights per class based on
   their convergence rate during training
2. Learning rate warm restarts: periodically reset the LR to escape
   local minima when progress stalls
3. Checkpoint-and-branch: save model state before an intervention,
   compare N epochs later, revert if worse

---

## 1. Dynamic Class Weighting

### The Math

Standard CrossEntropyLoss treats all classes equally. FocalLoss
down-weights easy samples globally. Dynamic class weighting adjusts
the per-class contribution based on how much each class has improved.

Define the per-class F1 improvement rate over a window of W epochs:

```
delta_F1_c(t) = F1_c(t) - F1_c(t - W)
```

If delta_F1_c(t) < epsilon (stalled threshold), class c gets a weight boost:

```
w_c(t+1) = w_c(t) * (1 + beta * indicator(delta_F1_c < epsilon))
```

Where beta controls aggressiveness (0.1 to 0.5 typically) and
epsilon is the convergence threshold (0.005 is reasonable: F1
improvement less than 0.5% over W=3 epochs means stalled).

These weights are passed to the loss function:

```python
class_weights = torch.tensor(w_c, dtype=torch.float32).to(DEVICE)
loss = F.cross_entropy(logits, targets, weight=class_weights)
```

### Implementation Sketch

```python
# in main(), before the training loop
class_boost = {cls: 1.0 for cls in classes}   # start equal
f1_history  = {cls: [] for cls in classes}     # track per epoch

for epoch in range(1, EPOCHS + 1):
    train_one_epoch(...)
    va_loss, va_acc = eval_one_epoch(...)

    # compute per-class F1 this epoch
    y_pred, y_true = collect_predictions(model, val_loader)
    report = classification_report(y_true, y_pred,
                                   target_names=classes,
                                   output_dict=True)

    for cls in classes:
        f1_history[cls].append(report[cls]["f1-score"])

    # check convergence every 3 epochs
    if epoch >= 4:
        for cls in classes:
            delta = f1_history[cls][-1] - f1_history[cls][-4]
            if delta < 0.005:
                class_boost[cls] = min(class_boost[cls] * 1.2, 10.0)
                print(f"  [boost] {cls}: weight -> {class_boost[cls]:.2f}")

    # rebuild loss with current weights
    weights  = torch.tensor([class_boost[c] for c in classes],
                             dtype=torch.float32).to(DEVICE)
    loss_fn  = FocalLoss(gamma=2.0, class_weights=weights)
```

The cap of 10.0 prevents runaway weight amplification.

### The Risk

If a class has low F1 because the model genuinely cannot distinguish
it (insufficient data, ambiguous patterns), boosting its weight may
destabilize training rather than fix the problem. Cap the maximum
weight and monitor val loss. If it starts rising while boosting,
the intervention is hurting not helping.

---

## 2. Learning Rate Warm Restarts (SGDR)

### The Math

The cosine annealing schedule decays LR following:

```
lr(t) = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max))
```

At t=0: lr = eta_max (full learning rate)
At t=T_max: lr = eta_min (near zero)

SGDR (Stochastic Gradient Descent with Warm Restarts) resets t to 0
after T_max epochs, jumping the LR back to eta_max. Each cycle,
T_max is multiplied by T_mult so cycles get longer over time.

### Why Restarts Help

The loss landscape is not a smooth bowl; it has many local minima.
Late in training, with a small LR, the optimizer is stuck in whatever
minimum it found. A warm restart with large LR can escape that minimum
and find a better one.

Cosine decay fine-tunes within a region, then the restart jumps to a
new region to explore, then cosine decay fine-tunes there. The final
saved checkpoint is typically from the end of the last cosine cycle
when LR is small and the model is well-settled.

### Implementation

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0   = 10,    # first restart after 10 epochs
    T_mult = 2,    # each cycle is 2x longer: 10, 20, 40...
    eta_min = 1e-6
)

# in the training loop, replace scheduler.step(va_loss) with:
scheduler.step(epoch - 1 + batch_idx / len(train_loader))
# or simply per epoch:
scheduler.step()
```

Save checkpoints at the END of each cosine cycle (when LR is at eta_min),
not just when val loss improves. The best model is often found there.

Original paper: https://arxiv.org/abs/1608.03983

---

## 3. Checkpoint-and-Branch

### The Concept

Before any intervention (weight boost, LR change, architecture change),
save a full model checkpoint. Apply the intervention. Train for N epochs.
Compare metrics. If the intervention made things worse, load the saved
checkpoint and try something else.

This is model version control applied to training dynamics.

```python
import copy

# save state before intervention
pre_intervention = {
    "model":     copy.deepcopy(model.state_dict()),
    "optimizer": copy.deepcopy(optimizer.state_dict()),
    "scheduler": copy.deepcopy(scheduler.state_dict()),
    "epoch":     epoch,
    "macro_f1":  macro_f1_before,
}
torch.save(pre_intervention, "checkpoints/pre_intervention.pt")

# apply intervention (e.g. boost class weights, change LR)
# train N more epochs
# evaluate
# if macro_f1_after < macro_f1_before - 0.01:  # worse by more than 1%
#     model.load_state_dict(pre_intervention["model"])
#     optimizer.load_state_dict(pre_intervention["optimizer"])
#     print("Reverted: intervention made things worse")
```

The key: save the optimizer state too, not just the model weights.
The optimizer has momentum estimates (m_hat, v_hat in Adam) that contain
history. Loading only model weights restarts the optimizer from scratch,
which can cause a sudden loss spike.

---

## 4. Opening the Black Box

Training feels like a black box because only the epoch-level summary is visible.
There are several ways to look inside.

### Per-Batch Loss Logging

Log the loss every N batches instead of just per epoch. This shows
whether loss decreases smoothly within an epoch or oscillates.

```python
for batch_idx, (x, y) in enumerate(loader):
    ...
    loss = loss_fn(logits, y)
    loss.backward()
    optimizer.step()

    if batch_idx % 50 == 0:
        print(f"  Batch {batch_idx}/{len(loader)} loss {loss.item():.4f}")
```

### Gradient Norm Monitoring

The norm of the gradients tells whether the optimizer is receiving
useful signal or noise.

```python
# after loss.backward(), before optimizer.step():
total_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        total_norm += p.grad.data.norm(2).item() ** 2
total_norm = total_norm ** 0.5
print(f"  Gradient norm: {total_norm:.4f}")
```

If gradient norm explodes (>> 10) or vanishes (< 0.001), training is
in trouble. Gradient clipping is the fix:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# add this line between loss.backward() and optimizer.step()
```

### Activation Statistics

Log the mean and std of activations at each layer. Activations that
saturate (mean near 0, std near 0) or explode (mean/std >> 1) indicate
BatchNorm is not doing its job or the LR is too high.

```python
# register a forward hook on a layer to inspect its output
def hook_fn(module, input, output):
    print(f"  Layer output: mean={output.mean():.4f} std={output.std():.4f}")

handle = model.features[0].register_forward_hook(hook_fn)
# run one batch through
# handle.remove()  # remove the hook after inspection
```

### Grad-CAM

Gradient-weighted Class Activation Mapping shows which pixels in the
wafer map the model looked at to make its decision.

For each prediction, compute the gradient of the class score with respect
to the last conv layer's feature maps. Regions with large gradient magnitude
are the ones the model used to decide.

If the model is predicting Edge-Ring correctly but its Grad-CAM shows it's
looking at the center of the wafer, something is wrong; it's learning a
spurious correlation. If it's looking at the ring, the model learned the
right feature.

This is the most direct way to verify the model is not cheating.

Paper: https://arxiv.org/abs/1610.02391
PyTorch implementation: pip install grad-cam

---

## 5. Connecting This to Research

The "differential score analyzer" concept is closest to:

**Class-Balanced Loss** (Cui et al., 2019)
Reweights loss by the effective number of samples per class, defined as
(1 - beta^n) / (1 - beta) where n is the sample count and beta is a
hyperparameter close to 1. More principled than 1/n weighting.
https://arxiv.org/abs/1901.05555

**Online Hard Example Mining (OHEM)** (Shrivastava et al., 2016)
During each batch, compute the loss for all samples but only backpropagate
through the K samples with the highest loss. Concentrates training on
the hardest examples dynamically without any per-class bookkeeping.
https://arxiv.org/abs/1604.03540

**Meta-Weight-Net** (Shu et al., 2019)
Trains a second small network to predict the optimal loss weight for
each sample based on a small clean validation set. The weight network
is trained alongside the main classifier. The closest existing method
to the differential score analyzer concept.
https://arxiv.org/abs/1902.07379

**Population Based Training (PBT)** (Jaderberg et al., 2017, DeepMind)
Trains a population of models simultaneously. Periodically copies weights
from better-performing models to worse-performing ones, then mutates
hyperparameters. Automated version of the checkpoint-and-branch approach.
https://arxiv.org/abs/1711.09846

The core problem with building this for a single training run: an intervention
cannot be evaluated until several more epochs pass, by which point the training
trajectory has already diverged. The solution is either (a) run multiple
parallel experiments and compare, or (b) use a held-out meta-validation set
only for evaluating interventions, keeping the main validation set clean for
final evaluation.

---

## Practical Next Step

Before building the full dynamic system, instrument the training loop
to collect the data needed to make decisions. Add this to yield_resnet_focal.py:

```python
# in main(), add per-epoch F1 tracking
epoch_log = []

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_one_epoch(...)
    va_loss, va_acc = eval_one_epoch(...)

    # compute per-class F1 every 5 epochs (expensive: full inference pass)
    if epoch % 5 == 0:
        y_pred, y_true = collect_predictions(model, val_loader)
        report = classification_report(y_true, y_pred,
                                       target_names=classes,
                                       output_dict=True)
        per_class_f1 = {c: report[c]["f1-score"] for c in classes}
        epoch_log.append({"epoch": epoch, "va_loss": va_loss, **per_class_f1})
        print(f"  Per-class F1: {per_class_f1}")

# save the log
import json
with open("epoch_log.json", "w") as f:
    json.dump(epoch_log, f, indent=2)
```

Once this data exists from several runs, it will be clear exactly when each
class stalls, which epochs are most productive, and where interventions would
have the most impact. Build the adaptive logic from observed patterns in the
data, not from theory alone.
