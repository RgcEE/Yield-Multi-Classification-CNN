# Reading Results
**Author: Reynaldo Gomez**
**Repo: Semiconductor-Engineering / Yield CNN**

---

## The Classification Report

```
              precision    recall  f1-score   support

      Center       0.90      0.95      0.93       859
       Donut       0.85      0.88      0.87       111
    Edge-Loc       0.75      0.89      0.81      1038
   Edge-Ring       0.96      0.98      0.97      1936
         Loc       0.74      0.81      0.77       718
   Near-full       0.94      0.97      0.95        30
      Random       0.89      0.90      0.90       173
     Scratch       0.79      0.82      0.80       239
        none       0.99      0.98      0.99     29486

    accuracy                           0.97     34590
   macro avg       0.87      0.91      0.89     34590
weighted avg       0.97      0.97      0.97     34590
```

---

## What Each Column Means

### support

The number of samples of that class in the validation set. Ground truth count,
how many wafers of that type actually exist.

Read this column first. It tells you how much to trust the other numbers.
- Edge-Ring: 1936 samples (high confidence metrics)
- Near-full:   30 samples (metrics have high uncertainty, see stats note below)
- Donut:      111 samples (moderate confidence)

### precision

Of all the wafers predicted as class X, what fraction actually were X.

```
precision = true_positives / (true_positives + false_positives)
```

Low precision means the model is raising false alarms.
Scratch precision 0.79 means 21% of wafers called Scratch were actually
something else (probably Edge-Loc or Loc).

Manufacturing implication: low precision wastes engineer time investigating
wafers that are not actually defective with that pattern.

### recall

Of all the wafers that actually are class X, what fraction the model found.

```
recall = true_positives / (true_positives + false_negatives)
```

Low recall means the model is missing real defects.
If Donut recall were 0.40, 60% of real Donut wafers would pass through as
something else, potentially reaching downstream processes.

Manufacturing implication: low recall is the dangerous failure mode.
Missing a real defect is worse than a false alarm.

### f1-score

The harmonic mean of precision and recall:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

The harmonic mean punishes imbalance between the two. A model with
precision=1.0 and recall=0.01 gets F1=0.02, not 0.50. F1 cannot be
cheated by being good at one and terrible at the other.

Go-to single number per class. Anything above 0.85 is solid.
Below 0.70 means the model is not reliably learning that class.

---

## The Summary Rows

### accuracy

```
accuracy = total correct predictions / total samples
```

Misleading for this dataset. The model could predict none for every single
wafer and get 85% accuracy (29,486 / 34,590). Accuracy is dominated by
the largest class. Do not use it as the headline metric.

### macro avg

Averages precision, recall, and F1 across all 9 classes, treating each
class equally regardless of sample count.

```
macro_F1 = (F1_Center + F1_Donut + ... + F1_none) / 9
```

This is the real headline metric. It gives equal weight to Near-full
(30 samples) and none (29,486 samples). If macro F1 goes up, the hard
classes improved. If only weighted F1 goes up, the model just got better at none.

Current progression:
- Baseline (plain CNN):          macro F1 = 0.80
- ResNet+Focal (20 epochs):     macro F1 = 0.87
- ResNet+Focal (40 epochs):     macro F1 = 0.89

### weighted avg

Averages metrics weighted by support (sample count). Dominated by none.
Nearly identical to accuracy. Not useful for evaluating performance on
rare defect classes. Ignore it for decision-making.

---

## Experiment-to-Experiment Comparison

```
Class       Baseline   20ep ResNet   40ep ResNet   Change (20->40)
---------   --------   -----------   -----------   ---------------
Center          0.88          0.91          0.93        +0.02
Donut           0.56          0.89          0.87        -0.02  (within uncertainty)
Edge-Loc        0.76          0.78          0.81        +0.03
Edge-Ring       0.97          0.97          0.97         0.00
Loc             0.68          0.75          0.77        +0.02
Near-full       0.85          0.93          0.95        +0.02
Random          0.76          0.90          0.90         0.00
Scratch         0.76          0.71          0.80        +0.09
none            0.99          0.98          0.99        +0.01

macro F1        0.80          0.87          0.89        +0.02
```

Key observations from 20 to 40 epochs:
- Scratch recovered from 0.71 to 0.80. The extra training time fixed what
  looked like a regression. The model needed more epochs to learn the Scratch
  pattern under Focal Loss.
- Donut dropped slightly (0.89 to 0.87), within statistical uncertainty for
  111 samples. Not a real regression.
- macro F1 +0.02, consistent improvement, confirms the model had not converged
  at 20 epochs.

The accuracy drop (97% to 97%) is negligible and expected. Accuracy is dominated
by none. As the model gets better at rare classes, it trades microscopic none
performance for meaningful rare class improvement. That is the correct direction.

---

## Statistical Uncertainty in the Metrics

F1 scores are estimates with uncertainty that depends on sample count.
The standard error of a proportion p from n samples:

```
SE = sqrt(p * (1 - p) / n)
```

Applied to the classes (95% confidence interval = 1.96 * SE):

```
none      (n=29486, F1=0.99):  SE = 0.001  -> 0.99 +/- 0.002  (very tight)
Edge-Ring (n=1936,  F1=0.97):  SE = 0.004  -> 0.97 +/- 0.008
Donut     (n=111,   F1=0.87):  SE = 0.032  -> 0.87 +/- 0.063
Near-full (n=30,    F1=0.95):  SE = 0.039  -> 0.95 +/- 0.076
```

Practical implication: a 0.02 change in Donut F1 between experiments is
within measurement noise. A 0.33 change (0.56 to 0.89) is real.
A 0.02 change in none F1 is also real because n=29,486.

This is why collecting more Donut and Near-full samples (synthetic data augmentation)
is not just about giving the model more to learn from; it also makes the
evaluation metrics more statistically reliable.

---

## The Confusion Matrix

The confusion matrix shows the full breakdown of what the model predicted
versus what was true. Run yield_evaluate.py to generate it as a PNG.

How to read it:
- Rows = true class
- Columns = predicted class
- Diagonal = correct predictions (recall per class)
- Off-diagonal = mistakes

Example cell: Row=Scratch, Col=Edge-Loc = 0.10 means 10% of actual
Scratch wafers were predicted as Edge-Loc. This is the specific confusion
pair to target with CoordConv; adding spatial position information
helps the model distinguish a scratch that crosses the edge zone from a
genuine edge-localized cluster defect.

What to look for:
- Bright diagonal means model learning well
- A column bright everywhere means model over-predicts that class
- A row dim on diagonal means model misses that class (low recall)
- Symmetric off-diagonal pairs means two classes are being confused with each other

---

## Val Loss vs Accuracy as Optimization Targets

Checkpoints are saved based on val loss, not accuracy or F1.

Val loss is differentiable; it measures the raw probability the model
assigns to correct answers across all classes simultaneously. It is the
direct signal that backpropagation optimizes.

Accuracy is non-differentiable (it is a step function: either right or
wrong per sample) and is dominated by the majority class.

F1 is also non-differentiable and requires a full inference pass to compute.

In practice: val loss improvement and macro F1 improvement are correlated
but not identical. It is possible for val loss to improve while macro F1
stays flat if the improvement is concentrated in the majority class.
This is why both (val_loss and macro_f1) are logged in tracker.py.
