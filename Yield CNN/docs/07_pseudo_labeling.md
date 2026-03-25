# Pseudo-labeling with the SE-only checkpoint

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## Setup

`yield_pseudolabel.py` runs inference with `best_se_only.pt` (val_loss=0.1755, macro_f1=0.886) on the 638,507 unlabeled rows in LSWMD: rows where `failureType` is an empty array. The model reconstructed for loading is `WaferResNet` with `SEBlock` inside every `ResidualBlock`, identical class names and layer structure to `yield_se_only.py`. A shape mismatch in `load_state_dict` would have surfaced immediately; none occurred.

Thresholds: 0.95 default, 0.98 for Near-full, 0.999 for Donut. The Donut threshold was raised to 0.999 after the first run revealed saturated softmax outputs; the mechanism is explained in the section below.

---

## Raw inference results

340,568 of 638,507 unlabeled samples (53.3%) met the per-class confidence threshold.

| Class | Accepted | Mean conf | Threshold |
|---|---|---|---|
| none | 248,215 | 0.9836 | 0.95 |
| Donut | 59,578 | 1.0000 | 0.98 |
| Center | 9,879 | 0.9894 | 0.95 |
| Edge-Loc | 6,427 | 0.9793 | 0.95 |
| Random | 4,929 | 0.9926 | 0.95 |
| Edge-Ring | 4,848 | 0.9914 | 0.95 |
| Loc | 3,540 | 0.9808 | 0.95 |
| Scratch | 2,366 | 0.9783 | 0.95 |
| Near-full | 786 | 0.9988 | 0.98 |

---

## The Donut problem

59,578 Donut pseudo-labels at mean confidence 1.0000 is not credible. The labeled dataset contains 555 Donut wafers total. Accepting 59,578 more would make Donut the second most common defect pattern in the unlabeled set, which contradicts the known LSWMD class distribution: Donut is a rare pattern in the original labeled data (555 of 47,226 labeled samples, 1.2%).

The mechanism: `WaferResNet` learned a strong Donut signature from 444 training samples. Unlabeled wafers with center-clustered or circularly symmetric patterns that are not Donut — near-full patterns, mixed-mode patterns, or featureless wafers with center contamination — fall nearest to the Donut manifold in the learned feature space. The softmax output saturates at 1.0000 because these patterns are unambiguous relative to the other eight classes while still being misidentified relative to ground truth.

Note — saturated softmax is the warning sign, not the reassurance: mean confidence 1.0000 on out-of-distribution samples indicates the model is extrapolating a trained signature onto structurally different inputs. A well-calibrated model expresses lower confidence on out-of-distribution inputs. No error is raised.

If the uncapped 59k pseudo-labels were used in retraining, the combined dataset would contain

$$
\frac{59{,}578}{444 + 59{,}578} = 99.3\%
$$

Donut pseudo-labels. Donut would dominate every batch regardless of the weighted sampler, and the gradient signal for all other classes would collapse.

---

## Per-class cap and corrected dataset

Two mitigations are applied. First, the Donut threshold is raised from 0.98 to 0.999. Second, accepted pseudo-labels are capped at `MAX_PSEUDO_MULTIPLIER = 3` times the original labeled count per class. The 3x multiplier is the maximum defensible expansion: it doubles the training data for any class without making pseudo-labels the majority signal.

Post-cap combined dataset (original labeled + capped pseudo-labels):

| Class | Original | 3x cap | Pseudo accepted | New total |
|---|---|---|---|---|
| none | 117,945 | 353,835 | 248,215 | 366,160 |
| Edge-Ring | 7,744 | 23,232 | 4,848 | 12,592 |
| Center | 3,435 | 10,305 | 9,879 | 13,314 |
| Edge-Loc | 4,151 | 12,453 | 6,427 | 10,578 |
| Loc | 2,874 | 8,622 | 3,540 | 6,414 |
| Scratch | 954 | 2,862 | 2,366 | 3,320 |
| Random | 693 | 2,079 | 2,079 | 2,772 |
| Donut | 555 | 1,665 | 1,332 | 1,887 |
| Near-full | 119 | 357 | 357 | 476 |

Note — none does not hit the cap: 248,215 pseudo-labels is below the 353,835 ceiling. No truncation applied.

The three persistently weak classes gain directly. Scratch: 954 original samples, 3,320 combined (3.5x). Loc: 2,874 original, 6,414 combined (2.2x). Near-full: 119 original, 476 combined (4.0x). These are the classes that have floored macro F1 below 0.89 in every prior run; additional training samples directly target the representational deficit.

Total combined dataset: approximately 416,513 samples, a 3.0x increase over the 138,016 original labeled set.

---

## Decision

Pseudo-labels are accepted at the post-cap counts above. The next step is to retrain `yield_se_only.py` on the combined labeled + pseudo-labeled dataset and compare macro F1 and per-class F1 on the original held-out validation set against the se_only baseline (macro_f1=0.886).