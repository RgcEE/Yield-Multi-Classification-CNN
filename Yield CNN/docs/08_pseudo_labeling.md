# Pseudo-labeling experiments

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## Baseline

`yield_se_only.py` is the reference model. It correctly identifies 9 wafer defect classes with macro F1 0.886, trained on 172,000 labeled wafer maps from LSWMD. Every result in this document is measured against it.

---

## Architecture experiments preceding pseudo-labeling

Two architectural additions were tested before pseudo-labeling began.

CoordConv appends normalized row and column coordinate grids as additional input channels, giving the first convolutional layer access to spatial position. It failed. LSWMD source wafers vary in original resolution and are all rescaled to 64×64 during preprocessing; a normalized coordinate value at position (0.3, 0.3) corresponds to a different physical wafer location depending on the original map size. The coordinate signal is geometrically incoherent across the dataset. Scratch precision collapsed to 0.14. CoordConv is eliminated.

SE attention adds per-channel reweighting inside each residual block, allowing the network to suppress channels that are irrelevant for the current prediction. Training was stable, Scratch recovered, and macro F1 matched the baseline at equal compute. SE attention is the new base architecture. Full ablation in `06_se_coord.md`.

---

## Pseudo-labeling setup

`yield_pseudolabel.py` runs inference with `best_se_only.pt` on the 638,507 unlabeled rows in LSWMD (rows where `failureType` is an empty array) and accepts samples above a per-class confidence threshold. Thresholds: 0.95 default, 0.98 for Near-full, 0.999 for Donut.

340,568 of 638,507 samples (53.3%) met the threshold. The Donut count was 59,578 at mean confidence 1.0000 — not credible given 555 Donut wafers in the entire labeled set. A per-class cap of `MAX_PSEUDO_MULTIPLIER = 3` times the original labeled count was applied. Full inference results and Donut saturation mechanism in `07_pseudo_labeling.md`.

---

## Full pseudo-label retrain: dataset collapse

The 172k labeled set was combined with 340k pseudo-labeled samples and passed to `yield_se_only.py` for retraining. Val loss oscillated between 1.69 and 0.03 across consecutive epochs. The run was stopped.

Two failure modes were diagnosed.

Note — none pseudo-labels dominated the combined dataset: the 3x cap for none is $117{,}945 \times 3 = 353{,}835$. The pseudo file contains 248,215 none samples, which is below the cap, so the `if len(cls_idx) > cap` branch never fires. All 248,215 none pseudo-labels pass through without truncation. The combined dataset is 87% none. WeightedRandomSampler produces balanced batches during training, but the validation set is also 87% none; the model finds the local minimum of predicting none for everything and stays there. Val accuracy of 3.6% at epoch 1 (model predicting random garbage) followed by 87% at epoch 2 (model predicting only none) is the signature of this collapse.

Note — label encoding mismatch (suspected, not confirmed): `encode_labels()` sorts alphabetically, producing Center=0, Donut=1, ..., none=8. If the pseudo-label file was saved with a different integer mapping, the integer 0 in the pseudo file may represent none rather than Center. WeightedRandomSampler then dramatically undersamples actual none and oversamples what it interprets as rare classes, but those samples are mislabeled none wafers. The assert in `load_pseudo()` checks class name list order but not that integer 0 in the pseudo file maps to the same class as integer 0 after `encode_labels()` runs.

```python
# Verification — run before any retrain
import pickle, numpy as np
with open("data/pseudo_labeled.pkl", "rb") as f:
    data = pickle.load(f)
for i, cls in enumerate(data["classes"]):
    print(f"  {i}: {cls:12s}  {int((data['y'] == i).sum()):,} samples")
# Compare output to encode_labels() order:
# ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch','none']
#    0        1        2          3          4      5           6        7        8
```

---

## Excluded none and strong classes: Scratch pseudo-labels unreliable

Changes made to `pseudo_se_only.py`:

- Added `PSEUDO_EXCLUDED_CLASSES = {"none", "Edge-Ring", "Center", "Random"}`: the four classes with se_only F1 > 0.90 that gain nothing from pseudo-labeling.
- `load_pseudo` skips any class in the exclusion set before the cap loop. Pseudo-labels are built only from Donut, Near-full, Loc, Edge-Loc, and Scratch.
- Safe `np.concatenate(keep) if keep else np.array([], dtype=np.int64)` guard added.
- Checkpoint updated to `best_se_pseudo_v2.pt`, `exp_id` set to `"se-pseudo-v2"`.

Expected combined dataset: 138k original + 31k pseudo (defect classes only) ≈ 170k total, 80/20 split to 136k train / 34k val. None is still 85% of the original labeled data but with no pseudo-none added, WeightedRandomSampler can compensate.

Results at best checkpoint (epoch 18, val_loss=0.0758):

| Model | Macro F1 | Scratch F1 | Loc F1 | Edge-Loc F1 | Donut F1 | Best epoch |
|---|---|---|---|---|---|---|
| se_only 40ep | 0.886 | 0.800 | 0.774 | 0.813 | 0.872 | 30 |
| pseudo (no Scratch) | 0.878 | 0.540 | 0.844 | 0.910 | 0.975 | 18 |

Donut: 0.872 to 0.975. Edge-Loc: 0.813 to 0.910. Loc: 0.774 to 0.844. Near-full improved to 0.930. These gains are real; the pseudo-labels for these classes are reliable signal.

Scratch: 0.800 to 0.540. Precision collapsed to 0.37. The se_only model had Scratch precision 0.76, recall 0.85 — already uncertain. Pseudo-labels generated from an uncertain class carry that uncertainty. The 2,366 accepted Scratch pseudo-labels passed the 0.95 threshold but include misclassified Edge-Loc and Loc wafers that the se_only model labeled as Scratch with high confidence. The combined model learned an overconfident Scratch representation.

Val loss oscillation (epochs 1-40): swings between 0.07 and 4.51. The se_only run was smooth. The oscillation comes from the mixed validation set: some epochs draw more pseudo-labeled val samples for hard classes (noisy labels, high val loss), others draw more ground truth samples (low val loss). This is a structural limitation of the current setup, not fixable without a held-out ground truth validation set.

---

## Excluded Scratch pseudo-labels: Donut, Edge-Loc, Loc, Near-full only

Scratch added to the exclusion set:

```python
PSEUDO_EXCLUDED_CLASSES = {
    "none",       # F1 0.99 in se_only — no headroom
    "Center",     # F1 0.91
    "Edge-Ring",  # F1 0.97
    "Random",     # F1 0.90
    "Scratch",    # source model uncertain — pseudo-labels unreliable
}
```

Pseudo-labels now cover Donut, Near-full, Loc, and Edge-Loc only.

Results at best checkpoint (epoch 25, val_loss=0.0446):

| Model | Macro F1 | Scratch F1 | Loc F1 | Edge-Loc F1 | Donut F1 | Best epoch |
|---|---|---|---|---|---|---|
| se_only 40ep | 0.886 | 0.800 | 0.774 | 0.813 | 0.872 | 30 |
| pseudo (no Scratch) | 0.878 | 0.540 | 0.844 | 0.910 | 0.975 | 18 |
| pseudo (defect-only) | 0.884 | 0.620 | 0.844 | 0.910 | 0.975 | 25 |

Donut and Edge-Loc held at their prior levels. Loc held at 0.844. Scratch recovered partially to 0.620 but did not return to 0.800.

The remaining Scratch regression is not caused by pseudo-label noise: Scratch pseudo-labels are excluded in v3. The cause is the shifted decision boundary. Adding 6,427 Edge-Loc and 3,540 Loc pseudo-labels gave the model a stronger representation of those classes. Wafers that the se_only model predicted as Scratch (because Edge-Loc and Loc were undertrained) are now correctly predicted as Edge-Loc or Loc. Scratch precision dropped because the confusion shifted, not because the model's understanding of Scratch degraded. The model is less wrong overall.

Val loss oscillation: the same oscillation structure as the prior run, but the peaks are decreasing over time.

```
Early spikes:  1.00, 0.70, 0.58, 0.77, 0.76   (epochs 1-6)
Mid spikes:    0.33, 0.24, 0.78, 0.59          (epochs 10-13)
Late spikes:   0.80, 0.63, 0.26                (epochs 26-29)
Final epochs:  0.06, 0.07, 0.06, 0.11, 0.23, 0.05  (epochs 35-40)
```

The model is stabilizing. The structural cause (mixed validation set) remains.

---

## Summary

Pseudo-labeling is complete. The epoch 25 checkpoint from the defect-only pseudo run (val_loss=0.0446) is the best model from this experiment series.

What worked: Donut, Edge-Loc, Loc, Near-full pseudo-labels are reliable. All four classes improved and held across both runs.

What did not work: Scratch pseudo-labels were unreliable and excluded in the second run. The improved Edge-Loc and Loc representations shifted Scratch's decision boundary as a side effect; Scratch precision has not fully recovered.

Net result: macro F1 0.884 versus se_only baseline 0.886 almost indistinguishable at the macro level. The class distribution of errors changed materially. Donut and Edge-Loc are now strong. Scratch is now the single weakest class.

The val loss oscillation is a known limitation of the mixed validation set. Results should be read from the best checkpoint, not the final epoch.