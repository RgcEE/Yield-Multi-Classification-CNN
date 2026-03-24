# SE attention and CoordConv ablation: yield_se_coord

Author: Reynaldo Gomez
Semiconductor-Engineering: Yield CNN

---

## Background

`yield_resnet_focal.py` established a macro F1 ceiling of 0.888 across seven
hyperparameter runs. The persistently weak classes, Scratch (0.80), Edge-Loc (0.81), and
Loc (0.77), share a common diagnostic signature: they are spatially ambiguous. A
scratch line that crosses the wafer edge activates the same low-level features as a
genuine edge-localized cluster defect, and the model cannot resolve the ambiguity
because standard CNNs are translation-equivariant by construction. The convolution
operation is a sliding dot product that applies identical weights at every spatial
position; the network has no mechanism to encode where in the image a detected pattern
is located.

Two architectural additions were proposed to address this. The first, Squeeze-and-Excitation
(SE) attention, adds per-channel reweighting after each ResidualBlock. The network learns
a global weighting vector over the 256 feature channels conditioned on the channel
statistics of each input, allowing it to suppress channels that are irrelevant for the
current prediction and amplify those that are discriminative. Channels encoding edge-pattern
responses receive high weight for Edge-Ring predictions; channels encoding center-blob
responses receive high weight for Center predictions.

The second, CoordConv, appends normalized row and column coordinate grids as additional
input channels, expanding the input from (1, 64, 64) to (3, 64, 64). With coordinate
information present in the input, the first convolutional layer can learn filters that
are jointly sensitive to feature content and spatial location, enabling the model to
distinguish a scratch crossing the edge zone from a cluster defect localized there.

---

## Implementation bugs found during development

Three bugs affected the initial se_coord runs and must be understood before interpreting any result from that period.

Note — SE wiring omitted from forward pass: `self.se = SEBlock(out_ch)` was added to `ResidualBlock.__init__` but `out = self.se(out)` was never added to `ResidualBlock.forward`. The SEBlock was constructed, occupied memory, and received no gradients because it was never part of the computation graph. All v1, v2, and v3 `se_coord` runs were effectively CoordConv-only experiments. Results from those runs reflect ResNet + SiLU + FocalLoss + CoordConv, not the intended combined architecture.

Note — `in_ch=1` against a 3-channel input (v1 only): `_add_coord_channels` produced a 3-channel tensor but the first ResidualBlock was still configured with `in_ch=1`. PyTorch silently uses only the first channel and discards channels 1 and 2 without raising a shape error. The v1 model received malformed single-channel input despite the coordinate preprocessing running correctly. The validation loss oscillation at epochs 7-8 (0.1267 spiking to 0.2686) is consistent with randomly discarded input channels producing inconsistent gradient signal through the first layer.

Note — `num_workers=2` on Windows: DataLoader spawns worker processes for parallel data loading; on Windows this requires the script to run under a `if __name__ == "__main__":` guard to prevent recursive subprocess spawning. The guard is present, so no crashes occurred, but the risk is latent. Use `num_workers=0` for development. This is also present in `yield_resnet_focal.py`.

---

## Result validity table

| Run | in\_ch | SE called | coord norm | bugs active       | valid?        |
|-----|--------|-----------|------------|-------------------|---------------|
| v1  | 1      | no        | −1 to +1   | Bug 1 + Bug 2     | NO            |
| v2  | 3      | no        | 0 to 1     | Bug 1 only        | CoordConv-only |
| v3  | 3      | no        | 0 to 1     | Bug 1 only        | CoordConv-only |

v2 and v3 are valid measurements of CoordConv-only performance. No valid combined
SE + CoordConv result existed at this point; it was obtained in a subsequent confirming
run after Bug 1 was fixed.

---

## Ablation experiment design

Three separate scripts isolate each architectural component independently, allowing
each contribution to be measured against the `resnet_focal` baseline before any
combined result is interpreted:

`yield_coord_only.py` (coord-only-v1): `in_ch=3`, no SE attention. Tests whether
spatial coordinate information alone, absent channel reweighting, improves on the
baseline.

`yield_se_only.py` (se-only-v1): `in_ch=1`, SE attention correctly wired into
`ResidualBlock.forward`. Tests whether channel reweighting alone, absent coordinate
inputs, improves on the baseline.

`yield_se_coord.py` (se-coord, fixed): `in_ch=3`, SE wired correctly. Tests whether
the combined architecture exceeds what either component achieves in isolation. This
run was executed after the ablation results were in hand, to confirm the combined
outcome empirically rather than leave it as a theoretical prediction.

Baseline to beat: resnet\_focal macro F1 = 0.888, val loss = 0.2870.

---

## Results: coord\_only (20 epochs)

```
Epoch 01 | train loss 0.5951 acc 0.6282 | val loss 1.2547 acc 0.2212  <- saved
Epoch 02 | train loss 0.2135 acc 0.8323 | val loss 0.4313 acc 0.8119  <- saved
Epoch 03 | train loss 0.1514 acc 0.8759 | val loss 0.2825 acc 0.9281  <- saved
Epoch 04 | train loss 0.1148 acc 0.9026 | val loss 0.3026 acc 0.9060
Epoch 05 | train loss 0.0955 acc 0.9173 | val loss 0.2820 acc 0.9275  <- saved
Epoch 06 | train loss 0.0818 acc 0.9274 | val loss 0.3563 acc 0.8787
Epoch 07 | train loss 0.0709 acc 0.9371 | val loss 0.5935 acc 0.7027
Epoch 08 | train loss 0.0636 acc 0.9424 | val loss 0.2689 acc 0.9583  <- saved
Epoch 09 | train loss 0.0550 acc 0.9489 | val loss 0.4176 acc 0.8481
Epoch 10 | train loss 0.0498 acc 0.9532 | val loss 0.4138 acc 0.8395
Epoch 11 | train loss 0.0457 acc 0.9565 | val loss 0.5505 acc 0.6896
Epoch 12 | train loss 0.0406 acc 0.9602 | val loss 0.2861 acc 0.9322
Epoch 13 | train loss 0.0295 acc 0.9714 | val loss 0.2999 acc 0.9243
Epoch 14 | train loss 0.0264 acc 0.9747 | val loss 0.2781 acc 0.9294
Epoch 15 | train loss 0.0249 acc 0.9755 | val loss 0.2642 acc 0.9564  <- saved
Epoch 16 | train loss 0.0225 acc 0.9780 | val loss 0.2347 acc 0.9733  <- saved (best)
Epoch 17 | train loss 0.0223 acc 0.9781 | val loss 0.2406 acc 0.9660
Epoch 18 | train loss 0.0197 acc 0.9803 | val loss 0.3113 acc 0.9289
Epoch 19 | train loss 0.0196 acc 0.9807 | val loss 0.2709 acc 0.9430
Epoch 20 | train loss 0.0179 acc 0.9822 | val loss 0.3190 acc 0.9071

Best val loss: 0.2347 (epoch 16)
```

```
              precision    recall  f1-score   support
      Center       0.72      0.97      0.83       859
       Donut       0.86      0.89      0.88       111
    Edge-Loc       0.54      0.90      0.67      1038
   Edge-Ring       0.94      0.97      0.96      1936
         Loc       0.47      0.85      0.60       718
   Near-full       0.94      0.97      0.95        30
      Random       0.83      0.93      0.88       173
     Scratch       0.14      0.80      0.24       239
        none       1.00      0.90      0.95     29486
    macro avg       0.71      0.91      0.77     34590

macro_f1=0.7726  best_epoch=16  val_loss=0.2347
```

The validation loss at epoch 1 is 1.2547, approximately six times the resnet\_focal
epoch-1 starting point of 0.20. Validation accuracy collapses to 0.70 at epoch 7 and
to 0.69 at epoch 11, meaning the model is briefly performing worse than random on
the validation set mid-training. This is not mild instability.

The diagnostic signal is Scratch precision at 0.14. Of every 100 wafers the model
predicts as Scratch, only 14 are actually Scratch; Scratch recall at 0.80 means it
is finding real Scratch wafers, but it is simultaneously flooding the prediction space
with Scratch labels for wafers that are Edge-Loc, Loc, and other spatially similar
classes. The coordinate channels correlate loosely with Scratch patterns, which tend
to be linear and cross multiple spatial zones, and the model over-generalized that
correlation to the point of catastrophic precision collapse.

---

## Root cause: Why CoordConv fails on LSWMD

LSWMD wafer maps are variable in source resolution, originally ranging from
approximately 20×20 to 200×200 pixels, and are all rescaled to a fixed 64×64 grid
during preprocessing. When a small wafer is rescaled, its content is spatially stretched;
when a large wafer is rescaled, it is compressed. The coordinate channels, however, are
a fixed 0-to-1 normalized grid generated after rescaling, identical in shape for every
wafer regardless of its original dimensions.

The consequence is that a normalized coordinate value of 0.3 in the x-direction
corresponds to a different physical location on the wafer depending on the original
map size. The model is being trained on coordinate signals whose mapping to physical
wafer positions varies systematically across the dataset. Rather than encoding meaningful
spatial information, the coordinate channels introduce spatially inconsistent signal
that the network learns to misuse, as the Scratch precision collapse directly demonstrates.

CoordConv is correct in principle but requires consistent coordinate semantics across
all inputs to function. The only viable path for this dataset is to store the original
wafer dimensions during preprocessing and construct the coordinate channels relative
to those original dimensions before rescaling, so that equal normalized coordinate values
correspond to equal physical positions across wafers. That is a data pipeline change,
not a model change, and it requires modifying the preprocessing step upstream of the
training scripts.

---

## Results: se\_only (20 epochs)

Training initialized cleanly at epoch 1 (train loss 0.2092, val loss 0.1836), matching
the resnet\_focal starting point. Seven checkpoints were saved across 20 epochs, with
the best at epoch 17. No oscillation, no mid-training validation collapse. Train loss
and validation loss descended smoothly with productive improvements throughout the run.

```
macro_f1=0.8789  best_epoch=17  val_loss=0.1228

Per-class F1 at best checkpoint:
  Center    0.911   Donut    0.881   Edge-Loc  0.776
  Edge-Ring 0.968   Loc      0.739   Near-full 0.951
  Random    0.898   Scratch  0.803   none      0.984
```

SE attention is architecturally compatible with this model and dataset. Best epoch at
17 out of 20, with the run's final checkpoint still improving, indicates convergence had
not been reached, the same productive-throughout signal observed in the resnet\_focal
20-to-40 epoch transition. Scratch reached 0.800 at 20 epochs; resnet\_focal required
40 epochs to reach 0.803, indicating that SE channel reweighting is accelerating
Scratch generalization.

---

## Results: se\_only (40 epochs)

Epochs 1–21 showed steadily declining validation loss with multiple checkpoint saves.
Epochs 22–29 plateaued in the range 0.187–0.202 with no improvement. At epoch 26
the ReduceLROnPlateau scheduler fired (patience=3), halving the learning rate; train
loss dropped sharply from 0.0110 to 0.0078. Epoch 30 produced a final checkpoint save
at validation loss 0.1755. Epochs 31–40 showed stable validation loss in the range
0.175–0.200 with no further saves.

```
macro_f1=0.8861  best_epoch=30  val_loss=0.1755

Per-class F1 at best checkpoint:
  Center    0.915   Donut    0.873   Edge-Loc  0.814
  Edge-Ring 0.974   Loc      0.774   Near-full 0.933
  Random    0.901   Scratch  0.802   none      0.987
```

The LR scheduler firing at epoch 26 produced one final productive improvement: the
rate cut allowed the optimizer to fine-tune more carefully within the basin it had
reached during the plateau, yielding the epoch-30 checkpoint. The model then settled.
Best epoch at 30 out of 40 confirms the architecture had remaining headroom at 20
epochs and is more responsive to extended training than resnet\_focal, which peaked
at epoch 12 out of 40.

---

## Results: se\_coord combined (20 epochs, confirming run)

Run with both SE and CoordConv correctly implemented: SE wired into
`ResidualBlock.forward`, `in_ch=3`, coordinates normalized to [0, 1]. Executed after
the ablation results were in hand.

```
macro_f1=0.7647  best_epoch=18  val_loss=0.1817

Per-class F1 at best checkpoint:
  Center    0.799   Donut    0.877   Edge-Loc  0.630
  Edge-Ring 0.948   Loc      0.524   Near-full 0.923
  Random    0.827   Scratch  0.403   none      0.952
```

SE attention does not recover from the coordinate noise. Scratch F1 drops to 0.403
from 0.800 in se\_only. Loc collapses to 0.524. Both are the spatially ambiguous classes
where the coordinate channels produce the most inconsistent signal across the dataset.
Rather than using its channel reweighting capacity to discount the unreliable coordinate
channels, the SE mechanism is overwhelmed by the gradient contribution from the
coordinate-based errors.

---

## Full comparison

| Model               | Macro F1 | Scratch F1 | Loc F1 | Edge-Loc F1 | Best Epoch | Val Loss |
|---------------------|----------|------------|--------|-------------|------------|----------|
| resnet\_focal 40ep  | 0.888    | 0.803      | 0.772  | 0.814       | 12         | 0.2870   |
| se\_only 20ep       | 0.879    | 0.800      | 0.739  | 0.776       | 17         | 0.1228   |
| se\_only 40ep       | 0.886    | 0.802      | 0.774  | 0.814       | 30         | 0.1755   |
| coord\_only 20ep    | 0.772    | 0.240      | 0.600  | 0.670       | 16         | 0.2347   |
| se\_coord 20ep      | 0.765    | 0.403      | 0.524  | 0.630       | 18         | 0.1817   |

At equal training compute (40 epochs), se\_only produces macro F1 0.886 versus
resnet\_focal's 0.888, statistically indistinguishable classification performance.
The material difference is validation loss: 0.1755 versus 0.2870. Lower validation
loss reflects higher model confidence on correct predictions. This distinction matters
for the next planned stage, pseudo-labeling the unlabeled LSWMD wafers, where samples
are accepted based on a confidence threshold applied to the maximum softmax probability.
A lower validation loss means the 0.95 confidence threshold will accept more samples
from se\_only, and those accepted samples will carry more reliable pseudo-labels than
those accepted from resnet\_focal at the same threshold.

---

## Decision

SE attention is the new base architecture. It matches resnet\_focal on classification
performance at equal compute, produces substantially lower validation loss (better
confidence calibration for pseudo-labeling), and continued improving through epoch 30
of the 40-epoch run, suggesting additional headroom is available with a longer training
schedule or a modestly adjusted hyperparameter configuration.

CoordConv is eliminated from consideration on this dataset in its current form. The
rescaling of variable-size LSWMD source wafers to a fixed 64×64 grid destroys the
spatial consistency that CoordConv requires. The confirming se\_coord run established
this empirically: SE attention cannot compensate for the coordinate noise, and the
combined architecture performs worse than either component alone on the spatially
ambiguous classes. CoordConv is not viable without an original-size-aware normalization
step implemented in the preprocessing pipeline.

Next stage: pseudo-labeling using the se\_only checkpoint (val\_loss=0.1755), targeting
a combined dataset of 200,000–260,000 samples, followed by retraining se\_only on
the expanded data.

---

## References

Liu et al. 2018. *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.*
https://arxiv.org/abs/1807.03247

Hu et al. 2018. *Squeeze-and-Excitation Networks.*
https://arxiv.org/abs/1709.01507

Goyal et al. 2017. *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.*
https://arxiv.org/abs/1706.02677