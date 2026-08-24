# TrustKnee model study

This document defines the first repeatable model experiment in the repository.
It is research software, not a medical device or a clinical validation.

## Question and priorities

The primary question is whether a sensor window can distinguish a correct
exercise execution from a wrong execution. Therefore the primary target is
binary: `Correct = 0`, `Wrong = 1`. The secondary nine-class target tests the
more difficult question of identifying the exercise and the particular wrong
movement label.

The comparison is between two complete approaches:

- The Transformer receives the order of the synchronized sensor readings.
- XGBoost receives the existing handcrafted summary features for the same
  windows.

## Input construction

The existing preprocessing keeps native sampling rates through filtering and
windowing. A 200 ms window has approximately 30 IMU samples and 252 sEMG
samples. For the Transformer, each window becomes:

```text
30 time positions × 56 values
    48 IMU channels (8 sensors × 6 channels)
     8 sEMG activity channels
```

The sEMG activity channels are produced by rectifying the filtered signal with
`abs()` and averaging it into 30 equal-duration time bins. This preserves a
time-varying muscle-activity envelope without pretending that the 1,259 Hz
sEMG waveform contains only 30 samples.

The model is:

```text
30 × 56
  -> Linear(56, 64)
  -> sinusoidal positional encoding
  -> TransformerEncoder: 2 layers, 4 heads, feed-forward 128
  -> mean over time
  -> Linear(64, classes)
```

Attention is supplied by PyTorch's tested `TransformerEncoder`; TrustKnee only
implements the input projection, positional encoding, pooling, and head.

## Leakage controls

All windows from one subject stay in one LOSO test fold. Overlapping windows
from one trial therefore cannot appear in both train and test. The inner
validation split is also subject-based. Sequence means and standard deviations
are fitted only on the training subjects in each fold. Subject IDs, trial IDs,
window indices, timestamps, labels, and demographics are not model columns.

At trial level, all window probabilities from a trial are averaged before one
trial decision is made.

## Metrics and outputs

The evaluator writes accuracy, macro-F1, balanced accuracy, per-class
precision/recall/F1, and confusion matrices. It writes window-level and
trial-level values for the Transformer, XGBoost, and a majority-class baseline.
`summary_results.csv` contains the mean and standard deviation across LOSO
folds; a best fold is not used as the headline result.

The experiment record also captures valid/skipped trial counts, label counts,
window settings, model settings, random seed, and software versions.

## References

- PyTorch, [TransformerEncoder documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder).
- XGBoost, [Python package documentation](https://xgboost.readthedocs.io/en/stable/python/).
- KneE-PAD, [source repository](https://github.com/ounospanas/KneE-PAD), used as a reference for the dataset layout and interpretation.
- [tsai](https://github.com/timeseriesAI/tsai), consulted for time-series Transformer design patterns.
- [IMU-Transformer](https://github.com/yolish/har-with-imu-transformer), consulted as an IMU sequence-classification example.
- [KneeGuard](https://github.com/KneeGuard/KneeGuard), consulted as a related IMU/sEMG rehabilitation project.

The external projects are references and building blocks, not copied ingestion
or training infrastructure. tsai and IMU-Transformer use different task/data
assumptions, while KneeGuard estimates gait-related quantities rather than
classifying the nine KneE-PAD labels.
