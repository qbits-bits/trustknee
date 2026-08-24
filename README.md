# TrustKnee

**An Interpretable and Confidence-Aware Minimal-Sensor System for Home-Based Knee Rehabilitation Assessment**

[![Dataset: KneE-PAD](https://img.shields.io/badge/Dataset-KneE--PAD-blue)](https://doi.org/10.1038/s41597-025-04963-4)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TrustKnee is a multi-modal sEMG + IMU signal processing and machine learning framework designed for accurate, explainable, and safe postural assessment during home-based knee rehabilitation exercises. Built on the open-source **KneE-PAD** dataset (*Kasnesis et al., Scientific Data, 2025*), the pipeline combines minimal sensor selection with calibrated uncertainty estimation and interpretable feature attributions.

The model study now compares a Transformer with an XGBoost baseline. Binary
Correct-versus-Wrong classification is the primary task; the nine KneE-PAD
labels are a secondary task. Results use leave-one-subject-out (LOSO) folds and
are reported for both individual windows and whole trials.

---

## Key Features

- **Multi-Modal Signal Processing:** Zero-phase 4th-order Butterworth low-pass filtering (6 Hz) for IMU DC-gravity retention; bandpass (20–450 Hz) + 50 Hz notch filtering for sEMG.
- **Multi-Rate Windowing Engine:** Native time-domain continuous slicing (200 ms windows, 50% overlap) maintaining hardware-synchronized IMU (~148 Hz) and sEMG (~1259 Hz) sample rates without interpolation artifacts.
- **Leave-One-Subject-Out (LOSO) Validation:** Strict participant-independent cross-validation protocol across 31 subjects to prevent inter-window temporal leakage.
- **Calibrated Uncertainty & Safety Failsafe:** Platt scaling / isotonic regression probability calibration to flag low-confidence predictions and suppress misleading feedback.
- **Explainable AI (XAI):** TreeSHAP and Integrated Gradients feature attributions mapped to plain-language patient corrective guidance.
- **Minimal Sensor Selection:** Sequential Backward Elimination (SBE) framework to identify optimal minimal sensor subsets for low-cost clinical deployment.

---

## Pipeline Overview

```
raw .npy trials (IMU + sEMG)
  ├── ingestion (lazy manifest builder, corrupt file validation)
  ├── filtering (zero-phase Butterworth LP @ 6 Hz for IMU, BP 20-450 Hz + 50 Hz notch for sEMG)
  ├── windowing (200 ms, 50% overlap, multi-rate time-sync)
  ├── feature extraction (ROM, RMS, jerk, MAV, IEMG, WL, MNF)
  ├── classification & calibration (Random Forest / XGBoost / PyTorch LSTM with Platt scaling)
  └── XAI & feedback (TreeSHAP feature attributions → natural language patient cues)
```

---

## Dataset Specifications
The KneE-PAD study contains 31 participants, eight wearable sensors, three
exercises, nine movement labels, and 2,086 trials. These counts describe the
source dataset; the local preparation report records the valid and skipped
trials actually used in an experiment.

The proposal also discusses confidence scores, explanations, patient feedback,
smaller sensor setups, and testing on new participants. Those parts remain
planned work; the initial Transformer/XGBoost comparison is now implemented.

The pipeline operates on the **KneE-PAD** dataset (31 subjects, 2,086 trials, 8 Delsys Trigno Avanti wireless sensors):

| Parameter | Specification |
|---|---|
| **Sensors** | 8 bilateral lower limb sensors (Rectus Femoris, Hamstrings, Tibialis Anterior, Gastrocnemius) |
| **Modalities** | 8 sEMG channels (~1259.26 Hz) + 48 IMU channels (~148.15 Hz: 3-axis accel + 3-axis gyro) |
| **Exercises** | 3 exercises × 3 execution variants (1 correct + 2 compensatory errors) = 9 labels |
| **Classes** | Squats (0–2), Seated Leg Extensions (3–5), Gait / Walking (6–8) |

| Folder or file | What it does |
| --- | --- |
| `src/config.py` | Stores sensor settings, labels, and filter/window settings |
| `src/ingestion/` | Reads metadata and validates trial files |
| `src/preprocessing/` | Cleans and synchronizes IMU/sEMG windows |
| `src/features/` | Calculates handcrafted signal features |
| `src/models/` | Builds model inputs, trains models, and evaluates LOSO folds |
| `tests/` | Runs pipeline and model tests on synthetic data |
| `docs/` | Project diagrams and model-study documentation |

Place the uncompressed dataset under `data/raw/`:

```
data/raw/
├── participants.csv
├── labels.csv
├── placement.csv
├── sensors.csv
└── dataset/
    └── Subject_<id>/<label_id>/Trial_<number>/
        ├── imu.npy    # shape (48, T) — 8 sensors × 6 channels
        └── emg.npy    # shape (8, T)  — 8 sensors × 1 channel
```

## Quick Start

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup and Docker instructions.

```bash
# Environment setup & dependency installation
bash setup.sh && source .venv/bin/activate

# Execute test suite
pytest -q

# Run feature extraction pipeline
python src/features/ml_features.py   # outputs data/processed/kneepad_features.csv
```

---

## Citation

If you use this codebase or the underlying dataset, please cite the KneE-PAD paper:

```bibtex
@article{Kasnesis2025KneEPAD,
  author    = {Kasnesis, Panagiotis and Plavoukou, Theodora and Konstantoudakis, Konstantinos and Moustris, Alexandros and Margaritis, Dimitrios and Moustakas, Konstantinos and Patrikakis, Charalampos Z.},
  title     = {A Knee Rehabilitation Exercises Dataset for Postural Assessment using Wearable Devices},
  journal   = {Scientific Data},
  volume    = {12},
  number    = {610},
  year      = {2025},
  doi       = {10.1038/s41597-025-04963-4}
}
```

---

## License

[MIT](LICENSE)

## Prepare metadata and run the model study

The raw trial archive and metadata text file are kept outside Git. If the
archive does not already contain the four CSV files expected by the reader,
prepare a local data root with:

```bash
python -m src.models.prepare_dataset \
    --dataset-root /path/to/extracted/dataset \
    --metadata-file /path/to/metadata.txt \
    --output-dir data/raw
```

The command validates the 31 participant IDs, nine label IDs, sensor
placements, signal shapes, and duration matching. It writes a
`preparation_report.json` and reports missing or malformed trials. It does not
invent metadata when the supplied files are incomplete.

Then run the comparison:

```bash
python -m src.models.evaluate --data-root data/raw --output-dir reports
```

For an automated smoke run, add `--quick`. The normal run writes
`fold_results.csv`, `summary_results.csv`, `per_class_results.csv`, confusion
matrix CSVs, `config.json`, and `experiment_record.md`.

The Transformer input is 30 time positions by 56 values: 48 IMU channels and
8 sEMG activity values. Each sEMG value is made by taking the absolute value
of the filtered waveform and averaging it in one of 30 equal-duration bins;
the high-frequency sEMG waveform is not simply interpolated down to 30 points.
XGBoost receives only the existing handcrafted feature columns. IDs, timestamps,
and label metadata remain outside both model inputs.

If Docker is installed, the test image can be built with:

```bash
docker build --target tester -t trustknee-tester .
docker build -t trustknee .
docker run --rm trustknee
```

Docker is optional. The first build needs an internet connection to download
packages.

## How the code works

The reader makes a list of all trials and joins each trial with its participant
and label information. It checks that files are not empty, have the expected
shape, and have matching durations.

The IMU signal is smoothed. Optional settings can remove slow signal changes.
The sEMG signal is cleaned by keeping its useful range and removing electrical
noise.

The code then makes 200 millisecond sections with 50% overlap. A new section
starts every 100 milliseconds. Each section has about 30 IMU samples and 252
sEMG samples. The feature code then calculates values such as average size,
peak, power, movement change, range of motion, and signal frequency.

## Future work and ownership

The names below come from the project proposal and commit history. They describe
planned ownership, not completed work.

| Person | Planned area |
| --- | --- |
| Nihal Kumar | Project structure, data reading, signal cleaning, connecting the parts, and future testing on new participants |
| Kaushal Bhatt | Feature creation, basic machine-learning models, and finding a smaller sensor setup |
| Shlok S. Limbhare | Neural-network models, confidence scores, and model explanations |
| Pakki Divya Ankitha | User feedback, data graphs, testing, and documentation |

The planned order is to study the dataset, try basic models, try sequence models,
check whether confidence scores are useful, add explanations and feedback, and
then test on participants who were not used for training.

## Model-study references

The implementation uses the official [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder)
and the standard [XGBoost Python interface](https://xgboost.readthedocs.io/en/stable/python/).
The [KneE-PAD repository](https://github.com/ounospanas/KneE-PAD) is used as a
dataset interpretation reference. The project keeps its own ingestion and
evaluation code so that input construction, leakage controls, and metrics stay
explicit.

## Testing the three historical PR areas

The remote copy has only `origin/main`; separate GitHub PR references were not
available. These commits represent the three requested areas:

1. `c6f85cb` — signal cleaning, time sections, and feature creation.
2. `964535d` — tests, code style checks, Docker, and development setup.
3. `35af48b` — fixes to the feature-building script.

The details and test results are in [docs/PR_TESTING.md](docs/PR_TESTING.md).

To inspect one commit without changing your current folder:

```bash
git worktree add /tmp/trustknee-ref-c6f85cb c6f85cb
git worktree remove /tmp/trustknee-ref-c6f85cb
```

## Limits of the current project

The model study does not prove clinical safety, diagnose a knee condition, or
show that the best model will work for every patient. A result from this
dataset is not automatically ready for home use. Confidence scores,
explanations, user feedback, sensor reduction experiments, and testing on
completely new cohorts remain future work.

The data may have noise, missing values, uneven labels, or too few participants.
Good test results would not prove that the system is safe or useful in a clinic.
This project does not diagnose illness or give medical advice.
