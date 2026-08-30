# TrustKnee

**An Interpretable and Confidence-Aware Minimal-Sensor System for Home-Based Knee Rehabilitation Assessment**

[![Dataset: KneE-PAD](https://img.shields.io/badge/Dataset-KneE--PAD-blue)](https://doi.org/10.1038/s41597-025-04963-4)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

TrustKnee is a multi-modal sEMG + IMU signal processing and machine learning framework designed for accurate, explainable, and safe postural assessment during home-based knee rehabilitation exercises. Built on the open-source **KneE-PAD** dataset (*Kasnesis et al., Scientific Data, 2025*), the pipeline combines minimal sensor selection with calibrated uncertainty estimation and interpretable feature attributions.

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

The pipeline operates on the **KneE-PAD** dataset (31 subjects, 2,086 trials, 8 Delsys Trigno Avanti wireless sensors):

| Parameter | Specification |
|---|---|
| **Sensors** | 8 bilateral lower limb sensors (Rectus Femoris, Hamstrings, Tibialis Anterior, Gastrocnemius) |
| **Modalities** | 8 sEMG channels (~1259.26 Hz) + 48 IMU channels (~148.15 Hz: 3-axis accel + 3-axis gyro) |
| **Exercises** | 3 exercises × 3 execution variants (1 correct + 2 compensatory errors) = 9 labels |
| **Classes** | Squats (0–2), Seated Leg Extensions (3–5), Gait / Walking (6–8) |

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
