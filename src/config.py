"""Config for the TrustKnee ingestion and preprocessing pipeline.

Values here come from the team's dataset metadata (sensors.csv, placement.csv,
labels.csv), not guesses. Update this file if the metadata changes.

Dataset: KneE-PAD, 31 participants, 8 sensors (1 EMG + 6 IMU channels each),
3 exercises x correct + 2 wrong variants = 9 labels.
Source: Konstantoudakis et al., Scientific Data, 2025.
"""

from dataclasses import dataclass
from pathlib import Path

# Point this at wherever the raw dataset lives. Override via build_manifest's
# data_root argument, not by editing this default.
DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/Data")

# Sensor hardware constants, confirmed against the KneE-PAD data descriptor.
N_SENSORS = 8
EMG_CHANNELS_PER_SENSOR = 1
IMU_CHANNELS_PER_SENSOR = 6  # accel x,y,z + gyro x,y,z

EMG_SAMPLING_RATE_HZ = 1259.2592592592594
IMU_SAMPLING_RATE_HZ = 148.14814814814815

EMG_UNIT = "millivolts"
ACCEL_UNIT = "g"
GYRO_UNIT = "deg/s"

# Trigno Avanti sensor's own published physiological bandwidths.
EMG_BANDWIDTH_HZ = (20.0, 450.0)
ACCEL_BANDWIDTH_HZ = (24.0, 470.0)
GYRO_BANDWIDTH_HZ = (24.0, 360.0)

IMU_CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

# Raw .npy row layout: IMU is (48, T), sensor-major, row = sensor_index * 6 +
# channel_index. EMG is (8, T), one row per sensor.

# Sensor placement, from placement.csv.
SENSOR_PLACEMENT = {
    1: "right rectus femoris",
    2: "right hamstrings",
    3: "right tibialis anterior",
    4: "right gastrocnemius",
    5: "left rectus femoris",
    6: "left hamstrings",
    7: "left tibialis anterior",
    8: "left gastrocnemius",
}


@dataclass(frozen=True)
class LabelInfo:
    label_id: int
    execution: str  # "Correct" | "Wrong"
    exercise: str
    description: str


# Labels, from labels.csv, confirmed verbatim against the published table.
LABELS = {
    0: LabelInfo(0, "Correct", "Squat", "Squat"),
    1: LabelInfo(1, "Wrong", "Squat", "Squat execution with weight transfer on the healthy leg"),
    2: LabelInfo(2, "Wrong", "Squat", "Squat execution placing the injured leg in front"),
    3: LabelInfo(3, "Correct", "Seated leg extension", "Seated leg extension"),
    4: LabelInfo(
        4,
        "Wrong",
        "Seated leg extension",
        "Seated leg extension of the injured knee without full range of motion",
    ),
    5: LabelInfo(
        5,
        "Wrong",
        "Seated leg extension",
        "Seated leg extension of the injured knee with lifting of the limb from the chair",
    ),
    6: LabelInfo(6, "Correct", "Walking", "Walking"),
    7: LabelInfo(
        7, "Wrong", "Walking", "Walking with the injured limb (knee joint) not full extended"
    ),
    8: LabelInfo(
        8,
        "Wrong",
        "Walking",
        "Walking with the injured limb in full knee extension with hip abduction",
    ),
}
# Label 7 keeps the source table's own typo ("not full extended") instead of
# correcting it, so this matches the dataset's documentation exactly.


@dataclass(frozen=True)
class FilterConfig:
    imu_lowpass_hz: float = 6.0  # see preprocessing/filters.py for why
    imu_lowpass_order: int = 4
    imu_drift_highpass_hz: float = 0.3
    imu_drift_highpass_order: int = 2
    emg_bandpass_hz: tuple = (20.0, 450.0)  # sensor's own physiological bandwidth
    emg_bandpass_order: int = 4
    emg_notch_hz: float = 50.0  # use 60.0 in a 60 Hz mains region
    emg_notch_q: float = 30.0


FILTERS = FilterConfig()

# Windowing defaults for downstream segmentation. Defined in time, not
# samples, so each modality keeps its own native sample rate.
WINDOW_MS = 200.0
WINDOW_OVERLAP = 0.5
