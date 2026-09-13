import re

import numpy as np
import pandas as pd


def apply_sensor_mask_tabular(
    df: pd.DataFrame,
    active_sensors: list[int] | np.ndarray,
    sensor_column_prefix: str = "s"
) -> pd.DataFrame:
    """
    Filters tabular features based on active sensor indices (supports 1-indexed and 0-indexed formats).
    Retains metadata, trial info, and labels automatically.
    """
    if isinstance(active_sensors, np.ndarray):
        active_sensors = active_sensors.tolist()

    active_set = set(active_sensors)
    allowed_cols = []

    # Metadata columns to retain unconditionally
    meta_cols = {"subject_id", "trial_num", "time", "label_id", "label"}

    for col in df.columns:
        if col in meta_cols:
            allowed_cols.append(col)
            continue

        # Standard single sensor features (s1_emg_mav, sensor_0_mean, etc.);
        if (
            col.startswith(sensor_column_prefix)
            and len(col) > len(sensor_column_prefix)
            and col[len(sensor_column_prefix)].isdigit()
        ):
            # Extract digit right after prefix
            s_id = int(col[len(sensor_column_prefix)])
            if s_id in active_set:
                allowed_cols.append(col)
            continue

        # Pairwise symmetry features (sym_ratio_emg_15, inter_sym_ctrl_15, etc.);
        is_pair_feature = False
        for L, R in [(1, 5), (2, 6), (3, 7), (4, 8), (0, 4), (1, 5), (2, 6), (3, 7)]:
            if f"_{L}{R}" in col:
                is_pair_feature = True
                if L in active_set and R in active_set:
                    allowed_cols.append(col)
                break

        # Whole-array ratios depend on every sensor through their denominator.
        # They are valid only for the unpruned eight-sensor feature set.
        if re.fullmatch(r"lrshare_(?:acc|gyro|emg)", col):
            if active_set in (set(range(1, 9)), set(range(0, 8))):
                allowed_cols.append(col)
            continue

        # Ratio features have sensor IDs without the standard "s<ID>" marker.
        ratio_match = re.fullmatch(r"(?:ctrl_ratio|loadshare_(?:acc|gyro|emg))_(\d+)", col)
        if ratio_match:
            if int(ratio_match.group(1)) in active_set:
                allowed_cols.append(col)
            continue

        # Other numeric feature names may still carry a sensor ID;
        if not is_pair_feature:
            parts = col.split("_")
            has_sensor_match = False
            for part in parts:
                if part.isdigit() and int(part) in active_set:
                    allowed_cols.append(col)
                    has_sensor_match = True
                    break

            # Contextual/global non-sensor features;
            if not has_sensor_match and not any(f"s{i}" in col for i in range(1, 9)):
                allowed_cols.append(col)

    return df[allowed_cols]


def apply_sensor_mask_sequence(
    x_seq: np.ndarray,
    active_sensors: list[int] | np.ndarray
) -> np.ndarray:
    """
    Masks channel dimension for sequence inputs shape (Batch, Channels/Sensors, Length).
    Zeroes out unselected sensor channels for sequence models.
    """
    masked_seq = x_seq.copy()
    active_set = set(active_sensors.tolist() if isinstance(active_sensors, np.ndarray) else active_sensors)
    # SBE emits one-based IDs; retain support for zero-based callers when 0 is present.
    is_one_based = 0 not in active_set
    all_sensors = set(range(1, 9)) if is_one_based else set(range(0, 8))
    inactive_sensors = all_sensors - active_set
    inactive_indices = [
        sensor - 1 if is_one_based else sensor for sensor in inactive_sensors
    ]

    if masked_seq.ndim == 3:
        masked_seq[:, inactive_indices, :] = 0.0
    return masked_seq
