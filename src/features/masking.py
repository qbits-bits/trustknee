# src/features/masking.py
import numpy as np
import pandas as pd
from typing import List, Union

def apply_sensor_mask_tabular(
    df: pd.DataFrame, 
    active_sensors: Union[List[int], np.ndarray], 
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
        if col.startswith(sensor_column_prefix) and col[len(sensor_column_prefix)].isdigit():
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

        # Ratio features (ctrl_ratio_1, loadshare_acc_3, etc.);
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
    active_sensors: Union[List[int], np.ndarray]
) -> np.ndarray:
    """
    Masks channel dimension for sequence inputs shape (Batch, Channels/Sensors, Length).
    Zeroes out unselected sensor channels for sequence models.
    """
    masked_seq = x_seq.copy()
    all_sensors = set(range(1, 9 if max(active_sensors) > 7 else 8))
    inactive_sensors = list(all_sensors - set(active_sensors))
    
    # Python zero-indexing offset if sensor IDs are 1-based;
    inactive_indices = [s - 1 if max(active_sensors) > 7 else s for s in inactive_sensors]

    if masked_seq.ndim == 3:
        masked_seq[:, inactive_indices, :] = 0.0
    return masked_seq