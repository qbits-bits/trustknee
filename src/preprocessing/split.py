"""Subject-level dataset splitting utilities."""

import numpy as np


def train_val_test_split(data, on="subject_id"):
    """Split rows into reproducible train, validation, and test partitions.

    Splitting is performed on unique subject IDs so windows from one subject
    cannot appear in more than one partition. The target proportions are
    70%, 15%, and 15% of subjects, with safeguards for small datasets.

    Args:
        data: A pandas DataFrame containing the grouping column.
        on: Name of the column whose unique values define the split groups.

    Returns:
        Three DataFrames in train, validation, and test order.
    """
    df = data.copy()
    subjects = df[on].unique()  # shuffle subject IDs;

    n_subjects = len(subjects)
    if n_subjects == 0:
        raise ValueError(f"No unique subjects found in column '{on}'.")

    rng = np.random.default_rng(42)  # for reproducibility's sake;
    rng.shuffle(subjects)

    # Calculate boundaries with minimum size guarantees;
    if n_subjects < 3:
        # Fallback for small datasets;
        train_end = max(1, n_subjects - 2)
        val_end = max(train_end + 1, n_subjects - 1)
    else:
        # Standard 70:15:15 proportional split;
        train_end = max(1, int(0.70 * n_subjects))
        val_end = max(train_end + 1, int(0.85 * n_subjects))

        val_end = min(val_end, n_subjects - 1)

    train_subjects = subjects[:train_end]
    val_subjects = subjects[train_end:val_end]
    test_subjects = subjects[val_end:]

    train = df[df[on].isin(train_subjects)].reset_index(drop=True)
    val = df[df[on].isin(val_subjects)].reset_index(drop=True)
    test = df[df[on].isin(test_subjects)].reset_index(drop=True)

    return train, val, test
