"""Subject-level dataset splitting utilities."""

import numpy as np


def train_val_test_split(data, on="subject_id"):
    """Split rows into reproducible train, validation, and test partitions.

    Splitting is performed on unique subject IDs so windows from one subject
    cannot appear in more than one partition. Target proportions are 70%, 15%,
    and 15% with deterministic fallbacks for small subject counts.

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

    # Handle small N explicitly to guarantee non-empty partitions;
    if n_subjects == 1:
        train_end, val_end = 1, 1  # val and test will be empty by necessity;
    elif n_subjects == 2:
        train_end, val_end = 1, 2  # 1 train, 1 val, 0 test;
    elif n_subjects == 3:
        train_end, val_end = 1, 2  # 1 train, 1 val, 1 test;
    elif n_subjects == 4:
        train_end, val_end = 2, 3  # 2 train, 1 val, 1 test;
    else:
        # Standard 70:15:15 proportional split (n >= 5);
        train_end = int(0.70 * n_subjects)
        val_end = int(0.85 * n_subjects)

        # Guarantee at least 1 subject in val and test;
        train_end = min(train_end, n_subjects - 2)
        val_end = max(train_end + 1, min(val_end, n_subjects - 1))

    train_subjects = subjects[:train_end]
    val_subjects = subjects[train_end:val_end]
    test_subjects = subjects[val_end:]

    train = df[df[on].isin(train_subjects)].reset_index(drop=True)
    val = df[df[on].isin(val_subjects)].reset_index(drop=True)
    test = df[df[on].isin(test_subjects)].reset_index(drop=True)

    return train, val, test
