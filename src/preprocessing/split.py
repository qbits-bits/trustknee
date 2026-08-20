"""Subject-level dataset splitting utilities."""

import numpy as np


def train_val_test_split(data, on="subject_id"):
    """Split rows into reproducible train, validation, and test partitions.

    Splitting is performed on unique subject IDs so windows from one subject
    cannot appear in more than one partition. The proportions are 70%, 15%,
    and 15% of the subjects, respectively.

    Args:
        data: A pandas DataFrame containing the grouping column.
        on: Name of the column whose unique values define the split groups.

    Returns:
        Three DataFrames in train, validation, and test order.
    """
    df = data
    subjects = df[on].unique()  # shuffle subject IDs;
    np.random.seed(42)  # for reproducibility's sake;
    np.random.shuffle(subjects)

    # Defined ratios of 70(train), 15(val), 15(test);
    train_end = int(0.70 * len(subjects))
    val_end = int(0.85 * len(subjects))

    train_subjects = subjects[:train_end]
    val_subjects = subjects[train_end:val_end]
    test_subjects = subjects[val_end:]

    train = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val = df[df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    test = df[df["subject_id"].isin(test_subjects)].reset_index(drop=True)

    return train, val, test
