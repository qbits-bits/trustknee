"""Prepare TrustKnee signal windows and PyTorch data loaders."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.dl_features import fetch_signal_features  # noqa: E402
from src.preprocessing.split import train_val_test_split  # noqa: E402
from src.preprocessing.util import normalize_signals, standardize_length  # noqa: E402
from src.wrapper.dataloader import create_dataloaders  # noqa: E402


def pipeline(
    data_dir: str = "data/raw",
    mode: str = "combined",  # options: "imu", "emg", or "combined";
    batch_size: int = 32,
    target_len: int = 200,
    num_workers: int = 4,
):
    """Build train, validation, and test DataLoaders for signal sequences.

    Args:
        data_dir: Dataset directory. Currently retained for API compatibility;
            the pipeline resolves ``data/raw`` relative to the project root.
        mode: Input representation: ``"imu"``, ``"emg"``, or ``"combined"``.
        batch_size: Number of sequences per DataLoader batch.
        target_len: Number of time steps after truncation or zero-padding.
        num_workers: Number of worker processes used by PyTorch DataLoaders.

    Returns:
        Train, validation, and test DataLoaders.
    """

    data_path = PROJECT_ROOT / "data" / "raw"
    data_df = fetch_signal_features(data_path, mode=mode)  # fetch data;

    match mode:
        case "combined":
            column = "emg_imu_combined"
        case "imu" | "emg":
            column = mode
        case _:
            raise ValueError(f"Invalid mode '{mode}'. Expected 'imu', 'emg', or 'combined'.")

    train_df, val_df, test_df = train_val_test_split(
        data_df, on="subject_id"
    )  # Split on subject_id;

    X_train = standardize_length(
        train_df[column].tolist(), target_len=target_len
    )  # padding with regards to the window;
    X_val = standardize_length(val_df[column].tolist(), target_len=target_len)
    X_test = standardize_length(test_df[column].tolist(), target_len=target_len)

    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values
    y_test = test_df["label_id"].values

    # Normalise the data;
    X_train_norm, X_val_norm, X_test_norm, mean, std = normalize_signals(X_train, X_val, X_test)

    # DataLoaders to prepare, batches;
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_norm,
        y_train,
        X_val_norm,
        y_val,
        X_test_norm,
        y_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Batch training loop;
    for batch in train_loader:
        signals = batch["signal"]  # Shape: (batch_size, channels, time_steps)
        labels = batch["label"]  # Shape: (batch_size,)

        # TODO: Pass your model here for batch training;
        # outputs = model(signals)

        print(f"Batch signals shape: {signals.shape} | Labels shape: {labels.shape}")
        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trustKnee Signal DataLoader")
    parser.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["imu", "emg", "combined"],
        help="Sensor type",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument(
        "--target_len", type=int, default=200, help="Signal window length (time-steps)"
    )
    parser.add_argument("--num_workers", type=int, default=2, help="Multiprocessing workers")

    args = parser.parse_args()

    pipeline(
        mode=args.mode,
        batch_size=args.batch_size,
        target_len=args.target_len,
        num_workers=args.num_workers,
    )
