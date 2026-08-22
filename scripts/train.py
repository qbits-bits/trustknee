"""Prepare TrustKnee signal windows and PyTorch data loaders."""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.features.dl_features import fetch_signal_features  # noqa: E402
from src.preprocessing.split import train_val_test_split  # noqa: E402
from src.preprocessing.util import normalize_signals  # noqa: E402
from src.wrapper.dataloader import create_dataloaders  # noqa: E402


def pipeline(
    data_dir: str = "data/raw",
    mode: str = "combined",  # options: "imu", "emg", or "combined";
    batch_size: int = 32,
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
    train_df, val_df, test_df = train_df.copy(), val_df.copy(), test_df.copy()

    # Normalise the data;
    train_norm, val_norm, test_norm, mean, std = normalize_signals(
        train_df[column].tolist(), val_df[column].tolist(), test_df[column].tolist()
    )

    norm_col = f"normalised_{column}"
    train_df[norm_col] = train_norm
    val_df[norm_col] = val_norm
    test_df[norm_col] = test_norm

    # Extract the X, y for dataloader;
    X_train = train_df[norm_col]
    X_val = val_df[norm_col]
    X_test = test_df[norm_col]

    y_train = train_df["label_id"].values.astype(int)
    y_val = val_df["label_id"].values.astype(int)
    y_test = test_df["label_id"].values.astype(int)

    # DataLoaders to prepare, batches;
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU;
    else:
        device = torch.device("cpu")  # PyTorch's default device;
    print(f"Using device: {device}")

    # model = TCN().to(device) # TODO: move model to the device;

    # Batch training loop;
    for signals, labels, mask in train_loader:
        # signals shape: (batch_size, channels, max_time_steps)
        # labels shape:  (batch_size,)
        # mask shape:    (batch_size, max_time_steps)

        # Send tensors to device (CPU/GPU/MPS)
        # signals = signals.to(device)
        # labels = labels.to(device)
        # mask = mask.to(device)

        # Forward pass through your TCN
        # TODO: Pass your model here for batch training;
        # outputs = model(signals, mask)
        # loss = criterion(outputs, labels)

        print(
            f"Batch signals shape: {signals.shape} | Labels shape: {labels.shape} | Mask shape: {mask.shape}"
        )
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
    parser.add_argument("--num_workers", type=int, default=2, help="Multiprocessing workers")

    args = parser.parse_args()

    pipeline(
        mode=args.mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
