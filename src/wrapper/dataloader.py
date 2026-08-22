"""PyTorch DataLoader construction for train, validation, and test splits."""

from torch.utils.data import DataLoader

from src.wrapper.dataset_seq import SequenceDataset, dynamic_padding_collate


def create_dataloaders(
    X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int = 32, num_workers: int = 2
):
    """Create non-overlapping DataLoaders for the three dataset partitions.

    Training batches are shuffled; validation and test batches preserve input
    order. Each input array must be shaped ``(samples, channels, time)``.

    Returns:
        A tuple containing the train, validation, and test DataLoaders.
    """
    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=dynamic_padding_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dynamic_padding_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dynamic_padding_collate,
    )

    return train_loader, val_loader, test_loader
