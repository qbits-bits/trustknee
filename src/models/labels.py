"""Label conversion helpers used by both model tasks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src import config


def binary_label_from_execution(execution: str) -> int:
    """Return ``0`` for a correct execution and ``1`` for a wrong execution."""
    value = str(execution).strip().lower()
    if value == "correct":
        return 0
    if value == "wrong":
        return 1
    raise ValueError(f"Expected execution 'Correct' or 'Wrong', got {execution!r}")


def label_id_to_binary(label_id: int) -> int:
    """Convert one of the nine KneE-PAD label IDs to the binary target."""
    try:
        label = config.LABELS[int(label_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unknown KneE-PAD label ID: {label_id!r}") from exc
    return binary_label_from_execution(label.execution)


def labels_to_binary(
    label_ids: Iterable[int], executions: Iterable[str | None] | None = None
) -> np.ndarray:
    """Convert label IDs, optionally checking/using the supplied execution values.

    Dataset metadata is authoritative when it is available.  The configured
    label table is used for rows where execution is missing, while conflicting
    metadata is rejected rather than silently corrected.
    """
    ids = list(label_ids)
    execution_values = list(executions) if executions is not None else [None] * len(ids)
    if len(execution_values) != len(ids):
        raise ValueError("label_ids and executions must have the same length")

    result = []
    for label_id, execution in zip(ids, execution_values, strict=True):
        from_id = label_id_to_binary(label_id)
        if execution is None or (isinstance(execution, float) and np.isnan(execution)):
            result.append(from_id)
            continue
        from_execution = binary_label_from_execution(execution)
        if from_execution != from_id:
            raise ValueError(
                f"Execution metadata conflicts with label ID {label_id}: {execution!r}"
            )
        result.append(from_execution)
    return np.asarray(result, dtype=np.int64)
