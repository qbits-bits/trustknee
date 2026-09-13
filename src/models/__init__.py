"""Modeling and subject-wise evaluation for TrustKnee.

The package deliberately keeps model inputs separate from trial metadata.  This
makes it harder to accidentally train on subject, trial, or timestamp columns.
"""

from src.models.labels import (
    binary_label_from_execution,
    label_id_to_binary,
    labels_to_binary,
)
from src.models.model_data import (
    ModelDataset,
    build_model_inputs,
    create_emg_activity_envelope,
    make_transformer_sequences,
)
from src.models.prepare_dataset import prepare_dataset

__all__ = [
    "ModelDataset",
    "binary_label_from_execution",
    "build_model_inputs",
    "create_emg_activity_envelope",
    "label_id_to_binary",
    "labels_to_binary",
    "make_transformer_sequences",
    "prepare_dataset",
]


def __getattr__(name: str):
    """Load the optional PyTorch model only when a caller requests it."""
    if name == "TransformerEncoderClassifier":
        from src.models.transformer import TransformerEncoderClassifier

        return TransformerEncoderClassifier
    if name == "run_loso_comparison":
        from src.models.evaluation import run_loso_comparison

        return run_loso_comparison
    if name == "run_fixed_split_comparison":
        from src.models.evaluation import run_fixed_split_comparison

        return run_fixed_split_comparison
    if name in {"TemperatureScaler", "compute_calibration_metrics", "select_uncertainty_threshold"}:
        from src.models.calibration import (
            TemperatureScaler,
            compute_calibration_metrics,
            select_uncertainty_threshold,
        )

        return {
            "TemperatureScaler": TemperatureScaler,
            "compute_calibration_metrics": compute_calibration_metrics,
            "select_uncertainty_threshold": select_uncertainty_threshold,
        }[name]
    if name == "integrated_gradients":
        from src.models.explainability import integrated_gradients

        return integrated_gradients
    if name == "run_phase3_transformer":
        from src.models.phase3 import run_phase3_transformer

        return run_phase3_transformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__.append("TransformerEncoderClassifier")
__all__.append("run_loso_comparison")
__all__.append("run_fixed_split_comparison")
__all__.extend(
    [
        "TemperatureScaler",
        "compute_calibration_metrics",
        "integrated_gradients",
        "run_phase3_transformer",
        "select_uncertainty_threshold",
    ]
)
