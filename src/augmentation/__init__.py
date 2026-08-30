"""Time-series augmentation for TrustKnee raw-signal windows."""

from src.augmentation.pipeline import augment_minority_classes
from src.augmentation.timegan_benchmark import TimeGANAugmenter
from src.augmentation.warping import jitter, magnitude_scale, permute_segments, time_warp

__all__ = [
    "TimeGANAugmenter",
    "augment_minority_classes",
    "jitter",
    "magnitude_scale",
    "permute_segments",
    "time_warp",
]
