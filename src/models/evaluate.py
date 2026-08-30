"""Command-line entry point for TrustKnee LOSO model comparison."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ingestion import build_manifest
from src.models.evaluation import run_loso_comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run TrustKnee Transformer/XGBoost LOSO evaluation"
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Prepared data root")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for result files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Use a tiny CPU configuration")
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Transformer training device; CPU remains the reproducible default",
    )
    parser.add_argument(
        "--include-held-out",
        action="store_true",
        default=False,
        help="Include held-out subjects (Subject 1) in LOSO evaluation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume after folds already checkpointed in the output directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size 64; larger CUDA batches are faster but change optimization",
    )
    parser.add_argument(
        "--augment-minority",
        action="store_true",
        default=False,
        help="Apply time-series data augmentation to minority classes (6, 7, 8) in training folds",
    )
    parser.add_argument(
        "--aug-multiplier",
        type=int,
        default=3,
        help="Multiplier for synthetic minority copies per eligible training trial",
    )
    parser.add_argument(
        "--aug-methods",
        type=str,
        default="jitter,magnitude_scale,time_warp",
        help="Comma-separated list of augmentation methods (jitter, magnitude_scale, time_warp, permute_segments)",
    )
    parser.add_argument(
        "--aug-targets",
        type=str,
        default="6,7,8",
        help="Comma-separated list of integer target label IDs to augment",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        # Respect the global LOSO held-out subject exclusion by default
        # to prevent inter-subject data leakage during model evaluation.
        exclude_subjects = set() if args.include_held_out else None
        manifest = build_manifest(args.data_root, exclude_subjects=exclude_subjects)
        aug_methods = tuple(m.strip() for m in args.aug_methods.split(",") if m.strip())
        aug_targets = tuple(int(lbl.strip()) for lbl in args.aug_targets.split(",") if lbl.strip())
        run_loso_comparison(
            manifest,
            args.output_dir,
            seed=args.seed,
            quick=args.quick,
            device=args.device,
            resume=args.resume,
            batch_size=args.batch_size,
            augment_minority=args.augment_minority,
            aug_multiplier=args.aug_multiplier,
            aug_methods=aug_methods,
            aug_target_labels=aug_targets,
        )
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        raise SystemExit(
            f"Real evaluation blocked: {exc}. Prepare the local dataset metadata and install "
            "the model dependencies before retrying."
        ) from exc
    return 0


if __name__ == "__main__":
    main()
