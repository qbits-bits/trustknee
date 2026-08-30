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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        # Respect the global LOSO held-out subject exclusion by default
        # to prevent inter-subject data leakage during model evaluation.
        exclude_subjects = set() if args.include_held_out else None
        manifest = build_manifest(args.data_root, exclude_subjects=exclude_subjects)
        run_loso_comparison(
            manifest,
            args.output_dir,
            seed=args.seed,
            quick=args.quick,
            device=args.device,
            resume=args.resume,
            batch_size=args.batch_size,
        )
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        raise SystemExit(
            f"Real evaluation blocked: {exc}. Prepare the local dataset metadata and install "
            "the model dependencies before retrying."
        ) from exc
    return 0


if __name__ == "__main__":
    main()
