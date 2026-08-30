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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        # The model evaluator owns the complete LOSO split.  The ingestion
        # default excludes the historical single-subject smoke-test holdout,
        # so explicitly include all participants here.
        manifest = build_manifest(args.data_root, exclude_subjects=set())
        run_loso_comparison(
            manifest,
            args.output_dir,
            seed=args.seed,
            quick=args.quick,
            device=args.device,
        )
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        raise SystemExit(
            f"Real evaluation blocked: {exc}. Prepare the local dataset metadata and install "
            "the model dependencies before retrying."
        ) from exc
    return 0


if __name__ == "__main__":
    main()
