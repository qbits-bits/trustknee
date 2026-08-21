"""Command-line entry point for building tabular TrustKnee features."""

import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.features.extract import build_feature_matrix  # noqa: E402
from src.ingestion import build_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trustknee.features")


def fetch_stat_features() -> None:
    """Build and save the processed KneE-PAD feature matrix.

    The raw dataset is read from ``data/raw`` relative to the project root and
    the resulting CSV is written to ``data/processed/kneepad_features.csv``.
    """
    # Defined paths dynamically based on project root;
    # Pivoting 3 levels up from `src/features/ml_features.py` reaches project root;
    project_root = Path(__file__).resolve().parents[2]

    raw_data_dir = project_root / "data" / "raw"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ingested raw dataset;
    logger.info("Scanning raw data under: %s", raw_data_dir)
    manifest = build_manifest(raw_data_dir)
    logger.info("Loaded manifest with %d trials.", len(manifest))

    # Process signals & compute features;
    logger.info("Extracting features (Window: %dms)...", config.WINDOW_MS)
    features_df = build_feature_matrix(
        manifest=manifest,
        window_ms=config.WINDOW_MS,
        overlap=config.WINDOW_OVERLAP,
        preprocess=True,
    )

    # 4. Save to `data/processed/`
    output_csv = output_dir / "kneepad_features.csv"
    features_df.to_csv(output_csv, index=False)

    logger.info("Pipeline complete! Saved %s features to %s", features_df.shape, output_csv)


if __name__ == "__main__":
    fetch_stat_features()
