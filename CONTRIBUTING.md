# Contributing

## Prerequisites

- Python 3.12+
- (Optional) Docker

## Setup

```bash
bash setup.sh
source .venv/bin/activate
```

This creates a virtualenv, installs dependencies from `requirements.txt`, and registers pre-commit hooks (ruff + commit-msg linting).

Manual alternative without hooks:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Running tests

```bash
pytest -q
```

Tests use synthetic data; the real KneE-PAD dataset is not required.

## Linting

```bash
ruff check .
ruff format --check .
pre-commit run --all-files
```

The pre-commit config includes a branch guard that blocks direct commits to `main`. That check is expected to fail when run on `main`; all other checks should pass.

## Building the feature matrix

After placing the dataset under `data/raw/`:

```bash
python src/features/build_features.py
```

Output: `data/processed/kneepad_features.csv`

## Docker

```bash
docker build --target tester -t trustknee-tester .   # lint + test during build
docker build -t trustknee .                           # minimal runtime image
docker run --rm trustknee
```

Docker is optional. First build requires internet access.

## Commit conventions

Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `test:`, `chore:`, etc.). Don't commit directly to `main` for feature work; use descriptive branches.
