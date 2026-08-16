# Testing notes for the three PR areas

The remote copy has only `origin/main`. Separate GitHub PR references were not
available. We therefore checked the matching commits from the project's history.
These commits are useful test points, but they are not the original GitHub PRs.

## General test information

Each commit can be placed in its own folder and tested with its own Python
environment. The tests create a small temporary dataset, so the real KneE-PAD
dataset is not needed.

The project uses NumPy, pandas, SciPy, pytest, Ruff, and pre-commit. Installing
packages needs an internet connection. Docker testing also needs Docker and
access to download the base images.

## 1. Data cleaning, windows, and features

- Commit: `c6f85cb`
- It adds signal cleaning, matching IMU and sEMG time sections, feature
  calculations, and more tests.
- It depends on the earlier filtering commit `6bc7cb5`.
- Tests check empty files, wrong shapes, different durations, filtering, 200 ms
  windows with 50% overlap, matching times, and valid feature values.
- This commit is not a complete standalone project from an empty folder because
  it uses earlier project history.

## 2. Tests and development tools

- Commit: `964535d`
- It adds the Python project settings, Ruff, pre-commit, Docker, setup commands,
  and automated GitHub checks.
- The useful checks are `pytest -q`, `ruff check .`, `ruff format --check .`,
  `pre-commit run --all-files`, and the Docker tester build.
- This commit comes before the later feature-builder fixes, so it must be tested
  as its own version.
- The `no-commit-to-branch` hook is expected to fail on `main`; it prevents a
  direct commit to that branch.

## 3. Feature-builder fixes

- Commit: `35af48b`
- Comparison commit: `e8fa718`.
- It fixes style problems in the feature-building script and keeps its output in
  `data/processed/kneepad_features.csv`.
- Run `pytest -q`, `ruff check .`, and `ruff format --check .`.
- Running the feature builder itself needs a real dataset in `data/raw/`.
- This is a small follow-up commit, not a complete project by itself.

## Test one commit in a separate folder

```bash
git worktree add /tmp/trustknee-pr1 c6f85cb
cd /tmp/trustknee-pr1
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PYTHONPATH=. .venv/bin/pytest -q
git worktree remove /tmp/trustknee-pr1
```

Use the same steps with `964535d` and `35af48b`. Add the Ruff, pre-commit, and
Docker commands when those files exist in the selected commit. If installation,
Docker, or a required history entry is unavailable, record it as a limitation
instead of calling the test successful.
