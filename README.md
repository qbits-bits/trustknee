# TrustKnee

TrustKnee is a student project about checking knee-rehabilitation exercises using
wearable sensors. The current code reads sensor data, cleans it, cuts it into
small time sections, and calculates useful numbers from each section.

The project is still under development. It is not a medical device, a diagnosis
tool, or a replacement for a physiotherapist.

## What works now

The current data flow is:

```text
raw KneE-PAD data
    -> read the files and check them
    -> clean the IMU and sEMG signals
    -> make matching 200 ms sections
    -> calculate signal features
    -> save a CSV table
```

The code uses about 148 samples per second for IMU data and about 1259 samples
per second for sEMG data. The two signals can have different numbers of samples,
but they must cover almost the same amount of time.

The proposal also discusses machine-learning models, confidence scores,
explanations, patient feedback, smaller sensor setups, and testing on new
participants. These parts are planned work. They are not finished in this
repository.

## Project folders

| Folder or file | What it does |
| --- | --- |
| `src/config.py` | Stores sensor settings, labels, filter settings, and window settings |
| `src/ingestion/` | Reads dataset information and checks trial files |
| `src/preprocessing/` | Cleans IMU and sEMG signals and makes time sections |
| `src/features/` | Calculates numbers from each time section |
| `tests/` | Runs tests using a small made-up dataset |
| `data/` | Empty folders for raw and processed data |
| `db/schema.sql` | Early database design |
| `docs/` | Project diagrams and testing notes |
| `Dockerfile` | Optional development and test image |
| `pyproject.toml` | Python project and test settings |

## Dataset layout

Put the dataset under `data/raw/` using this structure:

```text
data/raw/
├── participants.csv
├── labels.csv
├── placement.csv
├── sensors.csv
└── dataset/
    └── Subject_<id>/<label_id>/Trial_<number>/
        ├── imu.npy
        └── emg.npy
```

The files use these formats:

- `imu.npy` has shape `(48, T)`: 8 sensors, with 6 values from each sensor.
- `emg.npy` has shape `(8, T)`: one sEMG row for each sensor.
- The six IMU values are acceleration x/y/z and rotation x/y/z.
- The two files may have different sample counts, but their durations must be
  within 50 milliseconds of each other.

There are nine labels: correct and two incorrect versions for Squat, Seated leg
extension, and Walking. The full list is in `src/config.py` and [glossary.md](glossary.md).

The real dataset is not downloaded by this project. Get it separately and do
not commit participant data to Git.

## Install the project

Python 3.10 or newer is required. From the project folder, run:

```bash
bash setup.sh
source .venv/bin/activate
```

This creates a virtual environment, installs the required packages, and sets up
the code checks. You can install the packages without the hooks with:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run tests and checks

```bash
pytest -q
ruff check .
ruff format --check .
pre-commit run --all-files
```

The pre-commit command includes a check that prevents direct commits to `main`.
That one check is expected to fail when run on `main`; the other code checks
should pass.

The tests make a temporary dataset, so the real KneE-PAD dataset is not needed.

After putting real data in `data/raw/`, make the feature CSV with:

```bash
python src/features/build_features.py
```

The output is saved as `data/processed/kneepad_features.csv`. This command only
prepares the data. It does not train a model.

If Docker is installed, the test image can be built with:

```bash
docker build --target tester -t trustknee-tester .
docker build -t trustknee .
docker run --rm trustknee
```

Docker is optional. The first build needs an internet connection to download
packages.

## How the code works

The reader makes a list of all trials and joins each trial with its participant
and label information. It checks that files are not empty, have the expected
shape, and have matching durations.

The IMU signal is smoothed. Optional settings can remove slow signal changes.
The sEMG signal is cleaned by keeping its useful range and removing electrical
noise.

The code then makes 200 millisecond sections with 50% overlap. A new section
starts every 100 milliseconds. Each section has about 30 IMU samples and 252
sEMG samples. The feature code then calculates values such as average size,
peak, power, movement change, range of motion, and signal frequency.

## Future work and ownership

The names below come from the project proposal and commit history. They describe
planned ownership, not completed work.

| Person | Planned area |
| --- | --- |
| Nihal Kumar | Project structure, data reading, signal cleaning, connecting the parts, and future testing on new participants |
| Kaushal Bhatt | Feature creation, basic machine-learning models, and finding a smaller sensor setup |
| Shlok S. Limbhare | Neural-network models, confidence scores, and model explanations |
| Pakki Divya Ankitha | User feedback, data graphs, testing, and documentation |

The planned order is to study the dataset, try basic models, try sequence models,
check whether confidence scores are useful, add explanations and feedback, and
then test on participants who were not used for training.

## Testing the three historical PR areas

The remote copy has only `origin/main`; separate GitHub PR references were not
available. These commits represent the three requested areas:

1. `c6f85cb` — signal cleaning, time sections, and feature creation.
2. `964535d` — tests, code style checks, Docker, and development setup.
3. `35af48b` — fixes to the feature-building script.

The details and test results are in [docs/PR_TESTING.md](docs/PR_TESTING.md).

To inspect one commit without changing your current folder:

```bash
git worktree add /tmp/trustknee-ref-c6f85cb c6f85cb
git worktree remove /tmp/trustknee-ref-c6f85cb
```

## Limits of the current project

The repository does not yet train or run a model. It also does not yet provide
confidence scores, explanations, user feedback, sensor reduction experiments,
or testing on completely new participants.

The data may have noise, missing values, uneven labels, or too few participants.
Good test results would not prove that the system is safe or useful in a clinic.
This project does not diagnose illness or give medical advice.
