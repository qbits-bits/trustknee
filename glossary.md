# TrustKnee glossary

Simple explanations of words used in the project.

## Sensors and signals

- **IMU:** A small wearable device that measures movement. It usually contains
  an accelerometer and a gyroscope.
- **sEMG or EMG:** A sensor that measures electrical activity from a muscle
  through the skin.
- **Accelerometer:** Measures straight-line movement and the pull of gravity.
- **Gyroscope:** Measures how fast a sensor is turning.
- **Sensor placement:** Where a sensor is put on the body. The location changes
  what the sensor readings mean.

## Reading and cleaning data

- **Sampling rate:** How many readings are taken each second. TrustKnee uses
  about 148 IMU readings and 1259 sEMG readings each second.
- **Filter:** A method for removing unwanted parts of a signal.
- **Low-pass filter:** Keeps slow movement and removes very fast changes. The
  default IMU setting keeps movement below 6 Hz.
- **High-pass filter:** Removes slow changes in the signal.
- **Band-pass filter:** Keeps only a chosen range of frequencies. The sEMG
  setting keeps 20 to 450 Hz.
- **Notch filter:** Removes one narrow source of noise, such as 50 Hz electrical
  noise.
- **Drift correction:** Removes a slow unwanted change or offset in a signal.
- **Window:** A short piece of a longer signal. TrustKnee uses 200 millisecond
  windows.
- **Overlap:** The part shared by two nearby windows. A 50% overlap means a new
  window starts every 100 milliseconds.
- **Synchronization:** Making sure IMU and sEMG windows describe the same time
  period.

## Numbers made from the signals

- **Feature:** One useful number calculated from a signal, such as its average,
  largest value, or amount of movement.
- **Feature table:** A table where each row is one window and each column is a
  feature or label.
- **RMS:** A way to describe the typical size of a signal.
- **MAV:** The average of the signal's absolute values. It is used for sEMG.
- **Jerk:** How quickly acceleration changes.
- **ROM:** Range of motion. In this project it describes the amount of movement
  estimated from the gyroscope.
- **Waveform length:** The total amount of change between nearby signal values.
- **PSD:** A view of how much signal energy is found at different frequencies.
- **MNF or MPF:** The average frequency of a signal based on its energy.

## Labels and models

- **Correct and incorrect:** The two main exercise result types in the dataset.
- **Nine classes:** The dataset has three exercises. Each has one correct class
  and two incorrect classes: Squat (`0`, `1`, `2`), Seated leg extension (`3`,
  `4`, `5`), and Walking (`6`, `7`, `8`).
- **Machine-learning model:** A program that learns patterns from example data.
- **Basic model:** A model that uses the feature table, such as Random Forest or
  XGBoost. These models are planned, not finished.
- **Sequence model:** A model that reads data in time order, such as an LSTM or
  Transformer. These models are also planned, not finished.
- **Confidence score:** A number showing how sure a model is about its answer.
- **Calibration:** Checking whether confidence scores match real accuracy. For
  example, answers marked 80% confident should be right about 80% of the time.
- **Reliability diagram:** A graph used to check confidence scores.
- **ECE:** One number that summarizes how far confidence scores are from actual
  accuracy.
- **SHAP:** A method for showing which input values helped produce a model's
  answer.
- **Integrated Gradients:** Another method for showing which input values helped
  produce a model's answer.
- **LOSO:** Leave-One-Subject-Out testing. The model is tested on one whole
  participant who was not used for training, and this is repeated for each
  participant.

## Important warnings

- **Data leakage:** Test information accidentally gets into training. This can
  make results look better than they really are. Nearby windows from the same
  trial should not be treated as fully separate people.
- **Testing on new participants:** Keeping a participant completely out of
  training gives a better idea of how the system may work for another person.
- **Clinical limitation:** Good computer results do not prove that a system is
  safe, accurate in a hospital, or able to diagnose or treat a condition.
