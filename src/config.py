"""
Central configuration for the HAR project.
All hyperparameters, paths, and constants live here.
"""

import os

# ── Paths ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "DataSet")
DATASET_URL = (
    "https://www.utwente.nl/en/eemcs/ps/dataset-folder/"
    "sensors-activity-recognition-dataset-shoaib.rar"
)
DATASET_RAR = os.path.join(PROJECT_ROOT, "HAR_dataset.rar")
MODEL_CHECKPOINT = os.path.join(PROJECT_ROOT, "best_model.keras")
SAVED_MODEL_DIR = os.path.join(PROJECT_ROOT, "saved_model")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

# ── Dataset ─────────────────────────────────────────────────
# 9 sensor features from one pocket placement (Accelerometer, Linear Acc, Gyroscope)
FEATURE_COLUMNS = ["Ax", "Ay", "Az", "Lx", "Ly", "Lz", "Gx", "Gy", "Gz"]
LABEL_COLUMN = "Activity"
N_FEATURES = len(FEATURE_COLUMNS)

ACTIVITIES = [
    "biking",
    "downstairs",
    "jogging",
    "sitting",
    "standing",
    "upstairs",
    "walking",
]
N_CLASSES = len(ACTIVITIES)

# Known typos in the raw data
LABEL_TYPOS = {"upsatirs": "upstairs"}

# ── Preprocessing ───────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_TIME_STEPS = 100  # sliding window length

# ── Model ───────────────────────────────────────────────────
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
DENSE_DROPOUT_RATE = 0.2
L2_REG = 1e-6
LEARNING_RATE = 1e-3

# ── Training ────────────────────────────────────────────────
BATCH_SIZE = 1024
EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
