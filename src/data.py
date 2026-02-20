"""
Data pipeline: download, load, preprocess, and create TF datasets.
"""

import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

from src import config


def download_dataset():
    """Download the HAR dataset if it doesn't already exist."""
    if os.path.exists(config.DATASET_DIR):
        print("Dataset directory already exists — skipping download.")
        return

    print(f"Downloading dataset from {config.DATASET_URL} …")
    urllib.request.urlretrieve(config.DATASET_URL, config.DATASET_RAR)
    print("Download completed.")

    from pyunpack import Archive
    Archive(config.DATASET_RAR).extractall(config.PROJECT_ROOT)
    print("Extraction completed.")


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate all participant CSVs into one DataFrame."""
    files = sorted(
        f for f in os.listdir(config.DATASET_DIR) if f.endswith(".csv")
    )
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {config.DATASET_DIR}. "
            "Run download_dataset() first."
        )

    frames = [
        pd.read_csv(os.path.join(config.DATASET_DIR, f), header=1)
        for f in files
    ]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} rows from {len(files)} participants.")
    return df


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test : np.ndarray  — shape (samples, n_features)
    y_train, y_test : np.ndarray  — integer-encoded labels
    encoder         : LabelEncoder
    scaler          : StandardScaler
    """
    # ── 1. Extract left-pocket & right-pocket sensor data ───
    left_pocket = df[df.columns[1:10]].copy()
    right_pocket = df[df.columns[15:24]].copy()

    # Standardize column names so both pockets share the same schema
    right_pocket.columns = left_pocket.columns

    # Combine both pockets
    sensor_data = pd.concat([left_pocket, right_pocket], ignore_index=True)

    # ── 2. Activity labels ──────────────────────────────────
    label_col = df.columns[-1]  # "Unnamed: 69"
    labels = pd.concat(
        [df[label_col], df[label_col]], ignore_index=True
    )

    # Fix known typos
    for typo, correction in config.LABEL_TYPOS.items():
        labels = labels.replace(typo, correction)

    sensor_data[config.LABEL_COLUMN] = labels.values

    # Keep only the 9 feature columns + Activity
    sensor_data = sensor_data[config.FEATURE_COLUMNS + [config.LABEL_COLUMN]]
    sensor_data.dropna(inplace=True)

    print(f"Activities: {sensor_data[config.LABEL_COLUMN].unique().tolist()}")
    print(f"Total samples: {len(sensor_data):,}")

    # ── 3. Split features / labels ──────────────────────────
    X = sensor_data[config.FEATURE_COLUMNS].values
    y_raw = sensor_data[config.LABEL_COLUMN].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # ── 4. Train/test split (stratified) ────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # ── 5. Normalise features ───────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}   y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test, encoder, scaler


def make_timeseries_dataset(X, y, batch_size=None):
    """
    Create a windowed timeseries tf.data.Dataset.

    Uses tf.keras.utils.timeseries_dataset_from_array (the modern
    replacement for the deprecated TimeseriesGenerator).
    """
    batch_size = batch_size or config.BATCH_SIZE
    dataset = tf.keras.utils.timeseries_dataset_from_array(
        data=X,
        targets=y[config.N_TIME_STEPS:],   # align targets with windows
        sequence_length=config.N_TIME_STEPS,
        batch_size=batch_size,
        shuffle=True,
    )
    return dataset
