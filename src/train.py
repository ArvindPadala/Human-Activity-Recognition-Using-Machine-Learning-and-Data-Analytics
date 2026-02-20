"""
Training entry point for the HAR model.

Usage:
    python -m src.train              # train for EPOCHS defined in config
    python -m src.train --epochs 5   # override epoch count
"""

import argparse
import json
import os

import tensorflow as tf

from src import config
from src.data import download_dataset, load_raw_data, preprocess, make_timeseries_dataset
from src.model import build_model, compile_model, export_saved_model


def get_callbacks():
    """Return a list of Keras callbacks for training."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.MODEL_CHECKPOINT,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=config.LOG_DIR,
            histogram_freq=1,
        ),
    ]


def main(epochs: int | None = None):
    """Run the full training pipeline."""
    epochs = epochs or config.EPOCHS

    # ── 1. Data ─────────────────────────────────────────────
    print("=" * 60)
    print("Step 1 / 4 — Downloading dataset")
    print("=" * 60)
    download_dataset()

    print("\n" + "=" * 60)
    print("Step 2 / 4 — Loading & preprocessing")
    print("=" * 60)
    df = load_raw_data()
    X_train, X_test, y_train, y_test, encoder, scaler = preprocess(df)

    # Create windowed time-series datasets
    train_ds = make_timeseries_dataset(X_train, y_train)
    test_ds = make_timeseries_dataset(X_test, y_test, batch_size=config.BATCH_SIZE)

    # ── 2. Model ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3 / 4 — Building model")
    print("=" * 60)
    model = build_model()
    model = compile_model(model)
    model.summary()

    # ── 3. Train ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Step 4 / 4 — Training for {epochs} epoch(s)")
    print("=" * 60)
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=get_callbacks(),
    )

    # ── 4. Save history ─────────────────────────────────────
    os.makedirs(config.LOG_DIR, exist_ok=True)
    history_path = os.path.join(config.LOG_DIR, "history.json")
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, "w") as f:
        json.dump(hist_dict, f, indent=2)
    print(f"Training history saved to {history_path}")

    # ── 5. Export ───────────────────────────────────────────
    export_saved_model(model)
    print("\n✅  Training complete.")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAR LSTM model")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {config.EPOCHS})",
    )
    args = parser.parse_args()
    main(epochs=args.epochs)
