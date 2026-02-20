"""
Evaluation script for the trained HAR model.

Usage:
    python -m src.evaluate
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.data import load_raw_data, preprocess, make_timeseries_dataset
from src.model import load_saved_model


def plot_confusion_matrix(y_true, y_pred, labels, save_path=None):
    """Plot a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def plot_training_history(history_path=None, save_path=None):
    """
    Plot training / validation loss and accuracy curves.

    Expects a JSON file at *history_path* with keys
    ``loss``, ``val_loss``, ``accuracy``, ``val_accuracy``.
    If no history file exists, the step is silently skipped.
    """
    history_path = history_path or os.path.join(config.LOG_DIR, "history.json")

    if not os.path.exists(history_path):
        print(f"No training history found at {history_path} — skipping plot.")
        return

    with open(history_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history["loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["accuracy"], label="Train Accuracy")
    ax2.plot(epochs, history["val_accuracy"], label="Val Accuracy")
    ax2.set_title("Accuracy Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    plt.show()


def main():
    """Load the saved model, evaluate on test data, and produce visual reports."""
    # ── Load data ───────────────────────────────────────────
    print("Loading and preprocessing data …")
    df = load_raw_data()
    _, X_test, _, y_test, encoder, _ = preprocess(df)

    test_ds = make_timeseries_dataset(X_test, y_test, batch_size=config.BATCH_SIZE)

    # ── Load model ──────────────────────────────────────────
    model = load_saved_model()

    # ── Evaluate ────────────────────────────────────────────
    loss, accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}\n")

    # ── Predictions ─────────────────────────────────────────
    y_true_all = []
    y_pred_all = []

    for X_batch, y_batch in test_ds:
        preds = model.predict(X_batch, verbose=0)
        y_pred_all.append(np.argmax(preds, axis=1))
        y_true_all.append(y_batch.numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    # ── Classification report ───────────────────────────────
    labels = encoder.classes_
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=labels))

    # ── Confusion matrix ────────────────────────────────────
    os.makedirs(config.LOG_DIR, exist_ok=True)
    plot_confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        save_path=os.path.join(config.LOG_DIR, "confusion_matrix.png"),
    )

    # ── Training history curves ─────────────────────────────
    plot_training_history(
        save_path=os.path.join(config.LOG_DIR, "training_curves.png"),
    )

    print("✅  Evaluation complete.")


if __name__ == "__main__":
    main()
