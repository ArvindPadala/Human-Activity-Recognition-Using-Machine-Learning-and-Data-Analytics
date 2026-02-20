"""
LSTM model definition, compilation, and export utilities.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from src import config


def build_model() -> Sequential:
    """
    Build an improved LSTM model for HAR.

    Architecture (upgraded from the legacy single-LSTM version):
        LSTM(64, return_sequences=True)
        Dropout(0.3)
        LSTM(32)
        Dropout(0.3)
        Dense(64, relu)
        Dropout(0.2)
        Dense(n_classes, softmax)
    """
    model = Sequential([
        LSTM(
            config.LSTM_UNITS_1,
            return_sequences=True,
            input_shape=(config.N_TIME_STEPS, config.N_FEATURES),
            kernel_regularizer=l2(config.L2_REG),
            bias_regularizer=l2(config.L2_REG),
            name="lstm_1",
        ),
        Dropout(config.DROPOUT_RATE, name="dropout_1"),
        LSTM(
            config.LSTM_UNITS_2,
            return_sequences=False,
            kernel_regularizer=l2(config.L2_REG),
            bias_regularizer=l2(config.L2_REG),
            name="lstm_2",
        ),
        Dropout(config.DROPOUT_RATE, name="dropout_2"),
        Dense(
            config.DENSE_UNITS,
            activation="relu",
            kernel_regularizer=l2(config.L2_REG),
            bias_regularizer=l2(config.L2_REG),
            name="dense_1",
        ),
        Dropout(config.DENSE_DROPOUT_RATE, name="dropout_3"),
        Dense(
            config.N_CLASSES,
            activation="softmax",
            name="output",
        ),
    ], name="HAR_LSTM")

    return model


def compile_model(model: Sequential) -> Sequential:
    """Compile the model with Adam optimiser and sparse cross-entropy."""
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def export_saved_model(model: Sequential, path: str | None = None):
    """Export the trained model in TF SavedModel format."""
    path = path or config.SAVED_MODEL_DIR
    model.save(path)
    print(f"Model exported to {path}")


def load_saved_model(path: str | None = None) -> Sequential:
    """Load a previously exported model."""
    path = path or config.SAVED_MODEL_DIR
    model = tf.keras.models.load_model(path)
    print(f"Model loaded from {path}")
    return model
