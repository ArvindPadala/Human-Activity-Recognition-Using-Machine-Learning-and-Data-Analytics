# Human Activity Recognition Using Machine Learning

Classify smartphone sensor data into **7 daily activities** using a stacked LSTM neural network built with TensorFlow / Keras.

## Activities

| Label | Activity |
|-------|----------|
| 0 | Biking |
| 1 | Downstairs |
| 2 | Jogging |
| 3 | Sitting |
| 4 | Standing |
| 5 | Upstairs |
| 6 | Walking |

## Dataset

**Sensors Activity Recognition Dataset** by Shoaib et al. (University of Twente).
The dataset contains accelerometer, linear acceleration, and gyroscope readings from smartphones placed in left and right pockets of multiple participants performing activities of daily living.

- **Source:** [UT–Twente PS Lab](https://www.utwente.nl/en/eemcs/ps/dataset-folder/sensors-activity-recognition-dataset-shoaib.rar)
- Automatically downloaded and extracted on first training run.

## Model Architecture

```
Input  (batch, 100, 9)
  │
LSTM-64 (return_sequences=True, L2 reg)
  │
Dropout 0.3
  │
LSTM-32 (L2 reg)
  │
Dropout 0.3
  │
Dense-64 (ReLU, L2 reg)
  │
Dropout 0.2
  │
Dense-7 (Softmax)
```

## Quick Start

```bash
# 1. Create virtual environment (recommended)
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (downloads dataset automatically)
python -m src.train

# 4. Evaluate and visualize results
python -m src.evaluate
```

### CLI Options

```bash
python -m src.train --epochs 10   # override default epoch count
```

## Project Structure

```
.
├── src/
│   ├── __init__.py      # Package marker
│   ├── config.py        # Hyperparameters, paths, constants
│   ├── data.py          # Download, load, preprocess, windowed datasets
│   ├── model.py         # LSTM model definition, compile, export/load
│   ├── train.py         # Training entry point with callbacks
│   └── evaluate.py      # Evaluation, classification report, plots
├── HAR.ipynb            # Original notebook (preserved for reference)
├── requirements.txt     # Pinned dependencies
├── .gitignore
└── README.md
```

## Dependencies

| Package | Version |
|---------|---------|
| TensorFlow | ≥ 2.15, < 2.17 |
| NumPy | ≥ 1.24, < 2.0 |
| Pandas | ≥ 2.0, < 2.3 |
| Matplotlib | ≥ 3.8, < 3.10 |
| scikit-learn | ≥ 1.3, < 1.6 |
| Seaborn | ≥ 0.13 |
| pyunpack + patool | latest |

## Training Outputs

| Artifact | Path |
|----------|------|
| Best checkpoint | `best_model.keras` |
| SavedModel | `saved_model/` |
| TensorBoard logs | `logs/` |
| Training history | `logs/history.json` |
| Confusion matrix | `logs/confusion_matrix.png` |
| Loss / accuracy curves | `logs/training_curves.png` |

## License

This project is for educational and research purposes.