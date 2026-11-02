# CLI Anomaly Detection System (CPU)

A beginner-friendly, command-line anomaly detection starter project using
OpenCV, handcrafted image features, and Scikit-learn Isolation Forest.

## What it does

- Loads and resizes product images with OpenCV.
- Extracts color, intensity, texture, gradient, and edge statistics.
- Learns the distribution of normal training images using Isolation Forest.
- Predicts whether a new image is normal or anomalous.
- Prints anomaly-related scores in the terminal.

## Important limitation

This starter project performs **image-level anomaly detection using global
handcrafted features**. It does not train a CNN, use PatchCore, or create a
pixel-level anomaly heatmap. It is a practical implementation of the
preprocessing, feature extraction, and Isolation Forest concepts covered so
far. PatchCore can be a later extension.

## Requirements

- Python 3.10 or 3.11 recommended
- CPU is sufficient
- Windows, macOS, or Linux

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Add training images

Put at least **5 normal images of the same product type** in:

`dataset/train/good/`

Use consistent lighting, camera position, background, and product type.
More normal images are strongly recommended (for example, 50–200).

Put test images anywhere, or in `dataset/test/`.

## Run

From the project root:

```powershell
python -m src.main
```

Choose option `1` to train. Then choose option `2` and enter a test image path,
for example:

```text
dataset/test/sample.jpg
```

You can also use an absolute path:

```text
D:\images\product_test.jpg
```

## Understanding the scores

- `predict()` returns `1` for normal and `-1` for anomaly.
- `score_samples()` returns a raw score; lower values indicate more anomalous
  observations.
- `decision_function()` is negative for predictions classified as anomalies.

Scores are model-dependent and should not be interpreted as calibrated
probabilities or percentages.

## Project structure

```text
anomaly_detection_cli/
├── dataset/
│   ├── train/good/       # Add normal training images here
│   └── test/             # Optional test images
├── models/               # Trained model is saved here
├── outputs/
├── src/
│   ├── main.py           # CLI menu
│   ├── model.py          # Training and prediction
│   ├── preprocessing.py  # Image loading and resizing
│   └── features.py       # Feature extraction
├── requirements.txt
└── README.md
```

## Suggested next improvements

1. Evaluate on a labeled normal/defective test set.
2. Tune contamination using validation data.
3. Add precision, recall, F1-score, and confusion matrix.
4. Explore image-specific methods such as PatchCore/Anomalib for localization.
