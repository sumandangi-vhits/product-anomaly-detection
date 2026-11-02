from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from .features import extract_features
from .preprocessing import load_image, SUPPORTED

def image_paths(folder: str):
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Training folder not found: {root}")
    return sorted(p for p in root.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED)

def train_model(training_dir: str, model_path: str, contamination: float = 0.05):
    paths = image_paths(training_dir)
    if len(paths) < 5:
        raise ValueError(
            f"Need at least 5 normal training images; found {len(paths)} in {training_dir}. "
            "Use multiple normal images of the same product under consistent conditions."
        )
    matrix = np.vstack([extract_features(load_image(str(p))) for p in paths])
    model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(matrix)
    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_version": 1}, destination)
    return len(paths), str(destination)

def predict_image(image_path: str, model_path: str):
    artifact = joblib.load(model_path)
    model = artifact["model"]
    features = extract_features(load_image(image_path)).reshape(1, -1)
    prediction = int(model.predict(features)[0])  # 1 normal, -1 anomaly
    raw_score = float(model.score_samples(features)[0])  # lower = more anomalous
    decision_score = float(model.decision_function(features)[0])  # negative = anomaly
    return {
        "prediction": "ANOMALY" if prediction == -1 else "NORMAL",
        "label": prediction,
        "raw_score": raw_score,
        "decision_score": decision_score,
    }
