from pathlib import Path
from .model import train_model, predict_image

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "dataset" / "train" / "good"
MODEL_PATH = ROOT / "models" / "isolation_forest.joblib"

def main():
    if not MODEL_PATH.exists():
        print("Training model...")
        count, path = train_model(str(TRAIN_DIR), str(MODEL_PATH))
        print(f"Model trained using {count} normal images.")

    image_path = input("\nEnter image path: ").strip().strip('"')

    if not Path(image_path).is_file():
        print("Image not found!")
        return

    try:
        result = predict_image(image_path, str(MODEL_PATH))

        print("\n--- ANOMALY DETECTION RESULT ---")
        print(f"Image      : {Path(image_path).name}")
        print(f"Prediction : {result['prediction']}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
