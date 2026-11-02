from pathlib import Path
import cv2
import numpy as np

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def load_image(path: str, size=(224, 224)) -> np.ndarray:
    """Load an image, resize it, and return RGB float pixels in [0, 1]."""
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0
