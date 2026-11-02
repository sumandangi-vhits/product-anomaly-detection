import cv2
import numpy as np

def extract_features(rgb_image: np.ndarray) -> np.ndarray:
    """Return color, intensity, texture and edge statistics plus histograms."""
    image = np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    features = []

    for channel in cv2.split(image):
        features.extend([float(channel.mean()), float(channel.std())])
        hist = cv2.calcHist([channel], [0], None, [16], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-8)
        features.extend(hist.tolist())

    features.extend([
        float(gray.mean()),
        float(gray.std()),
        float(np.percentile(gray, 10)),
        float(np.percentile(gray, 50)),
        float(np.percentile(gray, 90)),
    ])

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    edges = cv2.Canny(gray, 80, 160)
    features.extend([
        float(magnitude.mean()),
        float(magnitude.std()),
        float((edges > 0).mean()),
    ])

    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    features.extend(hist.tolist())

    return np.asarray(features, dtype=np.float32)
