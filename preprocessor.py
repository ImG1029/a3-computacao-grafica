import cv2
import numpy as np
from pathlib import Path
from PIL import Image

FACE_SIZE = (200, 200)
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def preprocess(image: Image.Image) -> np.ndarray:
    """Convert PIL image to normalized 200x200 grayscale face crop.

    Falls back to center crop if Haar Cascade finds no face — needed for
    placeholder assets that aren't real photographs.
    """
    gray = _to_gray(image)
    roi = _detect_face(gray) or _center_crop(gray)
    x, y, w, h = roi
    crop = gray[y : y + h, x : x + w]
    return cv2.resize(crop, FACE_SIZE)


def preprocess_path(path: Path) -> np.ndarray:
    return preprocess(Image.open(path).convert("RGB"))


def _to_gray(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _detect_face(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    faces = _CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if not len(faces):
        return None
    return tuple(max(faces, key=lambda f: f[2] * f[3]))


def _center_crop(gray: np.ndarray) -> tuple[int, int, int, int]:
    h, w = gray.shape
    size = int(min(h, w) * 0.6)
    x = (w - size) // 2
    y = (h - size) // 2
    return x, y, size, size
