"""LBPH feature extraction, training and model persistence."""
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "lbph_model.yml"
DATA_PATH  = Path(__file__).parent / "model" / "training_data.npz"


def train(images: list[np.ndarray], labels: list[int]) -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)

    label_arr = np.array(labels, dtype=np.int32)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, label_arr)
    recognizer.save(str(MODEL_PATH))

    # Save raw data for top-N histogram comparison in recognizer.py
    np.savez_compressed(DATA_PATH, images=np.array(images), labels=label_arr)
    print(f"Modelo salvo: {MODEL_PATH}  ({len(images)} amostras, {len(set(labels))} sujeitos)")


def load_recognizer() -> cv2.face.LBPHFaceRecognizer:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODEL_PATH}. Execute o treinamento primeiro."
        )
    r = cv2.face.LBPHFaceRecognizer_create()
    r.read(str(MODEL_PATH))
    return r


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dados de treino não encontrados: {DATA_PATH}")
    data = np.load(DATA_PATH)
    return data["images"], data["labels"]


def is_trained() -> bool:
    return MODEL_PATH.exists() and DATA_PATH.exists()


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    """Compute 256-bin LBP histogram via vectorized 8-neighbor encoding."""
    h, w = gray.shape
    center = gray[1 : h - 1, 1 : w - 1].astype(np.int16)
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    for bit, (dr, dc) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        neighbor = gray[1 + dr : h - 1 + dr, 1 + dc : w - 1 + dc].astype(np.int16)
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist.astype(np.float32)
