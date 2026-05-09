"""Facial recognition: compares retrato falado against the trained database."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import database
import features
import preprocessor


@dataclass
class Match:
    rank: int
    label: int
    subject: str
    distance: float

    @property
    def similarity(self) -> float:
        """Normalized similarity score [0, 1]. Lower distance = higher similarity."""
        return round(1 / (1 + self.distance / 100), 4)


def recognize(image: Image.Image, top_n: int = 5) -> list[Match]:
    """Return top N matches for the given retrato falado image."""
    if not features.is_trained():
        raise RuntimeError("Modelo não treinado. Chame train_model() primeiro.")

    face = preprocessor.preprocess(image)
    query_hist = features.lbp_histogram(face)

    train_images, train_labels = features.load_training_data()

    # Compute chi-squared distance from query to every training sample
    best: dict[int, float] = {}
    for img, label in zip(train_images, train_labels):
        dist = _chi_squared(query_hist, features.lbp_histogram(img))
        label = int(label)
        if label not in best or dist < best[label]:
            best[label] = dist

    ranked = sorted(best.items(), key=lambda x: x[1])[:top_n]
    return [
        Match(
            rank=i + 1,
            label=label,
            subject=database.subject_name(label),
            distance=round(dist, 2),
        )
        for i, (label, dist) in enumerate(ranked)
    ]


def recognize_path(path: Path, top_n: int = 5) -> list[Match]:
    return recognize(Image.open(path).convert("RGB"), top_n)


def train_model() -> None:
    """Load database, preprocess and train the LBPH model."""
    images, labels = database.load()
    features.train(images, labels)


def _chi_squared(h1: np.ndarray, h2: np.ndarray) -> float:
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))
