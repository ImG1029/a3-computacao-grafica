"""Facial recognition: compares retrato falado against the trained database."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import composer
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


# ── REGION-AWARE RECOGNITION ────────────────────────────────────────────────
#
# A partial retrato falado (e.g. only "boca" + "cabelo") leaves most of the
# 500x600 canvas white. A holistic LBP histogram over the whole image is then
# dominated by that flat background and cannot attribute a region to the
# component's owner. Instead we compare EACH selected component against the same
# region of every subject and sum the per-region distances: the owner of each
# selected component scores ~0 in that region and surfaces in the top N.

_REGION_SIZE = (200, 200)


def _component_gray(path: Path) -> np.ndarray:
    """Render a single component over white → 200x200 grayscale (region template)."""
    img = Image.open(path).convert("RGBA")
    if img.size != composer.CANVAS_SIZE:
        img = img.resize(composer.CANVAS_SIZE, Image.LANCZOS)
    canvas = Image.new("RGBA", composer.CANVAS_SIZE, (255, 255, 255, 255))
    canvas = Image.alpha_composite(canvas, img)
    gray = np.array(canvas.convert("L"))
    return cv2.resize(gray, _REGION_SIZE)


def _subject_stems() -> list[str]:
    """All subject identifiers present across the component folders (face_NN)."""
    stems: set[str] = set()
    for category in composer.LAYER_ORDER:
        folder = composer.FACES_DIR / category
        if folder.exists():
            stems.update(p.stem for p in folder.glob("*.png"))
    return sorted(stems)


def recognize_selection(selection: dict[str, Path | None], top_n: int = 5) -> list[Match]:
    """Rank subjects by similarity of their facial regions to the selected components.

    `selection` maps each category to the chosen component Path (or None). Only
    categories with a real, non-constant component contribute to the score.
    """
    chosen = {cat: path for cat, path in selection.items() if path}
    if not chosen:
        raise RuntimeError("Selecione ao menos um componente para reconhecer.")

    query = {cat: features.lbp_histogram(_component_gray(path)) for cat, path in chosen.items()}

    scores: dict[str, float] = {}
    for subject in _subject_stems():
        total, n = 0.0, 0
        for cat in chosen:
            comp_path = composer.FACES_DIR / cat / f"{subject}.png"
            if not comp_path.exists():
                continue
            subj_hist = features.lbp_histogram(_component_gray(comp_path))
            total += _chi_squared(query[cat], subj_hist)
            n += 1
        if n:
            scores[subject] = total / n

    ranked = sorted(scores.items(), key=lambda x: x[1])[:top_n]
    return [
        Match(rank=i + 1, label=-1, subject=subject, distance=round(dist, 2))
        for i, (subject, dist) in enumerate(ranked)
    ]


def train_model() -> None:
    """Load database, preprocess and train the LBPH model."""
    images, labels = database.load()
    features.train(images, labels)


def _chi_squared(h1: np.ndarray, h2: np.ndarray) -> float:
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))
