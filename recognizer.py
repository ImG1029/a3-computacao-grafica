from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import composer
import features


@dataclass
class Match:
    rank: int
    label: int
    subject: str
    distance: float

    @property
    def similarity(self) -> float:
        return round(1 / (1 + self.distance / 100), 4)

_REGION_SIZE = (200, 200)


def _component_gray(path: Path) -> np.ndarray:
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


def _chi_squared(h1: np.ndarray, h2: np.ndarray) -> float:
    eps = 1e-10
    return float(np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))
