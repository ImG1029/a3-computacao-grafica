"""Loads and preprocesses a face image database.

Expected layout (AT&T / ORL format):
    database/
        s1/  1.pgm … 10.pgm
        s2/  1.pgm … 10.pgm
        ...

Any folder inside database/ whose name starts with a letter or digit is
treated as one subject. Supported formats: pgm, jpg, jpeg, png, bmp.
"""
import numpy as np
from pathlib import Path

import preprocessor

DATABASE_DIR = Path(__file__).parent / "database"
SUPPORTED = {".pgm", ".jpg", ".jpeg", ".png", ".bmp"}


def load() -> tuple[list[np.ndarray], list[int]]:
    """Return (images, labels) for all subjects found in database/."""
    subjects = sorted(p for p in DATABASE_DIR.iterdir() if p.is_dir())
    if not subjects:
        raise FileNotFoundError(
            f"Nenhum sujeito encontrado em {DATABASE_DIR}. "
            "Baixe a AT&T Face Database e extraia em database/."
        )

    images: list[np.ndarray] = []
    labels: list[int] = []

    for label, subject_dir in enumerate(subjects):
        files = sorted(f for f in subject_dir.iterdir() if f.suffix.lower() in SUPPORTED)
        for file in files:
            try:
                img = preprocessor.preprocess_path(file)
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"  [aviso] ignorando {file.name}: {e}")

    print(f"Base carregada: {len(subjects)} sujeitos, {len(images)} imagens")
    return images, labels


def subject_name(label: int) -> str:
    """Returns the folder name for a given label index."""
    subjects = sorted(p for p in DATABASE_DIR.iterdir() if p.is_dir())
    return subjects[label].name if label < len(subjects) else str(label)
