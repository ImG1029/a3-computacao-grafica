"""
Download Labeled Faces in the Wild (LFW) into database/.

LFW contains ~13 000 labeled photos of public figures collected from the web.
With --min-faces 20 you get ~62 people, each with 20+ images — enough to
train a reliable LBPH model.

Requirements:
    pip install scikit-learn

Usage:
    python download_lfw.py                   # default: >=20 photos/person
    python download_lfw.py --min-faces 10    # more people, fewer photos each
    python download_lfw.py --min-faces 5     # 158 people (harder to recognize)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def download(min_faces: int = 20, output_dir: str = "database") -> None:
    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        raise SystemExit(
            "scikit-learn não instalado.\n"
            "Execute: pip install scikit-learn"
        )

    print(f"Baixando LFW (minimo {min_faces} fotos/pessoa)...")
    print("Isso pode demorar alguns minutos na primeira execucao.\n")

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces,
        resize=1.0,   # keep original resolution (~125x94)
        color=True,
    )

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

    images = lfw.images  # (N, H, W, 3)
    # sklearn may return float32 in [0,1] or uint8
    if images.dtype != np.uint8:
        images = (images * 255).clip(0, 255).astype(np.uint8)

    counters: dict[str, int] = {}
    for img_arr, label in zip(images, lfw.target):
        name: str = lfw.target_names[label]
        person_dir = out / name
        person_dir.mkdir(exist_ok=True)
        idx = counters.get(name, 0)
        Image.fromarray(img_arr).save(person_dir / f"{idx:03d}.jpg", quality=95)
        counters[name] = idx + 1

    total = sum(counters.values())
    print(f"Salvo: {total} imagens de {len(counters)} pessoas em '{out.absolute()}'")
    print()
    for name, cnt in sorted(counters.items()):
        print(f"  {name.replace('_', ' '):<30} {cnt} imagens")
    print()
    print("Pronto! Execute 'python app.py' e treine o modelo pela interface.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download LFW dataset into database/")
    ap.add_argument(
        "--min-faces", type=int, default=20,
        help="Minimum number of images per person (default: 20)",
    )
    ap.add_argument("--output-dir", default="database")
    args = ap.parse_args()
    download(args.min_faces, args.output_dir)
