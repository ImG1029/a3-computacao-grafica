#!/usr/bin/env python3
"""
build_database.py — Constrói a base de reconhecimento a partir de source_faces/.

Cada foto source_faces/face_NN.jpg vira um sujeito database/face_NN/, salvo na
MESMA representação alinhada em escala de cinza de onde saem os componentes em
faces/ (extract_strips.py). Assim, o dono de cada componente selecionado existe
na base e tende a ranquear alto no TOP 5 do reconhecimento.

Para cada sujeito são salvas algumas variações (brilho/escala) que dão folga ao
histograma LBP, já que o retrato montado nunca é pixel-idêntico à face original.

Uso:
  python build_database.py                  # processa source_faces/
  python build_database.py --source pasta/  # outra pasta de fotos
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import extract_strips as es

ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "source_faces"
DATABASE_DIR = ROOT / "database"


def _aligned_gray_face(bgr: np.ndarray, lms) -> Image.Image | None:
    """Aligned grayscale face within the oval mask, flattened on white.

    Mirrors how a fully-composed retrato falado looks (aligned gray strips on a
    white canvas), so the trained subject is directly comparable to the query.
    """
    aligned, pts = es._align(bgr, lms)
    if aligned is None:
        return None

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask = es._face_mask(pts, expand=0.08, feather=24)  # (H, W) uint8 alpha

    rgba = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    face = Image.fromarray(rgba)

    canvas = Image.new("RGBA", (es.W, es.H), (255, 255, 255, 255))
    canvas = Image.alpha_composite(canvas, face)
    return canvas.convert("RGB")


def _variants(face: Image.Image) -> list[Image.Image]:
    """A few light augmentations so LBPH tolerates non-identical queries."""
    arr = np.array(face).astype(np.int16)
    out = [face]
    for delta in (-25, 25):  # brightness shift
        out.append(Image.fromarray(np.clip(arr + delta, 0, 255).astype(np.uint8)))
    # slight zoom (crop 6% and resize back)
    w, h = face.size
    m = int(min(w, h) * 0.06)
    out.append(face.crop((m, m, w - m, h - m)).resize((w, h), Image.LANCZOS))
    return out


def build(source: Path | None = None) -> int:
    source = source or SOURCE_DIR
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    if not photos:
        raise FileNotFoundError(f"Nenhuma foto em {source}/")

    import mediapipe as mp

    DATABASE_DIR.mkdir(exist_ok=True)
    detector = es._build_detector()
    ok = 0
    try:
        for photo in photos:
            bgr = cv2.imread(str(photo))
            if bgr is None:
                print(f"  [ERRO] {photo.name}")
                continue
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_img)
            if not result.face_landmarks:
                print(f"  [AVISO] rosto não detectado em {photo.name}")
                continue

            face = _aligned_gray_face(bgr, result.face_landmarks[0])
            if face is None:
                print(f"  [AVISO] alinhamento falhou em {photo.name}")
                continue

            subject = photo.stem  # face_01, face_02, ...
            subject_dir = DATABASE_DIR / subject
            if subject_dir.exists():
                shutil.rmtree(subject_dir)
            subject_dir.mkdir(parents=True)

            for i, var in enumerate(_variants(face)):
                var.save(subject_dir / f"{i:02d}.png", "PNG")

            ok += 1
            print(f"  ✓ {photo.name} → database/{subject}/")
    finally:
        detector.close()

    print(f"\n{ok}/{len(photos)} sujeitos adicionados à base.")
    print("Agora treine o modelo pela interface (ou via app) antes de reconhecer.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Constrói database/ a partir das fotos em source_faces/.")
    ap.add_argument("--source", default=str(SOURCE_DIR),
                    help="Pasta com as fotos-fonte (padrão: source_faces/)")
    args = ap.parse_args()
    build(Path(args.source))


if __name__ == "__main__":
    main()
