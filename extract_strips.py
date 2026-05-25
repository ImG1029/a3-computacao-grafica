#!/usr/bin/env python3
"""
extract_strips.py — Extrai faixas horizontais de componentes faciais.

Processa fotos em source_faces/ e gera faixas horizontais para cada
componente facial (cabelo, sobrancelhas, olhos, nariz, boca).

Cada faixa é uma imagem 500x600 RGBA onde apenas a faixa horizontal
do componente tem pixels visíveis, permitindo composição por sobreposição.

Uso:
  python extract_strips.py                     # processa source_faces/
  python extract_strips.py --keep-old          # mantém componentes antigos
  python extract_strips.py --source outra_pasta/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

W, H = 500, 600

ROOT       = Path(__file__).parent
FACES_DIR  = ROOT / "faces"
SOURCE_DIR = ROOT / "source_faces"
MODEL_DIR  = ROOT / "model"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

CATEGORIES = ["rosto", "cabelo", "sobrancelhas", "olhos", "nariz", "boca", "barba"]

FACE_OVAL  = [10,338,297,332,284,251,389,356,454,323,361,288,
              397,365,379,378,400,377,152,148,176,149,150,136,
              172,58,132,93,234,127,162,21,54,103,67,109]
LEFT_EYE   = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
RIGHT_EYE  = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398]
LEFT_BROW  = [70,63,105,66,107,55,65,52,53,46]
RIGHT_BROW = [300,293,334,296,336,285,295,282,283,276]
NOSE_IDX   = [1,2,98,327,168,6,197,195,5,4,45,275,220,440,94]
MOUTH_IDX  = [61,146,91,181,84,17,314,405,321,375,291,
              308,324,318,402,317,14,87,178,88,95,78,
              191,80,81,82,13,312,311,310,415,308]

LM_FOREHEAD  = 10
LM_CHIN      = 152
LM_NOSE_ROOT = 6
LM_EYE_L_IN  = 133
LM_EYE_L_OUT = 33
LM_EYE_R_IN  = 362
LM_EYE_R_OUT = 263

EYE_L_TARGET = (188, 285)
EYE_R_TARGET = (312, 285)

FEATHER_PX = 40


def _get_model() -> Path:
    import urllib.request
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print("Baixando modelo MediaPipe (~20 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _lm_to_arr(lms, h: int, w: int) -> np.ndarray:
    return np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms], dtype=np.int32)


def _align(img_bgr: np.ndarray, lms):
    h_src, w_src = img_bgr.shape[:2]
    pts = _lm_to_arr(lms, h_src, w_src)

    le = pts[[LM_EYE_L_IN, LM_EYE_L_OUT]].mean(axis=0)
    re = pts[[LM_EYE_R_IN, LM_EYE_R_OUT]].mean(axis=0)

    src_dist    = float(np.linalg.norm(re - le))
    target_dist = float(np.linalg.norm(
        np.array(EYE_R_TARGET, float) - np.array(EYE_L_TARGET, float)))

    if src_dist < 1:
        return None, None

    scale = target_dist / src_dist
    angle = float(np.degrees(np.arctan2(float(re[1] - le[1]), float(re[0] - le[0]))))
    src_c = ((le + re) / 2.0).tolist()
    tgt_c = ((EYE_L_TARGET[0] + EYE_R_TARGET[0]) / 2.0,
             (EYE_L_TARGET[1] + EYE_R_TARGET[1]) / 2.0)

    M = cv2.getRotationMatrix2D(src_c, angle, scale)
    M[0, 2] += tgt_c[0] - src_c[0]
    M[1, 2] += tgt_c[1] - src_c[1]

    aligned = cv2.warpAffine(img_bgr, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    pts_h       = np.hstack([pts.astype(np.float32),
                             np.ones((len(pts), 1), dtype=np.float32)])
    pts_aligned = (M @ pts_h.T).T.astype(np.int32)
    return aligned, pts_aligned


def _strip_alpha(y_top: int, y_bot: int, feather: int = FEATHER_PX) -> np.ndarray:
    """1-D alpha column (H,) with gradient edges for a horizontal strip."""
    col = np.zeros(H, dtype=np.float32)
    y_top = max(0, y_top)
    y_bot = min(H, y_bot)
    for y in range(y_top, y_bot):
        t = min((y - y_top) / feather, 1.0) if y_top > 0 and feather > 0 else 1.0
        b = min((y_bot - 1 - y) / feather, 1.0) if y_bot < H and feather > 0 else 1.0
        col[y] = min(t, b)
    return (col * 255).astype(np.uint8)


def _face_mask(lms: np.ndarray, expand: float = 0.08, feather: int = 24) -> np.ndarray:
    """Alpha mask shaped by the face oval landmarks."""
    pts  = lms[FACE_OVAL]
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))[:, 0, :]
    if expand > 0:
        cx, cy = hull.mean(axis=0)
        hull = (np.array([cx, cy]) + (hull - np.array([cx, cy])) * (1 + expand)).astype(np.int32)
    mask_img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask_img).polygon([tuple(map(int, p)) for p in hull], fill=255)
    if feather > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(feather))
    return np.array(mask_img)


def _hair_mask(lms: np.ndarray, face_mask: np.ndarray) -> np.ndarray:
    """Extends the face mask upward and sideways to include hair."""
    oval_pts   = lms[FACE_OVAL]
    forehead_y = int(lms[LM_FOREHEAD][1])
    left_x     = int(oval_pts[:, 0].min())
    right_x    = int(oval_pts[:, 0].max())
    pad_x      = int((right_x - left_x) * 0.25)

    hair_region = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(hair_region)
    draw.rectangle([max(0, left_x - pad_x), 0,
                    min(W, right_x + pad_x), forehead_y + 40], fill=255)
    hair_region = hair_region.filter(ImageFilter.GaussianBlur(20))

    return np.maximum(face_mask, np.array(hair_region))


def _extract_strip(bgr: np.ndarray, y_top: int, y_bot: int,
                   shape_mask: np.ndarray,
                   feather: int = FEATHER_PX) -> Image.Image:
    rgba    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    col     = _strip_alpha(y_top, y_bot, feather)
    strip   = np.broadcast_to(col[:, np.newaxis], (H, W)).copy()
    rgba[:, :, 3] = np.minimum(strip, shape_mask)
    return Image.fromarray(rgba)


def _save(img: Image.Image, category: str, tag: str) -> None:
    dest = FACES_DIR / category
    dest.mkdir(parents=True, exist_ok=True)
    img.save(dest / f"face_{tag}.png", "PNG")


def _build_detector():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_get_model())),
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def _process_one(img_path: Path, detector, idx: int) -> bool:
    import mediapipe as mp

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"  [ERRO] {img_path.name}")
        return False

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)
    if not result.face_landmarks:
        print(f"  [AVISO] Rosto não detectado em {img_path.name}")
        return False

    aligned, lms = _align(bgr, result.face_landmarks[0])
    if aligned is None:
        return False

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    aligned = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    brow_pts  = lms[LEFT_BROW + RIGHT_BROW]
    brow_top  = int(brow_pts[:, 1].min())
    brow_bot  = int(brow_pts[:, 1].max())
    eye_pts   = lms[LEFT_EYE + RIGHT_EYE]
    eye_top   = int(eye_pts[:, 1].min())
    eye_bot   = int(eye_pts[:, 1].max())
    nose_root = int(lms[LM_NOSE_ROOT][1])
    nose_bot  = int(lms[NOSE_IDX][:, 1].max())
    mouth_pts = lms[MOUTH_IDX]
    mouth_top = int(mouth_pts[:, 1].min())
    mouth_bot = int(mouth_pts[:, 1].max())
    chin_y    = int(lms[LM_CHIN][1])

    margin  = 35
    tag     = f"{idx:02d}"

    face_m = _face_mask(lms, expand=0.08, feather=24)
    hair_m = _hair_mask(lms, face_m)

    # rosto — face within oval mask (background becomes transparent → white)
    rosto_rgba = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGBA)
    rosto_rgba[:, :, 3] = face_m
    _save(Image.fromarray(rosto_rgba), "rosto", tag)

    # cabelo — uses hair mask (face oval + region above forehead)
    _save(_extract_strip(aligned, 0, brow_top + 20, hair_m, feather=FEATHER_PX),
          "cabelo", tag)

    # sobrancelhas, olhos, nariz, boca — face oval clips the background
    _save(_extract_strip(aligned, brow_top - margin, brow_bot + margin, face_m, FEATHER_PX),
          "sobrancelhas", tag)

    _save(_extract_strip(aligned, eye_top - margin, eye_bot + margin, face_m, FEATHER_PX),
          "olhos", tag)

    _save(_extract_strip(aligned, nose_root - margin, nose_bot + margin, face_m, FEATHER_PX),
          "nariz", tag)

    _save(_extract_strip(aligned, mouth_top - margin, min(chin_y + 35, H), face_m, FEATHER_PX),
          "boca", tag)

    # barba — placeholder transparente
    barba_dir = FACES_DIR / "barba"
    barba_dir.mkdir(parents=True, exist_ok=True)
    sem = barba_dir / "sem_barba.png"
    if not sem.exists():
        Image.new("RGBA", (W, H), 0).save(sem, "PNG")

    print(f"  ✓ {img_path.name} → face_{tag}")
    return True


def generate_strips(source: Path | None = None, clear_old: bool = True) -> int:
    """Process all source faces and generate strip components. Returns count of successes."""
    source = source or SOURCE_DIR
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = sorted(p for p in source.iterdir() if p.suffix.lower() in exts)
    if not photos:
        raise FileNotFoundError(f"Nenhuma foto em {source}/")

    if clear_old:
        for cat in CATEGORIES:
            d = FACES_DIR / cat
            if d.exists():
                for f in d.glob("*.png"):
                    f.unlink()

    print(f"Extraindo faixas horizontais de {len(photos)} foto(s)...\n")
    detector = _build_detector()
    try:
        ok = sum(1 for i, p in enumerate(photos, 1) if _process_one(p, detector, i))
    finally:
        detector.close()

    print(f"\n{ok}/{len(photos)} rostos processados com sucesso.")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai faixas horizontais de componentes faciais de fotos reais.")
    parser.add_argument("--keep-old", action="store_true",
                        help="Mantém componentes antigos em faces/")
    parser.add_argument("--source", type=str, default=str(SOURCE_DIR),
                        help="Diretório com fotos de rostos (padrão: source_faces/)")
    args = parser.parse_args()

    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print("[ERRO] mediapipe não instalado. Execute: pip install mediapipe")
        sys.exit(1)

    generate_strips(Path(args.source), clear_old=not args.keep_old)


if __name__ == "__main__":
    main()
