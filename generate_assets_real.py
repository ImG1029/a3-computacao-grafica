#!/usr/bin/env python3
"""
generate_assets_real.py
Extrai componentes faciais de fotos reais usando MediaPipe Face Landmarker.

  python generate_assets_real.py --download      # baixa 8 rostos + processa
  python generate_assets_real.py --download --n 15
  python generate_assets_real.py                 # processa source_faces/ existente

Requer:  pip install mediapipe opencv-contrib-python Pillow numpy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Suprime logs verbose do MediaPipe
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ── Canvas e diretórios ────────────────────────────────────────────────────────
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

# Posição-alvo dos olhos (mesmas coordenadas do gerador desenhado)
EYE_L_TARGET = (188, 285)
EYE_R_TARGET = (312, 285)

# ── Landmarks do MediaPipe Face Mesh (indices 0-467) ──────────────────────────

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
LM_NOSE_TIP  = 4
LM_NOSE_ROOT = 6
LM_MOUTH_L   = 61
LM_MOUTH_R   = 291
LM_MOUTH_TOP = 13
LM_MOUTH_BOT = 14
LM_EYE_L_IN  = 133
LM_EYE_L_OUT = 33
LM_EYE_R_IN  = 362
LM_EYE_R_OUT = 263

FEATHER = 20


# ── Modelo MediaPipe ───────────────────────────────────────────────────────────

def _get_model() -> Path:
    """Baixa o modelo do FaceLandmarker para model/ se ainda não existir."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Baixando modelo MediaPipe (~20 MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"  Salvo em {MODEL_PATH.relative_to(ROOT)}")
    return MODEL_PATH


# ── Utilitários de landmarks ───────────────────────────────────────────────────

def _lm_to_arr(lms, h: int, w: int) -> np.ndarray:
    """Converte lista de NormalizedLandmark para array (N, 2) em pixels."""
    return np.array([[int(lm.x * w), int(lm.y * h)] for lm in lms], dtype=np.int32)


def _eye_centers(pts: np.ndarray):
    le = pts[[LM_EYE_L_IN, LM_EYE_L_OUT]].mean(axis=0)
    re = pts[[LM_EYE_R_IN, LM_EYE_R_OUT]].mean(axis=0)
    return le, re


# ── Alinhamento facial ─────────────────────────────────────────────────────────

def _align(img_bgr: np.ndarray, lms) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Alinha o rosto no canvas 500×600:
      - olhos nas posições EYE_L_TARGET / EYE_R_TARGET
      - escala e rotação baseadas na distância inter-ocular
    Retorna (bgr alinhado, array (N,2) de landmarks em pixels).
    """
    h_src, w_src = img_bgr.shape[:2]
    pts = _lm_to_arr(lms, h_src, w_src)
    le, re = _eye_centers(pts)

    src_dist    = float(np.linalg.norm(re - le))
    target_dist = float(np.linalg.norm(
        np.array(EYE_R_TARGET, float) - np.array(EYE_L_TARGET, float)))

    if src_dist < 1:
        return None, None

    scale  = target_dist / src_dist
    angle  = float(np.degrees(np.arctan2(float(re[1] - le[1]), float(re[0] - le[0]))))
    src_c  = ((le + re) / 2.0).tolist()
    tgt_c  = ((EYE_L_TARGET[0] + EYE_R_TARGET[0]) / 2.0,
               (EYE_L_TARGET[1] + EYE_R_TARGET[1]) / 2.0)

    M = cv2.getRotationMatrix2D(src_c, angle, scale)
    M[0, 2] += tgt_c[0] - src_c[0]
    M[1, 2] += tgt_c[1] - src_c[1]

    aligned = cv2.warpAffine(img_bgr, M, (W, H),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    pts_h       = np.hstack([pts.astype(np.float32),
                             np.ones((len(pts), 1), dtype=np.float32)])
    pts_aligned = (M @ pts_h.T).T.astype(np.int32)
    return aligned, pts_aligned


# ── Máscara e extração ─────────────────────────────────────────────────────────

def _poly_mask(poly_pts: np.ndarray, feather: int = FEATHER) -> Image.Image:
    mask = Image.new("L", (W, H), 0)
    pts_list = [tuple(map(int, p)) for p in poly_pts]
    if len(pts_list) >= 3:
        ImageDraw.Draw(mask).polygon(pts_list, fill=255)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    return mask


def _hull_pts(pts: np.ndarray, pad: float = 0.0) -> np.ndarray:
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))[:, 0, :]
    if pad > 0:
        cx, cy = hull.mean(axis=0)
        hull = (np.array([cx, cy]) + (hull - np.array([cx, cy])) * (1 + pad)).astype(np.int32)
    return hull


def _canvas() -> Image.Image:
    return Image.new("RGBA", (W, H), 0)


def _to_rgba(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")


def _extract(bgr: np.ndarray, lm_pts: np.ndarray,
             pad: float = 0.35, feather: int = FEATHER) -> Image.Image:
    hull = _hull_pts(lm_pts, pad)
    mask = _poly_mask(hull, feather)
    out  = _canvas()
    out.paste(_to_rgba(bgr), mask=mask)
    return out


# ── Extratores por componente ──────────────────────────────────────────────────

def _extract_rosto(bgr: np.ndarray, lms: np.ndarray) -> Image.Image:
    return _extract(bgr, lms[FACE_OVAL], pad=0.02, feather=22)


def _extract_cabelo(bgr: np.ndarray, lms: np.ndarray) -> Image.Image:
    """Cabelo: acima da testa + laterais do rosto (para cabelo longo)."""
    oval_pts   = lms[FACE_OVAL]
    forehead_y = int(lms[LM_FOREHEAD][1])
    chin_y     = int(lms[LM_CHIN][1])
    left_x     = int(oval_pts[:, 0].min())
    right_x    = int(oval_pts[:, 0].max())
    pad_x      = int((right_x - left_x) * 0.18)

    top_poly = np.array([
        [max(0, left_x - pad_x), 0],
        [min(W, right_x + pad_x), 0],
        [min(W, right_x + pad_x), forehead_y + 30],
        [max(0, left_x - pad_x), forehead_y + 30],
    ], dtype=np.int32)

    left_poly = np.array([
        [0, forehead_y],
        [left_x + 5, forehead_y],
        [left_x + 5, chin_y],
        [0, chin_y],
    ], dtype=np.int32)

    right_poly = np.array([
        [right_x - 5, forehead_y],
        [W, forehead_y],
        [W, chin_y],
        [right_x - 5, chin_y],
    ], dtype=np.int32)

    combined = Image.fromarray(
        np.maximum(
            np.array(_poly_mask(top_poly, feather=25)),
            np.maximum(
                np.array(_poly_mask(left_poly, feather=18)),
                np.array(_poly_mask(right_poly, feather=18)),
            ),
        ).astype(np.uint8),
        "L",
    )
    out = _canvas()
    out.paste(_to_rgba(bgr), mask=combined)
    return out


def _extract_barba(bgr: np.ndarray, lms: np.ndarray) -> Image.Image:
    """Região da barba: abaixo da boca até o queixo."""
    mouth_pts   = lms[MOUTH_IDX]
    mouth_bot_y = int(mouth_pts[:, 1].max())
    chin_y      = int(lms[LM_CHIN][1])
    oval_pts    = lms[FACE_OVAL]

    lower_oval = oval_pts[oval_pts[:, 1] >= mouth_bot_y - 20]
    if len(lower_oval) < 3:
        cx = int(oval_pts[:, 0].mean())
        fw = int(oval_pts[:, 0].max() - oval_pts[:, 0].min())
        lower_oval = np.array([
            [cx - fw // 3, mouth_bot_y],
            [cx + fw // 3, mouth_bot_y],
            [cx + fw // 4, chin_y + 20],
            [cx - fw // 4, chin_y + 20],
        ], dtype=np.int32)

    mask = _poly_mask(_hull_pts(lower_oval, pad=0.08), feather=20)
    out  = _canvas()
    out.paste(_to_rgba(bgr), mask=mask)
    return out


# ── Classificadores automáticos ────────────────────────────────────────────────

def _dominant_hsv(bgr: np.ndarray, pts: np.ndarray) -> tuple[float, float, float]:
    mask_cv = np.zeros(bgr.shape[:2], dtype=np.uint8)
    hull    = cv2.convexHull(pts.reshape(-1, 1, 2))
    cv2.fillConvexPoly(mask_cv, hull, 255)
    pixels  = bgr[mask_cv > 0]
    if len(pixels) < 5:
        return 0.0, 0.0, 100.0
    mean_bgr = pixels.mean(axis=0).astype(np.uint8).reshape(1, 1, 3)
    hsv = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return float(hsv[0]), float(hsv[1]), float(hsv[2])


def _color_label(h: float, s: float, v: float) -> str:
    if v < 55:
        return "escuro"
    if v > 165 and s < 50:
        return "grisalho"
    if 10 <= h <= 32 and s > 50:
        return "castanho"
    return "escuro"


def _face_shape(lms: np.ndarray) -> str:
    oval = lms[FACE_OVAL]
    fw   = int(oval[:, 0].max() - oval[:, 0].min())
    fh   = int(lms[LM_CHIN][1] - lms[LM_FOREHEAD][1])
    r    = fw / max(fh, 1)
    if r > 0.84:
        return "oval_redondo"
    if r < 0.68:
        return "alongado"
    chin_pts = oval[oval[:, 1] > lms[LM_CHIN][1] - 65]
    if len(chin_pts) >= 2:
        jaw_w = chin_pts[:, 0].max() - chin_pts[:, 0].min()
        if jaw_w / fw > 0.86:
            return "quadrado"
    return "oval_padrao"


def _hair_label(bgr: np.ndarray, lms: np.ndarray) -> str:
    forehead_y = int(lms[LM_FOREHEAD][1])
    oval       = lms[FACE_OVAL]
    cx         = int(oval[:, 0].mean())

    y1 = max(0, forehead_y - 70)
    y2 = max(1, forehead_y - 5)
    x1 = max(0, cx - 35)
    x2 = min(W, cx + 35)
    patch = bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return "curto_escuro"
    mean_bgr = patch.mean(axis=(0, 1)).astype(np.uint8).reshape(1, 1, 3)
    hsv   = cv2.cvtColor(mean_bgr, cv2.COLOR_BGR2HSV)[0, 0]
    color = _color_label(float(hsv[0]), float(hsv[1]), float(hsv[2]))

    chin_y  = int(lms[LM_CHIN][1])
    left_x  = int(oval[:, 0].min())
    right_x = int(oval[:, 0].max())

    def _side_dark(x_start: int, x_end: int) -> bool:
        strip = bgr[forehead_y:chin_y, max(0, x_start):min(W, x_end)]
        return strip.size > 0 and float((strip.mean(axis=2) < 90).mean()) > 0.15

    length = "longo" if (_side_dark(left_x - 35, left_x) or
                          _side_dark(right_x, right_x + 35)) else "curto"
    return f"{length}_{color}"


def _nose_label(lms: np.ndarray) -> str:
    nose_pts = lms[NOSE_IDX]
    oval_pts = lms[FACE_OVAL]
    nose_w   = int(nose_pts[:, 0].max() - nose_pts[:, 0].min())
    face_w   = int(oval_pts[:, 0].max() - oval_pts[:, 0].min())
    face_h   = int(lms[LM_CHIN][1] - lms[LM_FOREHEAD][1])
    nose_h   = int(lms[LM_NOSE_TIP][1] - lms[LM_NOSE_ROOT][1])
    if nose_w / max(face_w, 1) > 0.35:
        return "largo"
    if nose_h / max(face_h, 1) < 0.27:
        return "arrebitado"
    return "pequeno"


def _mouth_label(lms: np.ndarray) -> str:
    top      = lms[LM_MOUTH_TOP]
    bot      = lms[LM_MOUTH_BOT]
    lc       = lms[LM_MOUTH_L]
    rc       = lms[LM_MOUTH_R]
    lip_h    = int(bot[1] - top[1])
    corner_y = (float(lc[1]) + float(rc[1])) / 2.0
    smile_d  = corner_y - float(top[1])
    if smile_d < 0:
        return "sorrindo"
    if lip_h > 17:
        return "grossa"
    if lip_h < 8:
        return "fina"
    return "neutra"


def _eye_label(bgr: np.ndarray, lms: np.ndarray) -> str:
    pts = lms[LEFT_EYE]
    cx, cy = pts.mean(axis=0).astype(int)
    patch = bgr[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5]
    if patch.size == 0:
        return "marrom"
    hsv    = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_mean = float(hsv[:, :, 0].mean())
    s_mean = float(hsv[:, :, 1].mean())
    if 90 <= h_mean <= 140 and s_mean > 50:
        return "azul"
    if 35 <= h_mean <= 90 and s_mean > 50:
        return "verde"
    return "marrom"


def _brow_label(lms: np.ndarray) -> str:
    lb = lms[LEFT_BROW]
    return "grossas" if int(lb[:, 1].max() - lb[:, 1].min()) >= 11 else "finas"


def _beard_label(bgr: np.ndarray, lms: np.ndarray) -> str:
    mouth_pts   = lms[MOUTH_IDX]
    mouth_bot_y = int(mouth_pts[:, 1].max())
    chin_y      = int(lms[LM_CHIN][1])
    oval_pts    = lms[FACE_OVAL]
    cx          = int(oval_pts[:, 0].mean())
    fw          = int(oval_pts[:, 0].max() - oval_pts[:, 0].min())

    y1 = min(mouth_bot_y, H - 1)
    y2 = min(chin_y + 20, H - 1)
    x1 = max(cx - fw // 3, 0)
    x2 = min(cx + fw // 3, W - 1)

    if y2 <= y1 or x2 <= x1:
        return "sem_barba"
    region = bgr[y1:y2, x1:x2]
    if region.size == 0:
        return "sem_barba"

    dark_ratio = float((region.mean(axis=2) < 85).mean())
    if dark_ratio > 0.22:
        return "longa_escura" if (y2 - y1) > 55 else "curta_escura"
    return "sem_barba"


# ── Download de amostras ───────────────────────────────────────────────────────

def download_sample_faces(n: int = 8) -> list[Path]:
    """Baixa n rostos de IA sem direitos autorais para source_faces/."""
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    print(f"Baixando {n} rostos de IA (thispersondoesnotexist.com)...")
    for i in range(n):
        out = SOURCE_DIR / f"face_{i + 1:02d}.jpg"
        if out.exists():
            downloaded.append(out)
            print(f"  {out.name} já existe, pulando.")
            continue
        try:
            req = urllib.request.Request(
                "https://thispersondoesnotexist.com/",
                headers={"User-Agent": "Mozilla/5.0 (compatible; face-compositor/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
            out.write_bytes(data)
            downloaded.append(out)
            print(f"  Baixado: {out.name}")
            time.sleep(1.5)
        except Exception as exc:
            print(f"  [ERRO] Rosto {i + 1}: {exc}")
    return downloaded


# ── Salvar componente ──────────────────────────────────────────────────────────

def _save_comp(img: Image.Image, category: str, name: str, counters: dict) -> None:
    n = counters[category] + 1
    counters[category] = n
    dest = FACES_DIR / category
    dest.mkdir(parents=True, exist_ok=True)
    img.save(dest / f"{name}_{n}.png", "PNG")


# ── Processar uma foto ─────────────────────────────────────────────────────────

def _build_detector():
    """Cria o FaceLandmarker (Task API do mediapipe 0.10+)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = _get_model()
    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def process_face(img_path: Path, detector, counters: dict) -> bool:
    """Detecta e extrai todos os componentes de uma foto."""
    import mediapipe as mp

    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"  [ERRO] Não foi possível ler {img_path.name}")
        return False

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)

    if not result.face_landmarks:
        print(f"  [AVISO] Rosto não detectado em {img_path.name}")
        return False

    aligned_bgr, lms = _align(bgr, result.face_landmarks[0])
    if aligned_bgr is None:
        print(f"  [AVISO] Alinhamento falhou em {img_path.name}")
        return False

    brow_pts  = lms[LEFT_BROW + RIGHT_BROW]
    brow_h, brow_s, brow_v = _dominant_hsv(aligned_bgr, brow_pts)

    shape = _face_shape(lms)
    hair  = _hair_label(aligned_bgr, lms)
    brow  = f"{_brow_label(lms)}_{_color_label(brow_h, brow_s, brow_v)}"
    eye   = f"abertos_{_eye_label(aligned_bgr, lms)}"
    nose  = _nose_label(lms)
    mouth = _mouth_label(lms)
    beard = _beard_label(aligned_bgr, lms)

    _save_comp(_extract_rosto(aligned_bgr, lms),                          "rosto",        shape,  counters)
    _save_comp(_extract_cabelo(aligned_bgr, lms),                         "cabelo",       hair,   counters)
    _save_comp(_extract(aligned_bgr, brow_pts, pad=0.5, feather=14),      "sobrancelhas", brow,   counters)
    _save_comp(_extract(aligned_bgr, lms[LEFT_EYE + RIGHT_EYE],
                        pad=0.55, feather=18),                             "olhos",        eye,    counters)
    _save_comp(_extract(aligned_bgr, lms[NOSE_IDX], pad=0.42, feather=18),"nariz",        nose,   counters)
    _save_comp(_extract(aligned_bgr, lms[MOUTH_IDX], pad=0.38, feather=18),"boca",        mouth,  counters)
    _save_comp(_extract_barba(aligned_bgr, lms),                          "barba",        beard,  counters)

    print(f"  ✓ {img_path.name} → {shape} | {hair} | {eye} | {mouth} | {beard}")
    return True


# ── Geração completa ───────────────────────────────────────────────────────────

def generate_real(clear_old: bool = True) -> None:
    """
    Processa todas as fotos em source_faces/ e salva componentes em faces/.
    Chamado por generate_assets.main() quando fotos reais estão disponíveis.
    """
    exts   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    photos = sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in exts)
    if not photos:
        raise FileNotFoundError(f"Nenhuma foto encontrada em {SOURCE_DIR}/")

    if clear_old:
        for cat in CATEGORIES:
            for old in (FACES_DIR / cat).glob("*.png"):
                old.unlink()

    print(f"Processando {len(photos)} foto(s)...\n")
    detector = _build_detector()
    counters = {cat: 0 for cat in CATEGORIES}

    try:
        ok = sum(1 for p in photos if process_face(p, detector, counters))
    finally:
        detector.close()

    print(f"\n{ok}/{len(photos)} rostos processados com sucesso.")
    for cat in CATEGORIES:
        n = counters[cat]
        if n:
            print(f"  faces/{cat}/: {n} componente(s)")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true",
                        help="Baixa amostras de rostos gerados por IA automaticamente")
    parser.add_argument("--n", type=int, default=8,
                        help="Quantidade de rostos a baixar (padrão: 8)")
    parser.add_argument("--keep-old", action="store_true",
                        help="Mantém PNGs antigos em faces/ (não limpa antes de gerar)")
    args = parser.parse_args()

    try:
        import mediapipe  # noqa: F401
    except ImportError:
        print("[ERRO] mediapipe não instalado.")
        print("       Execute: pip install mediapipe")
        sys.exit(1)

    SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    if args.download:
        download_sample_faces(args.n)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    has_photos = SOURCE_DIR.exists() and any(
        p.suffix.lower() in exts for p in SOURCE_DIR.iterdir()
    )
    if not has_photos:
        print(f"[ERRO] Nenhuma foto em {SOURCE_DIR}/")
        print("       Use --download para baixar amostras automáticas.")
        sys.exit(1)

    generate_real(clear_old=not args.keep_old)


if __name__ == "__main__":
    main()
