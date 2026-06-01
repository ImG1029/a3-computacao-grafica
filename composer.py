from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CANVAS_SIZE = (500, 600)
W, H = CANVAS_SIZE
FACES_DIR = Path(__file__).parent / "faces"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "retrato_falado_suspeito.png"

LAYER_ORDER = ["cabelo", "sobrancelhas", "olhos", "nariz", "boca", "queixo"]

CATEGORY_LABELS = {
    "cabelo": "Cabelo",
    "sobrancelhas": "Sobrancelhas",
    "olhos": "Olhos",
    "nariz": "Nariz",
    "boca": "Boca",
    "queixo": "Queixo",
}

# ── WARPING ───────────────────────────────────────────────────────────────────
#
# All components were generated for oval_padrao (the reference face).
# When a different face is selected we apply a per-category affine transform so
# each component stretches/shifts to match the target face's proportions.
#
# Each face is described by 4 numbers derived from generate_assets.py geometry:
#   top  – y coordinate of the top of the face ellipse
#   bot  – y coordinate of the bottom of the face ellipse
#   cx   – horizontal center (always 250)
#   hw   – horizontal half-width of the face ellipse
#
# Component zones are expressed as ratios relative to that bounding box:
#   y_ratios : 0 = face_top, 1 = face_bottom  (can be < 0 or > 1)
#   x_ratios : -1 = left face edge, +1 = right face edge

_FACE_PARAMS: dict[str, dict] = {
    # From FACE_GEOMETRY in generate_assets.py
    "oval_padrao":  dict(top=145, bot=488, cx=250, hw=118),  # reference
    "oval_redondo": dict(top=168, bot=470, cx=250, hw=140),
    "quadrado":     dict(top=142, bot=478, cx=250, hw=120),
    "alongado":     dict(top=108, bot=510, cx=250, hw=102),
}

# (y_top_ratio, y_bot_ratio, x_left_ratio, x_right_ratio)
# Derived from the actual pixel extents of each component in generate_assets.py.
_COMPONENT_ZONES: dict[str, tuple] = {
    "cabelo":       (-0.19,  0.40, -1.50,  1.50),
    "sobrancelhas": ( 0.26,  0.35, -1.03,  1.03),
    "olhos":        ( 0.36,  0.46, -0.86,  0.86),
    "nariz":        ( 0.44,  0.70, -0.19,  0.19),
    "boca":         ( 0.74,  0.88, -0.51,  0.51),
    "queixo":       ( 0.88,  1.05, -0.45,  0.45),
}

_REF = _FACE_PARAMS["oval_padrao"]


def _zone_pts(y_top_r: float, y_bot_r: float,
              x_left_r: float, x_right_r: float,
              p: dict) -> np.ndarray:
    """Three non-collinear control points for an affine transform from zone ratios."""
    h = p["bot"] - p["top"]
    y1 = p["top"] + y_top_r * h
    y2 = p["top"] + y_bot_r * h
    x1 = p["cx"] + x_left_r * p["hw"]
    x2 = p["cx"] + x_right_r * p["hw"]
    return np.float32([[x1, y1], [x2, y1], [x1, y2]])


def _extract_face_shape(rosto_path: Path | None) -> str:
    """Infer the face shape key from the selected rosto filename stem."""
    if rosto_path is None:
        return "oval_padrao"
    stem = rosto_path.stem  # e.g. "oval_redondo_14"
    for shape in _FACE_PARAMS:
        if stem.startswith(shape):
            return shape
    return "oval_padrao"


def _warp_component(img: Image.Image, category: str, face_shape: str) -> Image.Image:
    """Apply an affine warp to adapt a component to the selected face's geometry."""
    if category not in _COMPONENT_ZONES or face_shape not in _FACE_PARAMS:
        return img
    tgt = _FACE_PARAMS[face_shape]
    if tgt == _REF:
        return img

    zone = _COMPONENT_ZONES[category]
    src_pts = _zone_pts(*zone, _REF)
    dst_pts = _zone_pts(*zone, tgt)

    M = cv2.getAffineTransform(src_pts, dst_pts)
    arr = np.array(img)
    warped = cv2.warpAffine(
        arr, M, (W, H),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return Image.fromarray(warped)


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def list_components(category: str) -> list[tuple[str, Path]]:
    path = FACES_DIR / category
    if not path.exists():
        return []
    return [
        (p.stem.replace("_", " ").title(), p)
        for p in sorted(path.glob("*.png"))
    ]


def compose(selection: dict[str, Path | None]) -> Image.Image:
    canvas = Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 255))
    face_shape = _extract_face_shape(selection.get("rosto"))

    for layer in LAYER_ORDER:
        path = selection.get(layer)
        if not path:
            continue
        component = Image.open(path).convert("RGBA")
        if component.size != CANVAS_SIZE:
            component = component.resize(CANVAS_SIZE, Image.LANCZOS)
        if layer != "rosto":
            component = _warp_component(component, layer, face_shape)
        canvas = Image.alpha_composite(canvas, component)
    return canvas


def save(image: Image.Image, path: Path = OUTPUT_PATH) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path
