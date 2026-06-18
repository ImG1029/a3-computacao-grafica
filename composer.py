from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

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


# ── SEAMLESS FLATTENING ─────────────────────────────────────────────────────
#
# Each component is a horizontal strip whose alpha = min(vertical feather,
# feathered face oval). Toward the sides those two factors multiply, so the
# combined alpha never reaches 100%. Composited over the white canvas, that
# partial alpha lets the background bleed through as light/white streaks at the
# band junctions — the artifact this module fixes.
#
# The fix has two parts, applied after all components are stacked on a
# transparent canvas:
#   1. A single clean silhouette: one softly-feathered outer edge derived from
#      the union of every component's alpha, instead of each strip carrying its
#      own oval edge (which produced the serrated lateral streaks).
#   2. An alpha-weighted fill: behind every semi-transparent pixel we paint the
#      local average of the surrounding opaque face pixels, so the seams reveal
#      skin tone instead of the white background. Outside the silhouette the
#      canvas stays white, keeping the database-friendly white background.

_FILL_SIGMA = 22.0


def _silhouette(alpha: np.ndarray) -> np.ndarray:
    """One clean, softly-feathered mask from the union of component alphas."""
    hard = (alpha > 0.4).astype(np.uint8) * 255
    img = Image.fromarray(hard).filter(ImageFilter.MaxFilter(5))  # close tiny notches
    img = img.filter(ImageFilter.GaussianBlur(6))                 # single soft edge
    return np.asarray(img, dtype=np.float32) / 255.0


def _flatten_seamless(strips: Image.Image) -> Image.Image:
    """Flatten the stacked components over white without white seams."""
    arr = np.asarray(strips, dtype=np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3] / 255.0
    if alpha.max() <= 0:
        return Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 255))

    sil = _silhouette(alpha)[:, :, None]
    a3 = alpha[:, :, None]

    # Local face tone behind partial-alpha pixels (alpha-weighted Gaussian blur,
    # normalized so transparent areas borrow colour from opaque neighbours).
    fill_num = cv2.GaussianBlur(rgb * a3, (0, 0), _FILL_SIGMA)
    fill_den = cv2.GaussianBlur(alpha, (0, 0), _FILL_SIGMA)[:, :, None] + 1e-6
    fill = fill_num / fill_den

    filled = rgb * a3 + fill * (1 - a3)      # seams reveal skin, not white
    out = filled * sil + 255.0 * (1 - sil)   # outside the silhouette stays white

    rgba = np.dstack([out, np.full(alpha.shape, 255.0)])
    return Image.fromarray(rgba.clip(0, 255).astype(np.uint8), "RGBA")


def list_components(category: str) -> list[tuple[str, Path]]:
    path = FACES_DIR / category
    if not path.exists():
        return []
    return [
        (p.stem.replace("_", " ").title(), p)
        for p in sorted(path.glob("*.png"))
    ]


def compose(selection: dict[str, Path | None]) -> Image.Image:
    # Stack the components on a transparent canvas first, preserving alpha, so
    # _flatten_seamless can tell real face pixels from the (still empty) seams.
    strips = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
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
        strips = Image.alpha_composite(strips, component)

    return _flatten_seamless(strips)


def save(image: Image.Image, path: Path = OUTPUT_PATH) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path
