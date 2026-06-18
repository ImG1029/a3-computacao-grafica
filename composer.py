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

_FACE_PARAMS: dict[str, dict] = {
    "oval_padrao":  dict(top=145, bot=488, cx=250, hw=118),
    "oval_redondo": dict(top=168, bot=470, cx=250, hw=140),
    "quadrado":     dict(top=142, bot=478, cx=250, hw=120),
    "alongado":     dict(top=108, bot=510, cx=250, hw=102),
}

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
    h = p["bot"] - p["top"]
    y1 = p["top"] + y_top_r * h
    y2 = p["top"] + y_bot_r * h
    x1 = p["cx"] + x_left_r * p["hw"]
    x2 = p["cx"] + x_right_r * p["hw"]
    return np.float32([[x1, y1], [x2, y1], [x1, y2]])


def _extract_face_shape(rosto_path: Path | None) -> str:
    if rosto_path is None:
        return "oval_padrao"
    stem = rosto_path.stem
    for shape in _FACE_PARAMS:
        if stem.startswith(shape):
            return shape
    return "oval_padrao"


def _warp_component(img: Image.Image, category: str, face_shape: str) -> Image.Image:
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

_FILL_SIGMA = 22.0


def _silhouette(alpha: np.ndarray) -> np.ndarray:
    hard = (alpha > 0.4).astype(np.uint8) * 255
    img = Image.fromarray(hard).filter(ImageFilter.MaxFilter(5))
    img = img.filter(ImageFilter.GaussianBlur(6))
    return np.asarray(img, dtype=np.float32) / 255.0


def _flatten_seamless(strips: Image.Image) -> Image.Image:
    arr = np.asarray(strips, dtype=np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3] / 255.0
    if alpha.max() <= 0:
        return Image.new("RGBA", CANVAS_SIZE, (255, 255, 255, 255))

    sil = _silhouette(alpha)[:, :, None]
    a3 = alpha[:, :, None]

    fill_num = cv2.GaussianBlur(rgb * a3, (0, 0), _FILL_SIGMA)
    fill_den = cv2.GaussianBlur(alpha, (0, 0), _FILL_SIGMA)[:, :, None] + 1e-6
    fill = fill_num / fill_den

    filled = rgb * a3 + fill * (1 - a3)
    out = filled * sil + 255.0 * (1 - sil)

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
