"""Generates semi-realistic PNG face components (500x600, RGBA).

Each component is built from multiple layers (base shape, gradients,
highlights, shadows, hair strokes) to approximate volume and skin tones.
The geometry is anatomically aligned so that overlays compose correctly:

    forehead   y ~ 150-220
    eyebrows   y ~ 235-260
    eyes       y ~ 270-300   (centers x = 188 / 312)
    nose       y ~ 290-380   (axis  x = 250)
    mouth      y ~ 405-445
    jaw/chin   y ~ 440-490
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

W, H = 500, 600
CX = 250
FACES_DIR = Path(__file__).parent / "faces"

# ── PALETTES (warm caucasian skin, natural hair/eye tones) ────────────────
SKIN_BASE       = (232, 188, 152)
SKIN_LIGHT      = (250, 218, 188)
SKIN_MID        = (210, 162, 122)
SKIN_DARK       = (165, 115, 80)
SKIN_DEEP       = (110, 70, 45)
BLUSH           = (220, 130, 110)

LIP_LIGHT       = (210, 130, 118)
LIP_BASE        = (175, 90, 80)
LIP_DARK        = (120, 55, 50)

HAIR_BLACK      = (22, 18, 16)
HAIR_BLACK_HL   = (75, 65, 60)
HAIR_BROWN      = (75, 45, 22)
HAIR_BROWN_HL   = (135, 90, 50)
HAIR_GRAY       = (130, 125, 120)
HAIR_GRAY_HL    = (190, 188, 185)

EYE_WHITE       = (245, 240, 232)
EYE_WHITE_SH    = (215, 205, 195)
IRIS_BROWN_OUT  = (75, 48, 22)
IRIS_BROWN_IN   = (155, 110, 60)
IRIS_BLUE_OUT   = (35, 80, 130)
IRIS_BLUE_IN    = (130, 180, 220)
PUPIL           = (10, 8, 6)
LASH            = (15, 12, 10)


# ── PRIMITIVES ────────────────────────────────────────────────────────────

def _canvas() -> Image.Image:
    return Image.new("RGBA", (W, H), (0, 0, 0, 0))


def _save(img: Image.Image, name: str, category: str) -> None:
    path = FACES_DIR / category / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    print(f"  {path.relative_to(FACES_DIR.parent)}")


def _radial_mask(cx: float, cy: float, inner_r: float, outer_r: float) -> Image.Image:
    """L-mode radial gradient: 255 at center, 0 outside outer_r."""
    y, x = np.ogrid[:H, :W]
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    spread = max(outer_r - inner_r, 1.0)
    arr = np.clip(255 * (outer_r - d) / spread, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L")


def _ellipse_mask(bbox, feather: float = 0) -> Image.Image:
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).ellipse(bbox, fill=255)
    if feather:
        img = img.filter(ImageFilter.GaussianBlur(feather))
    return img


def _color_layer(color: tuple[int, int, int], mask: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", (W, H), (*color, 0))
    layer.putalpha(mask)
    return layer


def _clip(mask: Image.Image, clip_to: Image.Image) -> Image.Image:
    return ImageChops.multiply(mask, clip_to)


def _scale_alpha(mask: Image.Image, factor: float) -> Image.Image:
    return mask.point(lambda p: int(p * factor))


# ── ROSTO ─────────────────────────────────────────────────────────────────

FACE_GEOMETRY = {
    "oval_padrao":  {"face": (132, 145, 368, 488), "ears": (118, 282, 144, 342, 356, 282, 382, 342)},
    "oval_redondo": {"face": (110, 168, 390, 470), "ears": (96, 282, 124, 342, 376, 282, 404, 342)},
    "quadrado":     {"face": None, "ears": (118, 282, 144, 342, 356, 282, 382, 342)},
    "alongado":     {"face": (148, 108, 352, 510), "ears": (134, 268, 158, 338, 342, 268, 366, 338)},
}


def _face_alpha(shape: str) -> Image.Image:
    """Soft alpha mask of face + neck + ears (no head hair)."""
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)

    if shape == "quadrado":
        # Rounded square with a gentle taper toward the chin.
        d.rounded_rectangle([130, 142, 370, 478], radius=58, fill=255)
        d.polygon([(140, 410), (360, 410), (340, 488), (160, 488)], fill=255)
    else:
        bbox = FACE_GEOMETRY[shape]["face"]
        d.ellipse(bbox, fill=255)

    ex1, ey1, ex2, ey2, ex3, ey3, ex4, ey4 = FACE_GEOMETRY[shape]["ears"]
    d.ellipse([ex1, ey1, ex2, ey2], fill=255)
    d.ellipse([ex3, ey3, ex4, ey4], fill=255)

    # Neck
    if shape == "alongado":
        d.polygon([(212, 500), (288, 500), (300, 600), (200, 600)], fill=255)
    else:
        d.polygon([(208, 470), (292, 470), (302, 600), (198, 600)], fill=255)

    return mask.filter(ImageFilter.GaussianBlur(1.2))


def _build_face(shape: str) -> Image.Image:
    face = _face_alpha(shape)
    out = _canvas()

    # Base skin
    out = Image.alpha_composite(out, _color_layer(SKIN_BASE, face))

    # Forehead/cheek highlight (soft, centered)
    hl = _radial_mask(CX, 250, 20, 170)
    out = Image.alpha_composite(out, _color_layer(SKIN_LIGHT, _scale_alpha(_clip(hl, face), 0.55)))

    # T-zone vertical highlight (down the nose bridge to chin)
    tzone = Image.new("L", (W, H), 0)
    ImageDraw.Draw(tzone).ellipse([CX - 28, 230, CX + 28, 460], fill=255)
    tzone = tzone.filter(ImageFilter.GaussianBlur(28))
    out = Image.alpha_composite(out, _color_layer(SKIN_LIGHT, _scale_alpha(_clip(tzone, face), 0.30)))

    # Cheek blush
    for cheek_x in (188, 312):
        cheek = _radial_mask(cheek_x, 348, 0, 60).filter(ImageFilter.GaussianBlur(10))
        out = Image.alpha_composite(out, _color_layer(BLUSH, _scale_alpha(_clip(cheek, face), 0.28)))

    # Lateral shadows (depth on the sides of the face)
    for sx, sw in ((118, 90), (290, 90)):
        side = Image.new("L", (W, H), 0)
        ImageDraw.Draw(side).ellipse([sx, 160, sx + sw, 480], fill=255)
        side = side.filter(ImageFilter.GaussianBlur(24))
        out = Image.alpha_composite(out, _color_layer(SKIN_DARK, _scale_alpha(_clip(side, face), 0.50)))

    # Cheekbone definition (subtle dark crescent under each cheekbone)
    for cx_, cy_ in ((178, 360), (322, 360)):
        crescent = Image.new("L", (W, H), 0)
        ImageDraw.Draw(crescent).ellipse([cx_ - 35, cy_ - 12, cx_ + 35, cy_ + 12], fill=255)
        crescent = crescent.filter(ImageFilter.GaussianBlur(8))
        out = Image.alpha_composite(out, _color_layer(SKIN_DARK, _scale_alpha(_clip(crescent, face), 0.18)))

    # Chin/jaw under-shadow
    chin = Image.new("L", (W, H), 0)
    ImageDraw.Draw(chin).ellipse([155, 450, 345, 510], fill=255)
    chin = chin.filter(ImageFilter.GaussianBlur(14))
    out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, _scale_alpha(_clip(chin, face), 0.30)))

    # Neck shadow under jaw
    neck_sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(neck_sh).ellipse([175, 478, 325, 520], fill=255)
    neck_sh = neck_sh.filter(ImageFilter.GaussianBlur(10))
    out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, _scale_alpha(_clip(neck_sh, face), 0.55)))

    # Ear inner shadow
    for ex, ey, ex2, ey2 in ((118, 282, 144, 342), (356, 282, 382, 342)):
        inner = Image.new("L", (W, H), 0)
        ImageDraw.Draw(inner).ellipse([ex + 4, ey + 8, ex2 - 4, ey2 - 8], fill=255)
        inner = inner.filter(ImageFilter.GaussianBlur(2))
        out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, _scale_alpha(inner, 0.45)))

    # Soft outline along face perimeter
    edge = face.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.6))
    outline_alpha = edge.point(lambda p: min(p, 95))
    out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, outline_alpha))

    return out


def _gen_rosto() -> None:
    for shape in ("oval_padrao", "oval_redondo", "quadrado", "alongado"):
        _save(_build_face(shape), shape, "rosto")


# ── CABELO ────────────────────────────────────────────────────────────────

def _hair_strokes(mask_region: Image.Image, color: tuple[int, int, int],
                  density: int, length_range: tuple[int, int],
                  angle_jitter: float = 25, seed: int = 0,
                  clip: bool = True) -> Image.Image:
    """Draw many fine strokes inside a mask to simulate hair texture."""
    rng = random.Random(seed)
    arr = np.array(mask_region) > 30
    ys, xs = np.where(arr)
    if not len(xs):
        return _canvas()

    layer = _canvas()
    d = ImageDraw.Draw(layer)
    for _ in range(density):
        i = rng.randrange(len(xs))
        x, y = int(xs[i]), int(ys[i])
        length = rng.randint(*length_range)
        angle = math.radians(90 + rng.uniform(-angle_jitter, angle_jitter))
        x2 = x + int(length * math.cos(angle))
        y2 = y + int(length * math.sin(angle))
        jitter = rng.randint(-12, 12)
        c = tuple(max(0, min(255, ch + jitter)) for ch in color)
        d.line([(x, y), (x2, y2)], fill=(*c, 230), width=1)

    if clip:
        r, g, b, a = layer.split()
        a = ImageChops.multiply(a, mask_region)
        layer = Image.merge("RGBA", (r, g, b, a))
    return layer


def _hair_cap_mask(top_y: int, bot_y: int, left_x: int, right_x: int,
                   center_dip: int = 0) -> Image.Image:
    """Helmet-like hair mask: ellipse top + flat-ish bottom hairline."""
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse([left_x, top_y, right_x, bot_y + 40], fill=255)
    # Carve out the lower face area (so hair sits on top of forehead only)
    d.rectangle([0, 240 + center_dip, W, H], fill=0)
    return mask.filter(ImageFilter.GaussianBlur(2))


def _build_hair(shape: str, base_color: tuple[int, int, int],
                hl_color: tuple[int, int, int]) -> Image.Image:
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)

    if shape == "curto":
        # Scalp cap over the upper head, hairline arching down the forehead
        d.pieslice([72, 80, 428, 320], start=180, end=360, fill=255)  # top half of head
        # Add slight forehead coverage that arcs down at the temples
        d.chord([95, 165, 405, 270], start=180, end=360, fill=255)
        # Sideburns
        d.ellipse([90, 190, 138, 290], fill=255)
        d.ellipse([362, 190, 410, 290], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(2.0))

    elif shape == "longo":
        # Top cap covering the head
        d.pieslice([68, 72, 432, 320], start=180, end=360, fill=255)
        d.chord([95, 160, 405, 270], start=180, end=360, fill=255)
        # Side hair flowing past the ears, narrowing toward the tips
        d.polygon([(82, 195), (78, 540), (140, 560), (152, 230)], fill=255)
        d.polygon([(418, 195), (422, 540), (360, 560), (348, 230)], fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(3.0))

    out = _canvas()

    # Drop shadow underneath (creates depth between hair and forehead)
    shadow = mask.filter(ImageFilter.GaussianBlur(8))
    sh_layer = _color_layer((0, 0, 0), _scale_alpha(shadow, 0.35))
    sh_layer = ImageChops.offset(sh_layer, 0, 6)
    out = Image.alpha_composite(out, sh_layer)

    # Base hair color
    out = Image.alpha_composite(out, _color_layer(base_color, mask))

    # Bottom shadow (darken lower portion)
    bottom_sh = mask.copy()
    sh_arr = np.array(bottom_sh).astype(np.float32)
    y_grad = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    sh_arr = sh_arr * y_grad
    bottom_sh = Image.fromarray(np.clip(sh_arr, 0, 255).astype(np.uint8), "L")
    out = Image.alpha_composite(out, _color_layer(_darken(base_color, 0.55), _scale_alpha(bottom_sh, 0.5)))

    # Strand texture
    seed = sum(base_color)
    strokes_dark = _hair_strokes(mask, _darken(base_color, 0.65), 1400, (8, 22), 20, seed)
    strokes_light = _hair_strokes(mask, hl_color, 700, (6, 16), 20, seed + 1)
    out = Image.alpha_composite(out, _scale_alpha_layer(strokes_dark, 0.55))
    out = Image.alpha_composite(out, _scale_alpha_layer(strokes_light, 0.45))

    # Top highlight band
    hl_band = Image.new("L", (W, H), 0)
    ImageDraw.Draw(hl_band).ellipse([130, 100, 370, 175], fill=255)
    hl_band = hl_band.filter(ImageFilter.GaussianBlur(18))
    out = Image.alpha_composite(out, _color_layer(hl_color, _scale_alpha(_clip(hl_band, mask), 0.42)))

    return out


def _scale_alpha_layer(rgba: Image.Image, factor: float) -> Image.Image:
    r, g, b, a = rgba.split()
    a = a.point(lambda p: int(p * factor))
    return Image.merge("RGBA", (r, g, b, a))


def _darken(color: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(int(c * factor) for c in color)


def _gen_cabelo() -> None:
    _save(_canvas(), "careca", "cabelo")
    _save(_build_hair("curto", HAIR_BLACK, HAIR_BLACK_HL), "curto_escuro", "cabelo")
    _save(_build_hair("curto", HAIR_GRAY, HAIR_GRAY_HL), "curto_grisalho", "cabelo")
    _save(_build_hair("longo", HAIR_BROWN, HAIR_BROWN_HL), "longo_castanho", "cabelo")
    _save(_build_hair("longo", HAIR_BLACK, HAIR_BLACK_HL), "longo_preto", "cabelo")


# ── OLHOS ─────────────────────────────────────────────────────────────────

EYE_CENTERS = ((188, 285), (312, 285))


def _draw_eye(out: Image.Image, cx: int, cy: int,
              iris_out: tuple[int, int, int],
              iris_in: tuple[int, int, int],
              narrow: bool = False) -> Image.Image:
    """Anatomical eye: socket shadow → sclera → iris (radial) → pupil → highlight → lashes."""
    half_w = 36
    half_h = 11 if narrow else 16

    # Socket shadow (subtle skin tone shadow around eye)
    socket = Image.new("L", (W, H), 0)
    ImageDraw.Draw(socket).ellipse([cx - half_w - 8, cy - half_h - 6, cx + half_w + 8, cy + half_h + 8], fill=255)
    socket = socket.filter(ImageFilter.GaussianBlur(4))
    out = Image.alpha_composite(out, _color_layer(SKIN_DARK, _scale_alpha(socket, 0.18)))

    # Sclera (white of eye), almond shape
    eye_shape = Image.new("L", (W, H), 0)
    ed = ImageDraw.Draw(eye_shape)
    if narrow:
        ed.polygon([
            (cx - half_w, cy + 1),
            (cx - 12, cy - half_h),
            (cx + 12, cy - half_h),
            (cx + half_w, cy + 1),
            (cx + 12, cy + half_h),
            (cx - 12, cy + half_h),
        ], fill=255)
    else:
        ed.ellipse([cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=255)
    eye_shape = eye_shape.filter(ImageFilter.GaussianBlur(0.7))

    out = Image.alpha_composite(out, _color_layer(EYE_WHITE, eye_shape))

    # Upper sclera shadow (cast by upper lid)
    upper_sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(upper_sh).ellipse([cx - half_w, cy - half_h - 4, cx + half_w, cy - half_h + 8], fill=255)
    upper_sh = upper_sh.filter(ImageFilter.GaussianBlur(3))
    out = Image.alpha_composite(out, _color_layer(EYE_WHITE_SH, _scale_alpha(_clip(upper_sh, eye_shape), 0.85)))

    # Iris (radial gradient from inner color to outer color)
    iris_r = 13 if narrow else 14
    iris_outer = _radial_mask(cx, cy, 0, iris_r)
    iris_outer = _clip(iris_outer, eye_shape)
    out = Image.alpha_composite(out, _color_layer(iris_out, iris_outer))
    iris_inner = _radial_mask(cx, cy, 0, iris_r - 5)
    iris_inner = _clip(iris_inner, eye_shape)
    out = Image.alpha_composite(out, _color_layer(iris_in, _scale_alpha(iris_inner, 0.85)))

    # Iris fine streaks (texture)
    streak_layer = _canvas()
    sd = ImageDraw.Draw(streak_layer)
    rng = random.Random(cx)
    for _ in range(40):
        ang = rng.uniform(0, 2 * math.pi)
        r1 = rng.uniform(2, iris_r - 2)
        r2 = rng.uniform(iris_r - 2, iris_r)
        x1, y1 = cx + r1 * math.cos(ang), cy + r1 * math.sin(ang)
        x2, y2 = cx + r2 * math.cos(ang), cy + r2 * math.sin(ang)
        sd.line([(x1, y1), (x2, y2)], fill=(*iris_out, 140), width=1)
    streak_layer.putalpha(_clip(streak_layer.split()[3], eye_shape))
    out = Image.alpha_composite(out, streak_layer)

    # Pupil
    pupil_mask = _ellipse_mask([cx - 5, cy - 5, cx + 5, cy + 5], feather=0.5)
    pupil_mask = _clip(pupil_mask, eye_shape)
    out = Image.alpha_composite(out, _color_layer(PUPIL, pupil_mask))

    # Catchlight highlight
    hl_mask = _ellipse_mask([cx - 4, cy - 7, cx, cy - 3], feather=0.3)
    out = Image.alpha_composite(out, _color_layer((255, 255, 255), hl_mask))

    # Upper eyelid line (thick curved dark line)
    lid = Image.new("L", (W, H), 0)
    ld = ImageDraw.Draw(lid)
    if narrow:
        ld.line([(cx - half_w, cy + 1), (cx - 12, cy - half_h),
                 (cx + 12, cy - half_h), (cx + half_w, cy + 1)], fill=255, width=3)
    else:
        ld.arc([cx - half_w, cy - half_h - 1, cx + half_w, cy + half_h], start=180, end=360, fill=255, width=3)
    lid = lid.filter(ImageFilter.GaussianBlur(0.8))
    out = Image.alpha_composite(out, _color_layer(LASH, lid))

    # Lash strokes (a few short hairs along upper lid)
    lash_layer = _canvas()
    lashd = ImageDraw.Draw(lash_layer)
    n_lashes = 8
    for i in range(n_lashes):
        t = i / (n_lashes - 1)
        x = cx - half_w + t * 2 * half_w
        y = cy - half_h + 4 * abs(t - 0.5) ** 2
        x2 = x + 2 * (t - 0.5)
        y2 = y - 6
        lashd.line([(x, y), (x2, y2)], fill=(*LASH, 220), width=1)
    out = Image.alpha_composite(out, lash_layer)

    # Lower lid subtle line
    lower = Image.new("L", (W, H), 0)
    ld2 = ImageDraw.Draw(lower)
    if narrow:
        ld2.line([(cx - half_w, cy + 1), (cx - 12, cy + half_h),
                  (cx + 12, cy + half_h), (cx + half_w, cy + 1)], fill=255, width=1)
    else:
        ld2.arc([cx - half_w, cy - half_h, cx + half_w, cy + half_h + 1], start=0, end=180, fill=255, width=1)
    lower = lower.filter(ImageFilter.GaussianBlur(0.5))
    out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, _scale_alpha(lower, 0.6)))

    return out


def _gen_olhos() -> None:
    for name, iris_out, iris_in, narrow in (
        ("abertos_marrom", IRIS_BROWN_OUT, IRIS_BROWN_IN, False),
        ("abertos_azul",   IRIS_BLUE_OUT,  IRIS_BLUE_IN,  False),
        ("estreitos_marrom", IRIS_BROWN_OUT, IRIS_BROWN_IN, True),
    ):
        out = _canvas()
        for cx, cy in EYE_CENTERS:
            out = _draw_eye(out, cx, cy, iris_out, iris_in, narrow=narrow)
        _save(out, name, "olhos")

    # Glasses on top of brown eyes
    out = _canvas()
    for cx, cy in EYE_CENTERS:
        out = _draw_eye(out, cx, cy, IRIS_BROWN_OUT, IRIS_BROWN_IN, narrow=False)
    gd = ImageDraw.Draw(out)
    # Lenses
    gd.rounded_rectangle([148, 263, 228, 313], radius=14, outline=(30, 25, 22, 255), width=4)
    gd.rounded_rectangle([272, 263, 352, 313], radius=14, outline=(30, 25, 22, 255), width=4)
    # Bridge
    gd.line([(228, 280), (272, 280)], fill=(30, 25, 22, 255), width=4)
    # Temples
    gd.line([(148, 282), (108, 270)], fill=(30, 25, 22, 255), width=3)
    gd.line([(352, 282), (392, 270)], fill=(30, 25, 22, 255), width=3)
    # Faint lens reflection
    refl = Image.new("L", (W, H), 0)
    ImageDraw.Draw(refl).polygon([(160, 270), (200, 270), (180, 305), (155, 305)], fill=255)
    refl = refl.filter(ImageFilter.GaussianBlur(1.5))
    out = Image.alpha_composite(out, _color_layer((255, 255, 255), _scale_alpha(refl, 0.20)))
    refl2 = Image.new("L", (W, H), 0)
    ImageDraw.Draw(refl2).polygon([(284, 270), (324, 270), (304, 305), (279, 305)], fill=255)
    refl2 = refl2.filter(ImageFilter.GaussianBlur(1.5))
    out = Image.alpha_composite(out, _color_layer((255, 255, 255), _scale_alpha(refl2, 0.20)))
    _save(out, "com_oculos", "olhos")


# ── SOBRANCELHAS ──────────────────────────────────────────────────────────

def _draw_brow(out: Image.Image, cx: int, cy: int, color: tuple[int, int, int],
               thick: bool, mirror: bool, seed: int) -> Image.Image:
    """Eyebrow built from many short strokes following a slight upward arc."""
    rng = random.Random(seed)
    layer = _canvas()
    d = ImageDraw.Draw(layer)

    span = 60
    height = 10 if thick else 6
    n_strokes = 220 if thick else 120

    for _ in range(n_strokes):
        t = rng.uniform(0, 1)
        x = cx - span / 2 + t * span
        # Arc: lower at the ends, peak slightly inward
        peak_t = 0.4 if mirror else 0.6
        y = cy - height * (1 - 4 * (t - peak_t) ** 2)
        # Stroke direction: outward and slightly up
        outward = -1 if mirror else 1
        angle = math.radians(15 * outward + rng.uniform(-12, 12))
        length = rng.randint(5, 11)
        x2 = x + length * math.cos(angle)
        y2 = y + length * math.sin(angle) - rng.uniform(0, 2)
        jitter = rng.randint(-15, 10)
        c = tuple(max(0, min(255, ch + jitter)) for ch in color)
        a = rng.randint(180, 240)
        d.line([(x, y), (x2, y2)], fill=(*c, a), width=1)

    return Image.alpha_composite(out, layer)


def _gen_sobrancelhas() -> None:
    for name, color, thick in (
        ("finas_escuras",     HAIR_BLACK, False),
        ("grossas_escuras",   HAIR_BLACK, True),
        ("grossas_castanhas", HAIR_BROWN, True),
    ):
        out = _canvas()
        out = _draw_brow(out, 188, 248, color, thick, mirror=True,  seed=hash(name) & 0xffff)
        out = _draw_brow(out, 312, 248, color, thick, mirror=False, seed=(hash(name) >> 8) & 0xffff)
        _save(out, name, "sobrancelhas")


# ── NARIZ ─────────────────────────────────────────────────────────────────

def _draw_nose(width_factor: float = 1.0, length_factor: float = 1.0,
               tip_lift: int = 0) -> Image.Image:
    out = _canvas()
    nose_top = 295
    nose_tip_y = int(nose_top + 80 * length_factor) - tip_lift
    half = int(18 * width_factor)
    tip_half = int(22 * width_factor)

    # Nose shape mask (used to clip shading)
    shape = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shape)
    sd.polygon([
        (CX - 4, nose_top),
        (CX - half, nose_tip_y - 18),
        (CX - tip_half, nose_tip_y),
        (CX, nose_tip_y + 6),
        (CX + tip_half, nose_tip_y),
        (CX + half, nose_tip_y - 18),
        (CX + 4, nose_top),
    ], fill=255)
    shape = shape.filter(ImageFilter.GaussianBlur(3))

    # Bridge highlight (thin vertical bright band)
    bridge_hl = Image.new("L", (W, H), 0)
    ImageDraw.Draw(bridge_hl).rectangle([CX - 3, nose_top, CX + 3, nose_tip_y - 8], fill=255)
    bridge_hl = bridge_hl.filter(ImageFilter.GaussianBlur(5))
    out = Image.alpha_composite(out, _color_layer(SKIN_LIGHT, _scale_alpha(bridge_hl, 0.55)))

    # Side shadows (left and right of bridge)
    for dx in (-half - 2, half + 2):
        side = Image.new("L", (W, H), 0)
        ImageDraw.Draw(side).ellipse([CX + dx - 8, nose_top + 5, CX + dx + 8, nose_tip_y - 5], fill=255)
        side = side.filter(ImageFilter.GaussianBlur(8))
        out = Image.alpha_composite(out, _color_layer(SKIN_DARK, _scale_alpha(side, 0.55)))

    # Tip rounded highlight
    tip_hl = _radial_mask(CX, nose_tip_y - 4, 0, 14)
    out = Image.alpha_composite(out, _color_layer(SKIN_LIGHT, _scale_alpha(tip_hl, 0.45)))

    # Tip lower shadow (under tip)
    tip_sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(tip_sh).ellipse([CX - tip_half, nose_tip_y - 4, CX + tip_half, nose_tip_y + 12], fill=255)
    tip_sh = tip_sh.filter(ImageFilter.GaussianBlur(4))
    out = Image.alpha_composite(out, _color_layer(SKIN_DARK, _scale_alpha(tip_sh, 0.30)))

    # Nostrils (two small dark ovals)
    nostril_y = nose_tip_y - 2
    nostril_dx = max(7, int(tip_half * 0.45))
    for dx in (-nostril_dx, nostril_dx):
        nost = _ellipse_mask(
            [CX + dx - 5, nostril_y - 2, CX + dx + 5, nostril_y + 4],
            feather=1.0,
        )
        out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, _scale_alpha(nost, 0.78)))

    # Outline at bottom of nose (curve under tip)
    underline = Image.new("L", (W, H), 0)
    ImageDraw.Draw(underline).arc(
        [CX - tip_half - 2, nose_tip_y - 8, CX + tip_half + 2, nose_tip_y + 14],
        start=0, end=180, fill=255, width=2,
    )
    underline = underline.filter(ImageFilter.GaussianBlur(0.8))
    out = Image.alpha_composite(out, _color_layer(SKIN_DEEP, _scale_alpha(underline, 0.45)))

    return out


def _gen_nariz() -> None:
    _save(_draw_nose(width_factor=0.85, length_factor=0.85, tip_lift=0), "pequeno", "nariz")
    _save(_draw_nose(width_factor=1.25, length_factor=1.05, tip_lift=0), "largo", "nariz")
    _save(_draw_nose(width_factor=0.95, length_factor=0.80, tip_lift=12), "arrebitado", "nariz")


# ── BOCA ──────────────────────────────────────────────────────────────────

def _draw_mouth(width: int = 110, lower_lip_factor: float = 1.0,
                smile: bool = False, thin: bool = False) -> Image.Image:
    out = _canvas()
    cy = 415
    half_w = width // 2
    upper_h = 6 if thin else 10
    lower_h = int((10 if thin else 14) * lower_lip_factor)

    # Mouth corner Y offset for smile
    corner_dy = -6 if smile else 0

    # Upper lip mask: cupid's bow on top, arched bottom
    upper = Image.new("L", (W, H), 0)
    ud = ImageDraw.Draw(upper)
    ud.polygon([
        (CX - half_w, cy + corner_dy),
        (CX - half_w + 12, cy - upper_h + 2),
        (CX - 18, cy - upper_h),
        (CX - 6, cy - upper_h + 4),    # cupid dip left
        (CX, cy - upper_h - 1),         # cupid peak
        (CX + 6, cy - upper_h + 4),    # cupid dip right
        (CX + 18, cy - upper_h),
        (CX + half_w - 12, cy - upper_h + 2),
        (CX + half_w, cy + corner_dy),
        (CX, cy + 2),
    ], fill=255)
    upper = upper.filter(ImageFilter.GaussianBlur(0.8))

    # Lower lip mask: rounded fuller
    lower = Image.new("L", (W, H), 0)
    ld = ImageDraw.Draw(lower)
    ld.polygon([
        (CX - half_w, cy + corner_dy),
        (CX, cy + 2),
        (CX + half_w, cy + corner_dy),
        (CX + half_w - 18, cy + lower_h),
        (CX, cy + lower_h + 4),
        (CX - half_w + 18, cy + lower_h),
    ], fill=255)
    lower = lower.filter(ImageFilter.GaussianBlur(1.2))

    # Base lip color
    out = Image.alpha_composite(out, _color_layer(LIP_BASE, upper))
    out = Image.alpha_composite(out, _color_layer(LIP_LIGHT, lower))

    # Upper lip darker (shadow on upper)
    upper_sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(upper_sh).ellipse(
        [CX - half_w, cy - upper_h - 2, CX + half_w, cy + 2], fill=255,
    )
    upper_sh = upper_sh.filter(ImageFilter.GaussianBlur(2))
    out = Image.alpha_composite(out, _color_layer(LIP_DARK, _scale_alpha(_clip(upper_sh, upper), 0.40)))

    # Lower lip highlight
    lower_hl = Image.new("L", (W, H), 0)
    ImageDraw.Draw(lower_hl).ellipse(
        [CX - 28, cy + 4, CX + 28, cy + lower_h - 2], fill=255,
    )
    lower_hl = lower_hl.filter(ImageFilter.GaussianBlur(3))
    out = Image.alpha_composite(out, _color_layer((245, 195, 180), _scale_alpha(_clip(lower_hl, lower), 0.50)))

    # Lip line (between upper and lower lip)
    line = Image.new("L", (W, H), 0)
    if smile:
        ImageDraw.Draw(line).arc(
            [CX - half_w, cy - 3, CX + half_w, cy + 8], start=0, end=180, fill=255, width=2,
        )
    else:
        ImageDraw.Draw(line).line(
            [(CX - half_w + 2, cy + 1), (CX, cy + 3), (CX + half_w - 2, cy + 1)],
            fill=255, width=2,
        )
    line = line.filter(ImageFilter.GaussianBlur(0.6))
    out = Image.alpha_composite(out, _color_layer(LIP_DARK, _scale_alpha(line, 0.85)))

    # Faint shadow under lower lip (chin shadow)
    chin_sh = Image.new("L", (W, H), 0)
    ImageDraw.Draw(chin_sh).ellipse(
        [CX - 35, cy + lower_h, CX + 35, cy + lower_h + 12], fill=255,
    )
    chin_sh = chin_sh.filter(ImageFilter.GaussianBlur(5))
    out = Image.alpha_composite(out, _color_layer(SKIN_DARK, _scale_alpha(chin_sh, 0.20)))

    return out


def _gen_boca() -> None:
    _save(_draw_mouth(width=104, lower_lip_factor=1.0, smile=False, thin=False), "neutra", "boca")
    _save(_draw_mouth(width=120, lower_lip_factor=0.9, smile=True,  thin=False), "sorrindo", "boca")
    _save(_draw_mouth(width=98,  lower_lip_factor=0.6, smile=False, thin=True),  "fina", "boca")
    _save(_draw_mouth(width=108, lower_lip_factor=1.4, smile=False, thin=False), "grossa", "boca")


# ── BARBA ─────────────────────────────────────────────────────────────────

def _beard_region_mask(kind: str) -> Image.Image:
    """Soft mask of the area where beard hair lives."""
    mask = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(mask)
    if kind == "curta":
        # Stubble across lower jaw + upper lip + chin
        d.ellipse([130, 380, 370, 500], fill=255)
        carve = Image.new("L", (W, H), 0)
        ImageDraw.Draw(carve).ellipse([195, 395, 305, 445], fill=255)
        mask = ImageChops.subtract(mask, carve)
    elif kind == "longa":
        # Full beard along the jawline, tapering down past chin
        d.ellipse([130, 395, 370, 525], fill=255)
        d.polygon([(180, 460), (320, 460), (320, 560), (180, 560)], fill=255)
        # Carve: keep the lip+mouth area clear (wider than the mouth itself)
        carve = Image.new("L", (W, H), 0)
        ImageDraw.Draw(carve).ellipse([180, 388, 320, 442], fill=255)
        mask = ImageChops.subtract(mask, carve)
    elif kind == "bigode":
        d.polygon([(190, 388), (250, 380), (310, 388), (300, 410), (260, 405), (240, 405), (200, 410)], fill=255)
    return mask.filter(ImageFilter.GaussianBlur(2))


def _build_beard(kind: str, base_color: tuple[int, int, int],
                 hl_color: tuple[int, int, int]) -> Image.Image:
    region = _beard_region_mask(kind)
    out = _canvas()

    # Base soft fill (shadow underneath)
    if kind == "curta":
        fill_alpha = _scale_alpha(region, 0.55)
    elif kind == "longa":
        fill_alpha = _scale_alpha(region, 0.95)
    else:  # bigode
        fill_alpha = _scale_alpha(region, 0.95)
    out = Image.alpha_composite(out, _color_layer(_darken(base_color, 0.6), fill_alpha))

    # Strand texture
    seed = hash(kind) & 0xffff
    angle_jit = 18 if kind != "bigode" else 25
    density = {"curta": 4500, "longa": 6500, "bigode": 1200}[kind]
    length = {"curta": (3, 7), "longa": (4, 10), "bigode": (4, 9)}[kind]

    strokes = _hair_strokes(region, base_color, density, length, angle_jit, seed)
    out = Image.alpha_composite(out, strokes)
    light_strokes = _hair_strokes(region, hl_color, density // 4, length, angle_jit, seed + 1)
    out = Image.alpha_composite(out, _scale_alpha_layer(light_strokes, 0.5))

    # Small chin shadow under (depth)
    if kind in ("curta", "longa"):
        depth = Image.new("L", (W, H), 0)
        ImageDraw.Draw(depth).ellipse([180, 480, 320, 510], fill=255)
        depth = depth.filter(ImageFilter.GaussianBlur(6))
        out = Image.alpha_composite(out, _color_layer((0, 0, 0), _scale_alpha(depth, 0.20)))

    return out


def _gen_barba() -> None:
    _save(_canvas(), "sem_barba", "barba")
    _save(_build_beard("curta", HAIR_BLACK, HAIR_BLACK_HL), "curta_escura", "barba")
    _save(_build_beard("longa", HAIR_BLACK, HAIR_BLACK_HL), "longa_escura", "barba")
    _save(_build_beard("bigode", HAIR_BLACK, HAIR_BLACK_HL), "bigode", "barba")


# ── ENTRY POINT ───────────────────────────────────────────────────────────

def main() -> None:
    # Prioridade: fotos reais em source_faces/ + mediapipe instalado
    source_dir = Path(__file__).parent / "source_faces"
    _exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    has_photos = source_dir.exists() and any(
        p.suffix.lower() in _exts for p in source_dir.iterdir()
    )
    if has_photos:
        try:
            from generate_assets_real import generate_real
            print("Fotos reais encontradas em source_faces/. Gerando componentes realistas...")
            generate_real()
            return
        except ImportError:
            print("[AVISO] mediapipe não instalado. Usando gerador desenhado.")
            print("        Para componentes realistas: pip install mediapipe")

    print("Gerando assets de faces...")
    _gen_rosto()
    _gen_cabelo()
    _gen_sobrancelhas()
    _gen_olhos()
    _gen_nariz()
    _gen_boca()
    _gen_barba()
    total = sum(1 for _ in FACES_DIR.rglob("*.png"))
    print(f"\n{total} PNGs criados em {FACES_DIR.resolve()}")


if __name__ == "__main__":
    main()
