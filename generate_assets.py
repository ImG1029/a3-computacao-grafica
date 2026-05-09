"""Generates placeholder PNG face components (500x600, RGBA) for development."""
from pathlib import Path
from PIL import Image, ImageDraw

W, H = 500, 600
FACES_DIR = Path(__file__).parent / "faces"

# Color palette
SKIN = (240, 194, 153, 255)
SKIN_SHADOW = (195, 150, 110, 255)
BLACK = (20, 15, 10, 255)
BROWN = (101, 67, 33, 255)
DARK_BROWN = (65, 40, 15, 255)
GRAY = (140, 135, 130, 255)
EYE_WHITE = (242, 242, 245, 255)
IRIS_BROWN = (110, 72, 35, 255)
IRIS_BLUE = (70, 115, 170, 255)
PUPIL = (10, 8, 5, 255)
LIP = (210, 115, 105, 255)
LIP_DARK = (165, 75, 65, 255)


def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _save(img: Image.Image, name: str, category: str) -> None:
    path = FACES_DIR / category / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    print(f"  {path.relative_to(FACES_DIR.parent)}")


# ── ROSTO ──────────────────────────────────────────────────────────────────

def _gen_rosto() -> None:
    img, d = _canvas()
    d.ellipse([115, 140, 385, 485], fill=SKIN, outline=SKIN_SHADOW, width=2)
    _save(img, "oval_padrao", "rosto")

    img, d = _canvas()
    d.ellipse([90, 165, 410, 470], fill=SKIN, outline=SKIN_SHADOW, width=2)
    _save(img, "oval_redondo", "rosto")

    img, d = _canvas()
    d.rounded_rectangle([105, 145, 395, 480], radius=50, fill=SKIN, outline=SKIN_SHADOW, width=2)
    _save(img, "quadrado", "rosto")

    img, d = _canvas()
    d.ellipse([140, 100, 360, 510], fill=SKIN, outline=SKIN_SHADOW, width=2)
    _save(img, "alongado", "rosto")


# ── CABELO ─────────────────────────────────────────────────────────────────

def _gen_cabelo() -> None:
    # careca: blank layer
    img, _ = _canvas()
    _save(img, "careca", "cabelo")

    # curto escuro
    img, d = _canvas()
    d.ellipse([82, 82, 418, 248], fill=BLACK)
    _save(img, "curto_escuro", "cabelo")

    # longo castanho
    img, d = _canvas()
    d.ellipse([75, 78, 425, 252], fill=BROWN)
    d.polygon([(75, 178), (60, 480), (130, 500), (115, 232)], fill=BROWN)
    d.polygon([(425, 178), (440, 480), (370, 500), (385, 232)], fill=BROWN)
    _save(img, "longo_castanho", "cabelo")

    # curto grisalho
    img, d = _canvas()
    d.ellipse([82, 82, 418, 248], fill=GRAY)
    _save(img, "curto_grisalho", "cabelo")

    # longo preto
    img, d = _canvas()
    d.ellipse([75, 78, 425, 252], fill=BLACK)
    d.polygon([(75, 178), (55, 490), (125, 510), (112, 232)], fill=BLACK)
    d.polygon([(425, 178), (445, 490), (375, 510), (388, 232)], fill=BLACK)
    _save(img, "longo_preto", "cabelo")


# ── OLHOS ──────────────────────────────────────────────────────────────────

def _draw_eye(d: ImageDraw.ImageDraw, cx: int, cy: int, iris_color: tuple) -> None:
    d.ellipse([cx - 30, cy - 15, cx + 30, cy + 15], fill=EYE_WHITE, outline=BLACK, width=1)
    d.ellipse([cx - 14, cy - 13, cx + 14, cy + 13], fill=iris_color)
    d.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=PUPIL)
    d.ellipse([cx - 4, cy - 5, cx - 1, cy - 2], fill=(255, 255, 255, 200))


def _gen_olhos() -> None:
    img, d = _canvas()
    _draw_eye(d, 175, 268, IRIS_BROWN)
    _draw_eye(d, 325, 268, IRIS_BROWN)
    _save(img, "abertos_marrom", "olhos")

    img, d = _canvas()
    _draw_eye(d, 175, 268, IRIS_BLUE)
    _draw_eye(d, 325, 268, IRIS_BLUE)
    _save(img, "abertos_azul", "olhos")

    # estreitos
    img, d = _canvas()
    for cx in (175, 325):
        d.ellipse([cx - 30, cy := 268, cx + 30, cy + 10], fill=EYE_WHITE, outline=BLACK, width=1)
        d.ellipse([cx - 12, cy + 1, cx + 12, cy + 9], fill=IRIS_BROWN)
        d.ellipse([cx - 6, cy + 3, cx + 6, cy + 7], fill=PUPIL)
    _save(img, "estreitos_marrom", "olhos")

    # com óculos
    img, d = _canvas()
    _draw_eye(d, 175, 268, IRIS_BROWN)
    _draw_eye(d, 325, 268, IRIS_BROWN)
    d.rounded_rectangle([140, 252, 210, 286], radius=8, outline=BLACK, width=3)
    d.rounded_rectangle([290, 252, 360, 286], radius=8, outline=BLACK, width=3)
    d.line([(210, 269), (290, 269)], fill=BLACK, width=3)
    d.line([(140, 268), (100, 262)], fill=BLACK, width=3)
    d.line([(360, 268), (400, 262)], fill=BLACK, width=3)
    _save(img, "com_oculos", "olhos")


# ── SOBRANCELHAS ───────────────────────────────────────────────────────────

def _draw_eyebrow(d: ImageDraw.ImageDraw, cx: int, cy: int, color: tuple, thick: bool) -> None:
    bbox = [cx - 30, cy - 10, cx + 30, cy + 6]
    if thick:
        d.ellipse(bbox, fill=color)
    else:
        d.arc(bbox, start=205, end=335, fill=color, width=3)


def _gen_sobrancelhas() -> None:
    img, d = _canvas()
    _draw_eyebrow(d, 175, 237, BLACK, thick=False)
    _draw_eyebrow(d, 325, 237, BLACK, thick=False)
    _save(img, "finas_escuras", "sobrancelhas")

    img, d = _canvas()
    _draw_eyebrow(d, 175, 237, BLACK, thick=True)
    _draw_eyebrow(d, 325, 237, BLACK, thick=True)
    _save(img, "grossas_escuras", "sobrancelhas")

    img, d = _canvas()
    _draw_eyebrow(d, 175, 237, BROWN, thick=True)
    _draw_eyebrow(d, 325, 237, BROWN, thick=True)
    _save(img, "grossas_castanhas", "sobrancelhas")


# ── NARIZ ──────────────────────────────────────────────────────────────────

def _gen_nariz() -> None:
    img, d = _canvas()
    d.line([(250, 300), (236, 355)], fill=SKIN_SHADOW, width=3)
    d.line([(250, 300), (264, 355)], fill=SKIN_SHADOW, width=3)
    d.ellipse([228, 348, 246, 364], fill=SKIN_SHADOW)
    d.ellipse([254, 348, 272, 364], fill=SKIN_SHADOW)
    _save(img, "pequeno", "nariz")

    img, d = _canvas()
    d.line([(250, 295), (226, 355)], fill=SKIN_SHADOW, width=3)
    d.line([(250, 295), (274, 355)], fill=SKIN_SHADOW, width=3)
    d.ellipse([218, 348, 240, 368], fill=SKIN_SHADOW)
    d.ellipse([260, 348, 282, 368], fill=SKIN_SHADOW)
    _save(img, "largo", "nariz")

    img, d = _canvas()
    d.line([(250, 312), (240, 350)], fill=SKIN_SHADOW, width=3)
    d.line([(250, 312), (260, 350)], fill=SKIN_SHADOW, width=3)
    d.ellipse([232, 342, 252, 360], fill=SKIN_SHADOW)
    d.ellipse([248, 342, 268, 360], fill=SKIN_SHADOW)
    _save(img, "arrebitado", "nariz")


# ── BOCA ───────────────────────────────────────────────────────────────────

def _gen_boca() -> None:
    img, d = _canvas()
    d.polygon([(195, 398), (220, 390), (250, 393), (280, 390), (305, 398), (250, 403)], fill=LIP)
    d.ellipse([200, 400, 300, 424], fill=LIP)
    d.arc([200, 393, 300, 415], start=0, end=180, fill=LIP_DARK, width=2)
    _save(img, "neutra", "boca")

    img, d = _canvas()
    d.polygon([(195, 393), (220, 385), (250, 388), (280, 385), (305, 393), (250, 398)], fill=LIP)
    d.ellipse([200, 393, 300, 420], fill=LIP)
    d.arc([194, 388, 306, 428], start=0, end=180, fill=LIP_DARK, width=2)
    _save(img, "sorrindo", "boca")

    img, d = _canvas()
    d.arc([205, 393, 295, 412], start=0, end=180, fill=LIP, width=7)
    d.line([(205, 402), (295, 402)], fill=LIP_DARK, width=2)
    _save(img, "fina", "boca")

    img, d = _canvas()
    d.polygon([(190, 396), (215, 384), (250, 388), (285, 384), (310, 396), (250, 402)], fill=LIP)
    d.ellipse([193, 398, 307, 432], fill=LIP)
    d.arc([193, 390, 307, 422], start=0, end=180, fill=LIP_DARK, width=2)
    _save(img, "grossa", "boca")


# ── BARBA ──────────────────────────────────────────────────────────────────

def _gen_barba() -> None:
    img, _ = _canvas()
    _save(img, "sem_barba", "barba")

    img, d = _canvas()
    d.ellipse([150, 418, 350, 492], fill=(*BLACK[:3], 155))
    _save(img, "curta_escura", "barba")

    img, d = _canvas()
    d.ellipse([145, 415, 355, 492], fill=(*BLACK[:3], 180))
    d.polygon([(155, 455), (345, 455), (325, 535), (250, 555), (175, 535)], fill=(*BLACK[:3], 195))
    _save(img, "longa_escura", "barba")

    img, d = _canvas()
    d.polygon([(198, 422), (250, 410), (302, 422), (282, 435), (250, 430), (218, 435)], fill=(*BLACK[:3], 210))
    _save(img, "bigode", "barba")


# ── ENTRY POINT ────────────────────────────────────────────────────────────

def main() -> None:
    print("Gerando assets de faces...")
    _gen_rosto()
    _gen_cabelo()
    _gen_olhos()
    _gen_sobrancelhas()
    _gen_nariz()
    _gen_boca()
    _gen_barba()
    total = sum(1 for _ in FACES_DIR.rglob("*.png"))
    print(f"\n{total} PNGs criados em {FACES_DIR.resolve()}")


if __name__ == "__main__":
    main()
