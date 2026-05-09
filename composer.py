from pathlib import Path
from PIL import Image

CANVAS_SIZE = (500, 600)
FACES_DIR = Path(__file__).parent / "faces"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_PATH = OUTPUT_DIR / "retrato_falado_suspeito.png"

LAYER_ORDER = ["cabelo", "rosto", "sobrancelhas", "olhos", "nariz", "boca", "barba"]

CATEGORY_LABELS = {
    "cabelo": "Cabelo",
    "rosto": "Rosto",
    "sobrancelhas": "Sobrancelhas",
    "olhos": "Olhos",
    "nariz": "Nariz",
    "boca": "Boca",
    "barba": "Barba",
}


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
    for layer in LAYER_ORDER:
        path = selection.get(layer)
        if not path:
            continue
        component = Image.open(path).convert("RGBA")
        if component.size != CANVAS_SIZE:
            component = component.resize(CANVAS_SIZE, Image.LANCZOS)
        canvas = Image.alpha_composite(canvas, component)
    return canvas


def save(image: Image.Image, path: Path = OUTPUT_PATH) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return path
