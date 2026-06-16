"""Flask web application — Sistema de Retrato Falado."""
from __future__ import annotations
import base64
import io
import json
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

import composer
import features
import recognizer as rec

app = Flask(__name__)

SUPPORTED_IMG = {".pgm", ".jpg", ".jpeg", ".png", ".bmp"}
DATABASE_DIR = Path(__file__).parent / "database"
GROUPS_PATH = Path(__file__).parent / "faces" / "groups.json"


def _load_groups() -> dict:
    """Mapa de compatibilidade face→grupo (faces/groups.json).

    Gerado por compute_groups.py. Se ausente, devolve vazio e a interface
    simplesmente não aplica o filtro de compatibilidade.
    """
    try:
        return json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"groups": {}, "faces": {}}

_train_lock = threading.Lock()
_train_state: dict = {"status": "idle", "message": ""}


# ── helpers ───────────────────────────────────────────────────────────────

def _selection_from_body(data: dict) -> dict[str, Path | None]:
    selection: dict[str, Path | None] = {}
    for category in composer.LAYER_ORDER:
        name = data.get(category)
        if name:
            for opt_name, path in composer.list_components(category):
                if opt_name == name:
                    selection[category] = path
                    break
            else:
                selection[category] = None
        else:
            selection[category] = None
    return selection


def _img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, format="JPEG", quality=90)
    else:
        img.save(buf, format="PNG")
    buf.seek(0)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.read()).decode()}"


def _subject_preview(subject: str) -> str | None:
    subject_dir = DATABASE_DIR / subject
    if not subject_dir.exists():
        return None
    files = sorted(f for f in subject_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMG)
    if not files:
        return None
    img = Image.open(files[0]).convert("RGB")
    img.thumbnail((200, 200))
    return _img_to_b64(img, "JPEG")


# ── routes ────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/components")
def get_components():
    groups_data = _load_groups()
    face_groups = groups_data.get("faces", {})
    categories = {}
    for category in composer.LAYER_ORDER:
        options = composer.list_components(category)
        categories[category] = {
            "label": composer.CATEGORY_LABELS[category],
            "options": [
                {
                    "name": name,
                    "url": f"/api/component-image/{category}/{name}",
                    "group": face_groups.get(path.stem),
                }
                for name, path in options
            ],
        }
    return jsonify({"categories": categories, "groups": groups_data.get("groups", {})})


@app.get("/api/component-image/<category>/<name>")
def component_image(category: str, name: str):
    for opt_name, path in composer.list_components(category):
        if opt_name == name and path:
            return send_file(path, mimetype="image/png")
    return "", 404


@app.post("/api/compose")
def api_compose():
    selection = _selection_from_body(request.json or {})
    image = composer.compose(selection)
    return jsonify({"image": _img_to_b64(image)})


@app.post("/api/recognize")
def api_recognize():
    if not features.is_trained():
        return jsonify({"error": "Modelo não treinado. Clique em 'Treinar Modelo' primeiro."}), 400
    selection = _selection_from_body(request.json or {})
    image = composer.compose(selection)
    composer.save(image)
    try:
        matches = rec.recognize_selection(selection, top_n=5)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    return jsonify({
        "matches": [
            {
                "rank": m.rank,
                "subject": m.subject,
                "display_name": m.subject.replace("_", " ").title(),
                "distance": round(float(m.distance), 2),
                "similarity": round(float(m.similarity) * 100, 1),
                "face_image": _subject_preview(m.subject),
            }
            for m in matches
        ]
    })


@app.post("/api/train")
def api_train():
    global _train_state
    with _train_lock:
        if _train_state["status"] == "running":
            return jsonify({"error": "Treinamento já em andamento"}), 409
        _train_state = {"status": "running", "message": "Carregando base de dados..."}

    def _worker():
        global _train_state
        try:
            rec.train_model()
            with _train_lock:
                _train_state = {"status": "done", "message": "Modelo treinado com sucesso!"}
        except Exception as exc:
            with _train_lock:
                _train_state = {"status": "error", "message": str(exc)}

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"status": "started"})


@app.get("/api/train/status")
def api_train_status():
    with _train_lock:
        return jsonify(_train_state.copy())


@app.get("/api/model-info")
def api_model_info():
    trained = features.is_trained()
    subjects = []
    total = 0
    if DATABASE_DIR.exists():
        for d in sorted(DATABASE_DIR.iterdir()):
            if d.is_dir():
                imgs = [f for f in d.iterdir() if f.suffix.lower() in SUPPORTED_IMG]
                if imgs:
                    subjects.append(d.name.replace("_", " ").title())
                    total += len(imgs)
    return jsonify({
        "trained": trained,
        "subject_count": len(subjects),
        "image_count": total,
        "subjects": subjects,
    })


# ── bootstrap ─────────────────────────────────────────────────────────────

def _bootstrap() -> None:
    for d in ("output", "model", "database"):
        Path(d).mkdir(exist_ok=True)
    faces_dir = Path("faces")
    if not any(faces_dir.rglob("*.png")):
        import generate_assets
        generate_assets.main()


if __name__ == "__main__":
    _bootstrap()
    app.run(debug=True, port=5000, use_reloader=False)
