from pathlib import Path

DATABASE_DIR = Path(__file__).parent / "database"


def _bootstrap() -> None:
    for d in ("output", "model", "database"):
        Path(d).mkdir(exist_ok=True)

    faces_dir = Path("faces")
    if not any(faces_dir.rglob("*.png")):
        print("Assets não encontrados. Gerando placeholders...")
        import generate_assets
        generate_assets.main()

    subjects = [p for p in DATABASE_DIR.iterdir() if p.is_dir()]
    if not subjects:
        print(
            "\n[AVISO] Base de dados vazia.\n"
            "Para usar o reconhecimento facial:\n"
            "  1. Baixe a AT&T Face Database:\n"
            "     https://www.kaggle.com/datasets/kasikrit/att-database-of-faces\n"
            "  2. Extraia as pastas s1…s40 dentro de database/\n"
            "  3. Reinicie o sistema\n"
        )


if __name__ == "__main__":
    _bootstrap()
    from interface import FaceComposerApp
    FaceComposerApp().mainloop()
