import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from PIL import Image, ImageTk

import composer
import features
import recognizer as rec
from recognizer import Match

PREVIEW_W, PREVIEW_H = 350, 420
DATABASE_DIR = Path(__file__).parent / "database"
SUPPORTED_IMG = {".pgm", ".jpg", ".jpeg", ".png", ".bmp"}


class FaceComposerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Retrato Falado — Sistema A3")
        self.resizable(False, False)
        self._vars: dict[str, tk.StringVar] = {}
        self._choices: dict[str, list[tuple[str, Path | None]]] = {}
        self._preview_ref: ImageTk.PhotoImage | None = None
        self._buttons: list[ttk.Button] = []
        self._build_menu()
        self._build_ui()
        self._refresh_preview()

    # ── build ─────────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)
        model_menu = tk.Menu(menubar, tearoff=0)
        model_menu.add_command(label="Treinar modelo", command=self._train)
        model_menu.add_command(label="Re-treinar (forçar)", command=self._force_retrain)
        menubar.add_cascade(label="Modelo", menu=model_menu)
        self.config(menu=menubar)

    def _build_ui(self) -> None:
        self.configure(bg="#ececec")
        main = ttk.Frame(self, padding=14)
        main.grid(row=0, column=0)

        ttk.Label(main, text="Retrato Falado", font=("Helvetica", 15, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 14)
        )

        left = ttk.LabelFrame(main, text="Componentes", padding=10)
        left.grid(row=1, column=0, padx=(0, 12), sticky="ns")

        for row, category in enumerate(composer.LAYER_ORDER):
            label = composer.CATEGORY_LABELS[category]
            options: list[tuple[str, Path | None]] = [("Nenhum", None)]
            options += composer.list_components(category)
            self._choices[category] = options

            default = options[1][0] if len(options) > 1 else "Nenhum"
            var = tk.StringVar(value=default)
            self._vars[category] = var
            var.trace_add("write", lambda *_, c=category: self._refresh_preview())

            ttk.Label(left, text=f"{label}:", width=13, anchor="w").grid(
                row=row, column=0, sticky="w", pady=4
            )
            ttk.Combobox(
                left,
                textvariable=var,
                values=[name for name, _ in options],
                state="readonly",
                width=22,
            ).grid(row=row, column=1, sticky="w", pady=4)

        right = ttk.LabelFrame(main, text="Preview", padding=10)
        right.grid(row=1, column=1, sticky="nsew")

        self._canvas = tk.Canvas(
            right, width=PREVIEW_W, height=PREVIEW_H,
            bg="white", bd=0, highlightthickness=0,
        )
        self._canvas.grid(row=0, column=0)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(14, 0))

        btn_save = ttk.Button(btn_frame, text="Salvar Retrato", command=self._save)
        btn_save.pack(side="left", ipadx=10, ipady=4, padx=(0, 8))

        btn_rec = ttk.Button(btn_frame, text="Reconhecer Suspeito", command=self._recognize)
        btn_rec.pack(side="left", ipadx=10, ipady=4)

        self._buttons = [btn_save, btn_rec]

        self._status_var = tk.StringVar(value="Pronto")
        ttk.Label(main, textvariable=self._status_var, foreground="#555").grid(
            row=3, column=0, columnspan=2, pady=(6, 0)
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _selection(self) -> dict[str, Path | None]:
        return {
            cat: next((p for n, p in opts if n == self._vars[cat].get()), None)
            for cat, opts in self._choices.items()
        }

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.update_idletasks()

    def _set_buttons(self, state: str) -> None:
        for btn in self._buttons:
            btn.configure(state=state)

    def _has_database(self) -> bool:
        return DATABASE_DIR.exists() and any(p.is_dir() for p in DATABASE_DIR.iterdir())

    def _load_subject_image(self, subject: str) -> Image.Image | None:
        subject_dir = DATABASE_DIR / subject
        if not subject_dir.exists():
            return None
        files = sorted(f for f in subject_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMG)
        return Image.open(files[0]).convert("RGB") if files else None

    # ── actions ───────────────────────────────────────────────────────────

    def _refresh_preview(self) -> None:
        image = composer.compose(self._selection())
        preview = image.resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        self._preview_ref = ImageTk.PhotoImage(preview)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._preview_ref)

    def _save(self) -> None:
        image = composer.compose(self._selection())
        path = composer.save(image)
        self._status(f"Salvo: {path.name}")
        messagebox.showinfo("Salvo", f"Retrato salvo em:\n{path}")

    def _train(self, force: bool = False) -> None:
        if not self._has_database():
            messagebox.showwarning(
                "Base vazia",
                "Nenhum sujeito encontrado em database/.\n\n"
                "Baixe a AT&T Face Database e extraia as pastas s1…s40 dentro de database/.",
            )
            return
        if not force and features.is_trained():
            messagebox.showinfo("Modelo", "Modelo já treinado.\nUse 'Re-treinar' para forçar.")
            return
        self._run_in_thread(
            task=rec.train_model,
            status_running="Treinando modelo LBPH...",
            on_done=lambda _: self._status("Modelo treinado com sucesso"),
        )

    def _force_retrain(self) -> None:
        self._train(force=True)

    def _recognize(self) -> None:
        if not self._has_database():
            messagebox.showwarning(
                "Base vazia",
                "Nenhum sujeito encontrado em database/.\n\n"
                "Baixe a AT&T Face Database e extraia as pastas s1…s40 dentro de database/.",
            )
            return

        selection = self._selection()
        image = composer.compose(selection)

        def task() -> tuple[Image.Image, list[Match]]:
            if not features.is_trained():
                self.after(0, lambda: self._status("Treinando modelo LBPH..."))
                rec.train_model()
            self.after(0, lambda: self._status("Reconhecendo..."))
            composer.save(image)
            matches = rec.recognize_selection(selection, top_n=5)
            return image, matches

        def on_done(result: tuple[Image.Image, list[Match]]) -> None:
            img, matches = result
            self._status(f"Concluído — melhor: {matches[0].subject} (dist {matches[0].distance})")
            self._show_results(img, matches)

        self._run_in_thread(task=task, status_running="Iniciando reconhecimento...", on_done=on_done)

    def _run_in_thread(self, task, status_running: str, on_done) -> None:
        self._set_buttons("disabled")
        self._status(status_running)

        def worker() -> None:
            try:
                result = task()
                self.after(0, lambda: on_done(result))
            except Exception as exc:
                self.after(0, lambda e=exc: self._on_error(e))
            finally:
                self.after(0, lambda: self._set_buttons("normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_error(self, exc: Exception) -> None:
        self._status("Erro")
        messagebox.showerror("Erro", str(exc))

    def _show_results(self, retrato: Image.Image, matches: list[Match]) -> None:
        import matplotlib.pyplot as plt

        n = len(matches)
        fig, axes = plt.subplots(1, n + 1, figsize=(3 * (n + 1), 4))
        fig.suptitle("Resultado do Reconhecimento Facial — Top 5", fontsize=13, fontweight="bold")

        axes[0].imshow(retrato)
        axes[0].set_title("Retrato\nFalado", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        for ax, match in zip(axes[1:], matches):
            face_img = self._load_subject_image(match.subject)
            if face_img is not None:
                ax.imshow(face_img, cmap="gray")
            else:
                ax.text(0.5, 0.5, "?", ha="center", va="center", fontsize=32, transform=ax.transAxes)
                ax.set_facecolor("#ddd")

            color = "#2a7a2a" if match.rank == 1 else "#333"
            ax.set_title(
                f"#{match.rank}  {match.subject}\n"
                f"Dist: {match.distance:.1f}   Sim: {match.similarity:.1%}",
                fontsize=9,
                color=color,
            )
            ax.axis("off")

        plt.tight_layout()
        plt.show(block=False)
