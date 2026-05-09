Projeto A3 de Computação Gráfica (UNA 2026.1).

**Objetivo:** Sistema em Python que monta um retrato falado via composição de PNGs com transparência e compara com base de rostos usando reconhecimento facial LBPH (OpenCV).

**Stack:** Python, OpenCV (cv2 + contrib), Pillow, NumPy, Matplotlib, Tkinter.

**Módulos principais:**
- `composer.py` — sobreposição de camadas PNG (Pillow)
- `preprocessor.py` — Haar Cascade, escala de cinza, normalização
- `database.py` — carrega AT&T Face Database
- `features.py` — treina/salva modelo LBPH
- `recognizer.py` — compara retrato com base, retorna top 5

**Assets:** `/faces/{rosto,cabelo,olhos,sobrancelhas,nariz,boca,barba}/` — PNGs 500×600 com canal alpha.

**Saída:** `output/retrato_falado_suspeito.png` | Modelo: `model/lbph_model.yml`

Leia o CLAUDE.md para detalhes completos antes de prosseguir.
