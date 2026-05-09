Fluxo completo do sistema A3 (referência rápida para contexto):

```
[Interface Tkinter]
  Usuário seleciona: rosto + cabelo + olhos + sobrancelhas + nariz + boca + barba
        ↓
[composer.py] — Pillow, sobreposição canal alpha, 500×600px
        ↓
  output/retrato_falado_suspeito.png
        ↓
[preprocessor.py] — cv2 grayscale → Haar Cascade detect → crop → resize 200×200
        ↓
[features.py / recognizer.py] — LBPHFaceRecognizer (model/lbph_model.yml)
        ↓
  Comparação com database/ (AT&T Face Database: 40 pessoas × 10 fotos)
        ↓
  Top 5 matches: ID + distância LBPH (menor = mais similar)
        ↓
[Matplotlib] — exibe retrato + top 5 faces com scores
```

Arquivos-chave: `composer.py`, `preprocessor.py`, `database.py`, `features.py`, `recognizer.py`, `main.py`.
