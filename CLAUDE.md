# A3 — Retrato Falado e Reconhecimento Facial

Projeto acadêmico (UC Computação Gráfica e Realidade Virtual — UNA 2026.1).

## Objetivo
Sistema que monta um retrato falado a partir de componentes faciais PNG e o compara com uma base de dados de rostos usando reconhecimento facial (LBPH/OpenCV).

## Stack
- Python 3.x
- OpenCV (`cv2`), Pillow (`PIL`), NumPy
- Interface: Flask (web)

## Estrutura de Pastas
```
A3/
├── app.py                         # Entry point (Flask)
├── composer.py                    # Composição de camadas (Etapa 1-2)
├── preprocessor.py                # Pré-processamento (Etapa 3)
├── database.py                    # Carregamento da base (Etapa 4)
├── features.py                    # Extração LBPH (Etapa 5)
├── recognizer.py                  # Reconhecimento + ranking (Etapa 6)
├── templates/                     # HTML do front-end Flask
├── static/                        # CSS/JS do front-end
├── faces/                         # Assets PNG com fundo transparente
│   ├── cabelo/
│   ├── sobrancelhas/
│   ├── olhos/
│   ├── nariz/
│   ├── boca/
│   ├── queixo/
│   └── groups.json                # Grupos de compatibilidade pose/proporção
├── database/                      # Base de rostos (face_NN/)
├── output/
│   └── retrato_falado_suspeito.png
└── model/
    ├── lbph_model.yml             # Modelo treinado salvo
    └── training_data.npz          # Dados para ranking top-N
```

## Fluxo Principal
```
Interface web (seleciona componentes)
  → composer.py    → output/retrato_falado_suspeito.png
  → preprocessor.py → face recortada + normalizada
  → features.py    → vetor de características (LBPH)
  → recognizer.py  → top 5 matches com distância
```

## Etapas de Desenvolvimento
1. Estrutura + assets PNG
2. Compositor de camadas (Pillow)
3. Interface de seleção (Tkinter)
4. Geração da imagem final
5. Pré-processamento (Haar Cascade)
6. Base de dados (AT&T Face Database)
7. Extração LBPH + treinamento
8. Reconhecimento + ranking top 5
9. Integração e testes
10. Relatório técnico

## Como Rodar
```bash
pip install -r requirements.txt
python app.py   # abre em http://localhost:5000
```

## Entregáveis
- Código fonte, interface funcional, base de dados, relatório técnico, resumo ExpoUna

## Critérios de Avaliação
| Critério | Peso |
|---|---|
| Modelagem gráfica do retrato | 25% |
| Reconhecimento facial | 25% |
| Qualidade do código | 15% |
| Relatório técnico | 15% |
| Apresentação | 10% |
| ExpoUna | 10% |
