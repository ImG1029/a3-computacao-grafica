# A3 — Retrato Falado e Reconhecimento Facial

Projeto acadêmico (UC Computação Gráfica e Realidade Virtual — UNA 2026.1).

## Objetivo
Sistema que monta um retrato falado a partir de componentes faciais PNG e o compara com uma base de dados de rostos usando reconhecimento facial (LBPH/OpenCV).

## Stack
- Python 3.x
- OpenCV (`cv2`), Pillow (`PIL`), NumPy, Matplotlib
- Interface: Tkinter (ou CLI simples)

## Estrutura de Pastas
```
A3/
├── main.py                        # Entry point
├── composer.py                    # Composição de camadas (Etapa 1-2)
├── preprocessor.py                # Pré-processamento (Etapa 3)
├── database.py                    # Carregamento da base (Etapa 4)
├── features.py                    # Extração LBPH (Etapa 5)
├── recognizer.py                  # Reconhecimento + ranking (Etapa 6)
├── faces/                         # Assets PNG com fundo transparente
│   ├── rosto/
│   ├── cabelo/
│   ├── olhos/
│   ├── sobrancelhas/
│   ├── nariz/
│   ├── boca/
│   └── barba/
├── database/                      # Base de rostos (AT&T ou similar)
├── output/
│   └── retrato_falado_suspeito.png
└── model/
    └── lbph_model.yml             # Modelo treinado salvo
```

## Fluxo Principal
```
Interface (seleciona componentes)
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
pip install opencv-contrib-python Pillow numpy matplotlib
python main.py
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
