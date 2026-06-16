#!/usr/bin/env python3
"""
compute_groups.py — Agrupa as faces por compatibilidade de pose/proporção.

O alinhamento da extração (extract_strips._align) fixa apenas a linha dos olhos
e a escala; nariz, boca, queixo e a largura do rosto continuam variando muito
entre as faces (40–80px). Misturar componentes de faces com proporções
incompatíveis deixa o retrato desencaixado.

Este script mede, para cada foto-fonte já alinhada, uma assinatura de
proporção e agrupa as faces em N grupos (k-means determinístico). O resultado
é gravado em faces/groups.json e usado pela interface para só permitir misturar
componentes de faces do mesmo grupo.

Uso:
  python compute_groups.py            # 3 grupos (padrão)
  python compute_groups.py --groups 4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

import extract_strips as es

ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "source_faces"
GROUPS_PATH = ROOT / "faces" / "groups.json"

# Rótulos por nº de grupos, ordenados do rosto mais curto para o mais longo.
_LABELS = {
    2: ["Rosto curto", "Rosto longo"],
    3: ["Rosto curto", "Rosto médio", "Rosto longo"],
    4: ["Rosto curto", "Rosto médio-curto", "Rosto médio-longo", "Rosto longo"],
}


def _signature(path: Path, detector) -> np.ndarray | None:
    """Vetor de proporção (após alinhar pelos olhos): nariz, boca, queixo, largura."""
    import mediapipe as mp

    bgr = cv2.imread(str(path))
    if bgr is None:
        return None
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_img)
    if not result.face_landmarks:
        return None
    aligned, lms = es._align(bgr, result.face_landmarks[0])
    if aligned is None:
        return None
    oval = lms[es.FACE_OVAL]
    return np.array([
        float(lms[es.NOSE_IDX][:, 1].max()),                  # base do nariz (y)
        float(lms[es.MOUTH_IDX][:, 1].mean()),                # boca (y)
        float(lms[es.LM_CHIN][1]),                            # queixo (y)
        float(oval[:, 0].max() - oval[:, 0].min()),           # largura do rosto
    ], dtype=np.float64)


def _kmeans(X: np.ndarray, k: int, iters: int = 50) -> np.ndarray:
    """K-means determinístico (init por ponto mais distante). Retorna labels."""
    # init: ponto de menor soma padronizada, depois os mais distantes (k-means++ fixo)
    idx = [int(X.sum(axis=1).argmin())]
    while len(idx) < k:
        d = np.min([np.linalg.norm(X - X[i], axis=1) for i in idx], axis=0)
        idx.append(int(d.argmax()))
    centroids = X[idx].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new = dists.argmin(axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for c in range(k):
            pts = X[labels == c]
            if len(pts):
                centroids[c] = pts.mean(axis=0)
    return labels


def compute(n_groups: int = 3) -> dict:
    photos = sorted(p for p in SOURCE_DIR.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not photos:
        raise FileNotFoundError(f"Nenhuma foto em {SOURCE_DIR}/")

    detector = es._build_detector()
    sigs: dict[str, np.ndarray] = {}
    try:
        for p in photos:
            s = _signature(p, detector)
            if s is not None:
                sigs[p.stem] = s
            else:
                print(f"  [AVISO] sem detecção em {p.name}")
    finally:
        detector.close()

    stems = list(sigs)
    X = np.array([sigs[s] for s in stems])
    # padroniza para que largura e posições verticais pesem igual
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    raw = _kmeans(Xz, min(n_groups, len(stems)))

    # reordena os ids por comprimento médio do rosto (queixo) → curto..longo
    order = sorted(set(raw), key=lambda c: X[raw == c][:, 2].mean())
    remap = {old: new for new, old in enumerate(order)}
    labels = _LABELS.get(n_groups, [f"Grupo {i}" for i in range(n_groups)])

    faces = {s: remap[int(raw[i])] for i, s in enumerate(stems)}
    groups = {str(i): labels[i] for i in range(len(order))}
    return {"groups": groups, "faces": faces}


def main() -> None:
    ap = argparse.ArgumentParser(description="Agrupa faces por pose/proporção.")
    ap.add_argument("--groups", type=int, default=3, help="número de grupos (padrão 3)")
    args = ap.parse_args()

    try:
        import mediapipe  # noqa: F401
    except ImportError:
        raise SystemExit("[ERRO] mediapipe não instalado. Execute: pip install mediapipe")

    data = compute(args.groups)
    GROUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUPS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(data['faces'])} faces agrupadas em {len(data['groups'])} grupos → {GROUPS_PATH}")
    for gid, label in data["groups"].items():
        members = sorted(s for s, g in data["faces"].items() if g == int(gid))
        print(f"  [{gid}] {label}: {', '.join(members)}")


if __name__ == "__main__":
    main()
