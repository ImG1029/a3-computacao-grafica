import numpy as np


def lbp_histogram(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    center = gray[1 : h - 1, 1 : w - 1].astype(np.int16)
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)

    for bit, (dr, dc) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        neighbor = gray[1 + dr : h - 1 + dr, 1 + dc : w - 1 + dc].astype(np.int16)
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    return hist.astype(np.float32)
