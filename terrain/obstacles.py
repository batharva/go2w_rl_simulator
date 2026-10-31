"""Block / barrier obstacles and randomly placed stepping platforms."""
from __future__ import annotations

import numpy as np


def generate_obstacles(shape, res, params, rng) -> np.ndarray:
    """Randomly positioned small blocks (70 %) and low barriers (30 %)."""
    nx, ny = shape
    hm = np.zeros(shape)
    area = nx * ny * res ** 2
    count = int(params.obstacle_density * area * (0.15 + 0.85 * params.difficulty) * 0.35)
    h_hi = max(params.max_height, 0.02) * (0.3 + 0.7 * params.difficulty)
    for _ in range(count):
        h = rng.uniform(0.02, max(h_hi, 0.03))
        if rng.random() < 0.7:
            a = max(int(params.obstacle_size * rng.uniform(0.5, 1.4) / res), 1)
            b = max(int(params.obstacle_size * rng.uniform(0.5, 1.4) / res), 1)
        else:                                                   # low barrier
            a, b = max(int(rng.uniform(0.8, 1.6) / res), 2), max(int(rng.uniform(0.05, 0.12) / res), 1)
            if rng.random() < 0.5:
                a, b = b, a
        i, j = rng.integers(0, max(nx - a, 1)), rng.integers(0, max(ny - b, 1))
        hm[i:i + a, j:j + b] = np.maximum(hm[i:i + a, j:j + b], h)
    return hm


def generate_stepping(shape, res, params, rng) -> np.ndarray:
    """Platforms of random size/height placed on a jittered grid with random spacing."""
    nx, ny = shape
    hm = np.zeros(shape)
    h_hi = max(params.max_height, 0.03) * (0.3 + 0.7 * params.difficulty)
    spacing = rng.uniform(0.7, 1.3)
    xs = np.arange(0, nx * res, spacing)
    ys = np.arange(0, ny * res, spacing)
    for x in xs:
        for y in ys:
            if rng.random() < 0.15:                             # leave random gaps in the layout
                continue
            sx, sy = rng.uniform(0.35, 0.9, 2)
            cx, cy = x + rng.uniform(-0.2, 0.2), y + rng.uniform(-0.2, 0.2)
            i0, i1 = int(max(cx - sx / 2, 0) / res), int(min(cx + sx / 2, nx * res) / res)
            j0, j1 = int(max(cy - sy / 2, 0) / res), int(min(cy + sy / 2, ny * res) / res)
            hm[i0:i1, j0:j1] = np.maximum(hm[i0:i1, j0:j1], rng.uniform(0.02, h_hi))
    return hm
