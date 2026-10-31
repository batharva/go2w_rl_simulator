"""Helpers shared by all terrain generators. Height maps are indexed ``hm[ix, iy]`` (x then y), in metres."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def axes(shape: Tuple[int, int], res: float) -> Tuple[np.ndarray, np.ndarray]:
    """World x / y coordinate (centre of the map = (0, 0)) of every cell, as 2-D arrays."""
    nx, ny = shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * res
    y = (np.arange(ny) - (ny - 1) / 2.0) * res
    return np.meshgrid(x, y, indexing="ij")


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bilinear_resize(coarse: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Bilinear up-sampling of a coarse random grid to ``shape`` (no scipy needed)."""
    cx, cy = coarse.shape
    gx = np.linspace(0, cx - 1, shape[0])
    gy = np.linspace(0, cy - 1, shape[1])
    x0 = np.clip(np.floor(gx).astype(int), 0, cx - 2)
    y0 = np.clip(np.floor(gy).astype(int), 0, cy - 2)
    fx = (gx - x0)[:, None]
    fy = (gy - y0)[None, :]
    a = coarse[np.ix_(x0, y0)]
    b = coarse[np.ix_(x0 + 1, y0)]
    c = coarse[np.ix_(x0, y0 + 1)]
    d = coarse[np.ix_(x0 + 1, y0 + 1)]
    return a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy


def smooth_noise(rng: np.random.Generator, shape: Tuple[int, int], res: float, cell: float) -> np.ndarray:
    """Spatially-correlated random field in [0, 1] with feature size ~``cell`` metres."""
    cx = int(np.ceil(shape[0] * res / cell)) + 2
    cy = int(np.ceil(shape[1] * res / cell)) + 2
    return bilinear_resize(rng.random((cx, cy)), shape)


def pad_mask(shape: Tuple[int, int], res: float, pad: float, blend: float = 0.8) -> np.ndarray:
    """0 inside the flat spawn pad, smoothly rising to 1 at ``pad + blend`` metres from the centre."""
    X, Y = axes(shape, res)
    r = np.sqrt(X ** 2 + Y ** 2)
    return smoothstep((r - pad) / max(blend, 1e-6))
