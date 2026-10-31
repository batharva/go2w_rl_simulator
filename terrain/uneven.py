"""Random uneven (smooth rolling) terrain and rough (fine-grained, bumpy) terrain."""
from __future__ import annotations

import numpy as np

from terrain.common import smooth_noise


def _range(params):
    """Height band [lo, hi] scaled by difficulty."""
    lo = params.min_height * params.difficulty
    hi = params.max_height * params.difficulty
    return min(lo, hi), max(lo, hi)


def generate_uneven(shape, res, params, rng) -> np.ndarray:
    """Smooth hills: two octaves of correlated noise (0.5 m and 1.5 m features)."""
    lo, hi = _range(params)
    n = 0.65 * smooth_noise(rng, shape, res, 0.5) + 0.35 * smooth_noise(rng, shape, res, 1.5)
    n = (n - n.min()) / max(np.ptp(n), 1e-9)
    return lo + (hi - lo) * (0.4 + 0.6 * params.roughness) * n + (hi - lo) * 0.0


def generate_rough(shape, res, params, rng) -> np.ndarray:
    """Fine random heights (cell-scale) + small scattered bumps."""
    lo, hi = _range(params)
    amp = max(hi - lo, 0.0)
    fine = smooth_noise(rng, shape, res, max(res * 1.5, 0.10))
    mid = smooth_noise(rng, shape, res, 0.4)
    hm = amp * params.roughness * (0.6 * fine + 0.4 * mid)
    nb = int(params.obstacle_density * 30 * (0.3 + params.difficulty))
    nx, ny = shape
    for _ in range(nb):                                        # small bumps (3-6 cells wide)
        w = rng.integers(2, 5)
        i, j = rng.integers(0, nx - w), rng.integers(0, ny - w)
        hm[i:i + w, j:j + w] += rng.uniform(0.2, 1.0) * amp * 0.5
    return hm + lo
