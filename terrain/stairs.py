"""Ascending / descending staircases with random riser heights and tread widths."""
from __future__ import annotations

import numpy as np

from terrain.common import axes

MAX_TOTAL_HEIGHT = 0.7   # m, total rise/descent of one staircase


def _profile(dist: np.ndarray, start: float, params, rng, sign: float) -> np.ndarray:
    """Height as function of distance ``dist`` (>0) from the centre along one direction."""
    h_max = np.interp(params.difficulty, [0, 1], [0.03, max(params.stair_height, 0.04)])
    out = np.zeros_like(dist)
    x, h = start, 0.0
    for _ in range(60):
        w = params.stair_width * rng.uniform(0.8, 1.25)
        step = h_max * rng.uniform(0.75, 1.25)
        if abs(h + step) > MAX_TOTAL_HEIGHT:
            h_final = h
            out[dist >= x] = sign * h_final
            return out
        h += step
        out[dist >= x] = sign * h
        x += w
        if x > dist.max():
            break
    return out


def generate_stairs(shape, res, params, rng) -> np.ndarray:
    """Staircases around the spawn pad.

    mode "both": ascending towards +x, descending towards -x;  "up": ascending on both sides;
    "down": descending on both sides.  The map is rotated by a random multiple of 90 degrees so stairs are
    met from every direction.
    """
    X, _ = axes(shape, res)
    start = params.spawn_pad + 0.3
    mode = str(rng.choice(["both", "up", "down"]))
    sign_pos = -1.0 if mode == "down" else 1.0
    sign_neg = 1.0 if mode == "up" else -1.0
    hm = np.where(X > 0, _profile(np.abs(X), start, params, rng, sign_pos),
                  _profile(np.abs(X), start, params, rng, sign_neg))
    return np.rot90(hm, int(rng.integers(0, 4))).copy()
