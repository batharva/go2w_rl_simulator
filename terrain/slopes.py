"""Inclined / declined planar slopes with random slope angles and signs."""
from __future__ import annotations

import numpy as np

from terrain.common import axes

MAX_TOTAL_HEIGHT = 0.9


def generate_slopes(shape, res, params, rng) -> np.ndarray:
    X, Y = axes(shape, res)
    start = params.spawn_pad + 0.2
    ang = np.deg2rad(max(params.slope_angle * params.difficulty, 1.0))

    def ramp(d):
        return np.clip(np.maximum(d - start, 0.0) * np.tan(ang * rng.uniform(0.6, 1.0)), 0.0, MAX_TOTAL_HEIGHT)

    sx_pos, sx_neg = rng.choice([-1.0, 1.0], 2)
    hm = np.where(X > 0, sx_pos * ramp(np.abs(X)), sx_neg * ramp(np.abs(X)))
    if rng.random() < 0.5:                                     # add lateral tilt in y on some maps
        sy = rng.choice([-1.0, 1.0], 2)
        hm = hm + 0.5 * np.where(Y > 0, sy[0] * ramp(np.abs(Y)), sy[1] * ramp(np.abs(Y)))
    return np.clip(hm, -MAX_TOTAL_HEIGHT, MAX_TOTAL_HEIGHT)
