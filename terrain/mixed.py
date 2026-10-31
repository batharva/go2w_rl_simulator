"""Mixed terrain: several terrain types blended in angular sectors around the spawn pad."""
from __future__ import annotations

import copy

import numpy as np

from terrain.common import axes, smoothstep
from terrain.flat import generate_flat
from terrain.obstacles import generate_obstacles, generate_stepping
from terrain.slopes import generate_slopes
from terrain.stairs import generate_stairs
from terrain.uneven import generate_rough, generate_uneven

_POOL = {
    "uneven": generate_uneven, "rough": generate_rough, "stepping": generate_stepping,
    "stairs": generate_stairs, "slopes": generate_slopes, "obstacles": generate_obstacles,
}


def generate_mixed(shape, res, params, rng) -> np.ndarray:
    names = list(rng.permutation(list(_POOL))[:4])
    X, Y = axes(shape, res)
    theta = np.arctan2(Y, X)
    weights, maps = [], []
    for k, name in enumerate(names):
        centre = -np.pi + (k + 0.5) * np.pi / 2
        d = np.abs((theta - centre + np.pi) % (2 * np.pi) - np.pi)       # angular distance to sector centre
        weights.append(smoothstep(1.0 - (d - np.pi / 4) / 0.5))             # smooth edges, no height cliffs
        p = copy.copy(params)
        maps.append(_POOL[name](shape, res, p, rng))
    w = np.stack(weights)
    w = w / np.maximum(w.sum(0, keepdims=True), 1e-9)
    return (w * np.stack(maps)).sum(0)
