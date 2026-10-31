"""Flat terrain (friction is a property of the body and is set by the generator)."""
from __future__ import annotations

import numpy as np


def generate_flat(shape, res, params, rng) -> np.ndarray:
    return np.zeros(shape, dtype=np.float64)
