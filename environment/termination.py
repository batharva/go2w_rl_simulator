"""Episode termination (failure) and truncation (time / arena) logic."""
from __future__ import annotations

from typing import Any, Dict, Tuple


def check_termination(cfg: Dict[str, Any], *, roll: float, pitch: float, height_above_feet: float,
                      base_contact: bool) -> Tuple[bool, str]:
    """Failure conditions -> (terminated, reason).  A termination counts as a *fall*."""
    t = cfg["termination"]
    if base_contact:
        return True, "base_contact"
    if abs(roll) > t["max_roll"]:
        return True, "roll"
    if abs(pitch) > t["max_pitch"]:
        return True, "pitch"
    if height_above_feet < t["min_height_above_feet"]:
        return True, "collapsed"
    return False, ""


def check_truncation(cfg: Dict[str, Any], *, step: int, max_steps: int, x: float, y: float,
                     half_size: float) -> Tuple[bool, str]:
    """Non-failure ends: time limit or leaving the generated arena."""
    if step >= max_steps:
        return True, "time_limit"
    if max(abs(x), abs(y)) > half_size - cfg["termination"]["arena_margin"]:
        return True, "arena_border"
    return False, ""
