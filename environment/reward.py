"""Reward function: a weighted sum of separately logged components.

Every component is computed as a non-negative *raw* value; the (signed) weight in ``reward.weights`` decides whether it is
a reward (>0) or a penalty (<0).  The per-step total is the sum of ``weight * raw``.  All components are returned so they
can be logged / plotted individually.

Anti reward-hacking measures
----------------------------
* velocity tracking uses an exponential kernel (bounded, no incentive to overshoot);
* ``stall`` penalises standing still when a non-zero command is given (projected speed shortfall);
* ``feet_air_time`` only pays out for touchdown events after a real swing phase, and only while walking, so shuffling or
  dragging the feet earns nothing;
* ``base_height`` is measured relative to the *feet* (not to a fixed z), so crawling on the belly is penalised and the
  term is valid on slopes / stairs;
* ``body_contact`` / termination catch belly dragging, ``foot_slip`` catches sliding.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

COMPONENTS = ["track_vx", "track_vy", "track_wz", "alive", "foot_contact", "feet_air_time", "orientation", "base_height",
              "lin_vel_z", "ang_vel_xy", "joint_vel", "joint_acc", "action_rate", "energy", "foot_slip",
              "default_pose", "stall", "body_contact", "self_collision", "fall"]


class RewardCalculator:
    def __init__(self, cfg: Dict[str, Any], default_q: np.ndarray, target_height: float, dt: float):
        r = cfg["reward"]
        self.weights: Dict[str, float] = {k: float(v) for k, v in r["weights"].items()}
        self.sigma_v = float(r["sigma_velocity"])
        self.sigma_w = float(r["sigma_angular"])
        self.air_thr = float(r["air_time_threshold"])
        self.deadband = float(cfg["commands"]["deadband"])
        self.default_q = default_q
        self.target_height = float(r["target_base_height"]) or target_height
        self.dt = dt
        self.reset()

    def set_weights(self, w: Dict[str, float]) -> None:
        for k, v in w.items():
            if k not in COMPONENTS:
                raise KeyError(f"Unknown reward component '{k}'")
            self.weights[k] = float(v)

    def reset(self) -> None:
        self.air_time = np.zeros(4)
        self.prev_qd = np.zeros(12)

    def compute(self, *, cmd: np.ndarray, lin_vel_b: np.ndarray, ang_vel_b: np.ndarray, gravity_b: np.ndarray,
                height_above_feet: float, q: np.ndarray, qd: np.ndarray, action: np.ndarray, prev_action: np.ndarray,
                power: np.ndarray, foot_contact: np.ndarray, foot_vel: np.ndarray, body_contacts: int,
                self_collisions: int, fell: bool) -> Tuple[float, Dict[str, float]]:
        w = self.weights
        moving_cmd = float(np.linalg.norm(cmd[:2])) > self.deadband or abs(cmd[2]) > self.deadband
        raw: Dict[str, float] = {}
        raw["track_vx"] = float(np.exp(-(lin_vel_b[0] - cmd[0]) ** 2 / self.sigma_v))
        raw["track_vy"] = float(np.exp(-(lin_vel_b[1] - cmd[1]) ** 2 / self.sigma_v))
        raw["track_wz"] = float(np.exp(-(ang_vel_b[2] - cmd[2]) ** 2 / self.sigma_w))
        raw["alive"] = 1.0
        n_c = int(foot_contact.sum())
        raw["foot_contact"] = float(n_c >= (2 if moving_cmd else 3))

        # swing-phase reward, paid on touchdown only, only when commanded to walk
        first_contact = (self.air_time > 0.0) & foot_contact
        self.air_time += self.dt
        air = float(np.sum((self.air_time - self.air_thr) * first_contact)) if moving_cmd else 0.0
        raw["feet_air_time"] = max(air, 0.0)
        self.air_time *= ~foot_contact

        raw["orientation"] = float(np.sum(gravity_b[:2] ** 2))
        raw["base_height"] = float((height_above_feet - self.target_height) ** 2)
        raw["lin_vel_z"] = float(lin_vel_b[2] ** 2)
        raw["ang_vel_xy"] = float(np.sum(ang_vel_b[:2] ** 2))
        raw["joint_vel"] = float(np.sum(qd ** 2))
        raw["joint_acc"] = float(np.sum(((qd - self.prev_qd) / self.dt) ** 2))
        self.prev_qd = qd.copy()
        raw["action_rate"] = float(np.sum((action - prev_action) ** 2))
        raw["energy"] = float(np.sum(power))
        slip = np.sum(foot_vel[:, :2] ** 2, axis=1) * foot_contact
        raw["foot_slip"] = float(slip.sum())
        raw["default_pose"] = float(np.sum((q - self.default_q) ** 2))

        cmd_n = float(np.linalg.norm(cmd[:2]))
        if cmd_n > self.deadband:
            progress = float(np.dot(lin_vel_b[:2], cmd[:2]) / cmd_n)
            raw["stall"] = float(np.clip(1.0 - progress / cmd_n, 0.0, 1.0))
        elif abs(cmd[2]) > self.deadband:
            raw["stall"] = float(np.clip(1.0 - ang_vel_b[2] * np.sign(cmd[2]) / abs(cmd[2]), 0.0, 1.0))
        else:
            raw["stall"] = 0.0
        raw["body_contact"] = float(body_contacts)
        raw["self_collision"] = float(self_collisions)
        raw["fall"] = 1.0 if fell else 0.0

        parts = {k: w.get(k, 0.0) * v for k, v in raw.items()}
        total = float(sum(parts.values()))
        return total, parts
