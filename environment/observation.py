"""Observation construction (fixed layout, fixed-scale normalisation, optional sensor noise).

Layout (OBS_DIM = 52)
---------------------
 idx   size  content                                   scale
 0:3     3   base linear velocity, body frame          x 2.0
 3:6     3   base angular velocity, body frame         x 0.25
 6:9     3   projected gravity (unit vector, body)     x 1
 9:12    3   command (vx, vy, wz)                      x (2, 2, 0.25)
 12:24  12   joint positions minus default pose        x 1
 24:36  12   joint velocities                          x 0.05
 36:48  12   previous action                           x 1
 48:52   4   foot contact flags (FL, FR, RL, RR)       0 / 1
All entries are clipped to +-``observation.clip``.  No terrain information is given (blind locomotion); noise is only
added to quantities a real robot would measure with an IMU / encoders / state estimator.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

OBS_DIM = 52
OBS_LAYOUT = [("base_lin_vel", 3), ("base_ang_vel", 3), ("projected_gravity", 3), ("command", 3),
              ("joint_pos_rel", 12), ("joint_vel", 12), ("previous_action", 12), ("foot_contact", 4)]
assert sum(n for _, n in OBS_LAYOUT) == OBS_DIM


class ObservationBuilder:
    def __init__(self, cfg: Dict[str, Any], default_q: np.ndarray):
        o = cfg["observation"]
        self.s_lin = float(o["scales"]["lin_vel"])
        self.s_ang = float(o["scales"]["ang_vel"])
        self.s_jv = float(o["scales"]["joint_vel"])
        self.s_cmd = np.asarray(o["scales"]["command"], dtype=np.float32)
        self.clip = float(o["clip"])
        self.noise = o["noise"]
        self.default_q = default_q.astype(np.float32)

    def build(self, lin_vel_b, ang_vel_b, gravity_b, command, q, qd, prev_action, foot_contact,
              rng: np.random.Generator, noise_level: float = 0.0) -> np.ndarray:
        n = self.noise
        if noise_level > 0.0:
            g = lambda s, k: rng.normal(0.0, s * noise_level, k)   # noqa: E731
            lin_vel_b = lin_vel_b + g(n["lin_vel"], 3)
            ang_vel_b = ang_vel_b + g(n["ang_vel"], 3)
            gravity_b = gravity_b + g(n["gravity"], 3)
            q = q + g(n["joint_pos"], 12)
            qd = qd + g(n["joint_vel"], 12)
        obs = np.concatenate([
            lin_vel_b * self.s_lin, ang_vel_b * self.s_ang, gravity_b, np.asarray(command) * self.s_cmd,
            q - self.default_q, qd * self.s_jv, prev_action, foot_contact.astype(np.float32)]).astype(np.float32)
        return np.clip(obs, -self.clip, self.clip)
