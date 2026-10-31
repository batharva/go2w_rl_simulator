"""Domain randomisation: dynamics, sensors, pushes and initial conditions.

All ranges come from ``cfg['randomization']`` and are applied with the environment's seeded RNG, so a given seed always
produces the same sequence of randomised episodes.  The curriculum can switch randomisation on/off per stage and scale
the push strength.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from simulation.pybullet_sim import PyBulletSim


@dataclass
class EpisodeRandomization:
    mass_scale: float = 1.0
    damping_scale: float = 1.0
    friction: float = 0.9
    motor_strength: np.ndarray = field(default_factory=lambda: np.ones(12))
    kp_scale: float = 1.0
    noise_level: float = 0.0
    next_push_time: float = np.inf
    init_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    init_yaw: float = 0.0
    init_q_noise: np.ndarray = field(default_factory=lambda: np.zeros(12))

    def as_dict(self) -> Dict[str, float]:
        return {"mass_scale": self.mass_scale, "damping_scale": self.damping_scale, "friction": self.friction,
                "motor_strength_mean": float(self.motor_strength.mean()), "kp_scale": self.kp_scale,
                "noise_level": self.noise_level}


class DomainRandomizer:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.active = bool(cfg["randomization"]["enabled"])
        self.push_scale = 1.0           # 0 disables pushes, 1 = full push_velocity (curriculum controlled)

    def sample(self, rng: np.random.Generator, terrain_friction: float, enabled: Optional[bool] = None
               ) -> EpisodeRandomization:
        r = self.cfg["randomization"]
        on = self.active if enabled is None else enabled
        e = EpisodeRandomization(friction=terrain_friction)
        if not on:
            return e
        u = lambda lohi: float(rng.uniform(lohi[0], lohi[1]))   # noqa: E731
        e.mass_scale = u(r["mass_scale"])
        e.damping_scale = u(r["joint_damping_scale"])
        e.friction = u(r["friction"])
        e.motor_strength = rng.uniform(r["motor_strength"][0], r["motor_strength"][1], 12)
        e.kp_scale = u(r["kp_scale"])
        e.noise_level = 1.0 if r["sensor_noise"] else 0.0
        if r["push_enabled"] and self.push_scale > 0.0:
            e.next_push_time = u(r["push_interval_s"])
        e.init_xy = rng.uniform(-r["init_pos_xy"], r["init_pos_xy"], 2)
        e.init_yaw = float(rng.uniform(-r["init_yaw"], r["init_yaw"]))
        e.init_q_noise = rng.uniform(-r["init_joint_noise"], r["init_joint_noise"], 12)
        return e

    @staticmethod
    def apply(sim: PyBulletSim, e: EpisodeRandomization) -> None:
        sim.scale_masses(e.mass_scale)
        sim.controller.randomize(e.motor_strength, e.kp_scale, e.damping_scale)
        sim.terrain_gen.set_surface(e.friction)

    def maybe_push(self, sim: PyBulletSim, e: EpisodeRandomization, t: float, rng: np.random.Generator) -> bool:
        """Apply an external velocity push when its time has come; returns True if a push happened."""
        if t < e.next_push_time or self.push_scale <= 0.0:
            return False
        r = self.cfg["randomization"]
        mag = float(r["push_velocity"]) * self.push_scale * rng.uniform(0.3, 1.0)
        ang = rng.uniform(0, 2 * np.pi)
        sim.apply_push(mag * np.array([np.cos(ang), np.sin(ang)]))
        e.next_push_time = t + float(rng.uniform(r["push_interval_s"][0], r["push_interval_s"][1]))
        return True
