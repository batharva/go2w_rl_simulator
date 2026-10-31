"""Physics engine configuration (time step, solver, gravity, sub-stepping)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pybullet as p


@dataclass
class PhysicsConfig:
    timestep: float = 1.0 / 240.0
    control_freq: float = 50.0
    gravity: float = -9.81
    solver_iterations: int = 50

    @property
    def substeps(self) -> int:
        """Physics steps per control step (>=1)."""
        return max(1, int(round(1.0 / (self.control_freq * self.timestep))))

    @property
    def control_dt(self) -> float:
        """Effective control period. With 1/240 s and 50 Hz this is 5/240 s (48 Hz)."""
        return self.substeps * self.timestep

    @classmethod
    def from_cfg(cls, cfg: Dict[str, Any]) -> "PhysicsConfig":
        s = cfg["simulation"]
        return cls(float(s["timestep"]), float(s["control_freq"]), float(s["gravity"]),
                   int(s["solver_iterations"]))

    def validate(self) -> None:
        if not (1e-4 <= self.timestep <= 1.0 / 60.0):
            raise ValueError(f"timestep {self.timestep} outside the numerically sensible range [1e-4, 1/60] s")
        if self.control_freq <= 0 or self.control_freq > 1.0 / self.timestep:
            raise ValueError("control_freq must be > 0 and <= 1/timestep")
        if self.gravity > 0:
            raise ValueError("gravity must be negative (z-up world)")

    def apply(self, client: int) -> None:
        """Push the configuration into a PyBullet client."""
        self.validate()
        p.setGravity(0, 0, self.gravity, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.timestep,
            numSolverIterations=int(self.solver_iterations),
            numSubSteps=0,                  # we sub-step manually so we can apply torques every physics step
            enableConeFriction=1,
            deterministicOverlappingPairs=1,
            physicsClientId=client,
        )
        p.setTimeStep(self.timestep, physicsClientId=client)
