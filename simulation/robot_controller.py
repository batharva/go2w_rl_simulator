"""Actuator model: PD position control converted to *torques* with URDF effort / velocity limits.

    tau = Kp (q_des - q) + Kd (qdot_des - qdot)  -  c_damp * qdot  -  c_fric * tanh(qdot / eps)

The torque is clipped to the URDF ``effort`` of each joint (scaled by a per-joint motor-strength factor used for
domain randomisation) and, optionally, derated linearly towards zero at the URDF ``velocity`` limit (a simple DC-motor
torque-speed curve).  The network never writes torques directly: it only chooses joint position targets.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pybullet as p

from simulation.robot_loader import RobotModel


class JointPDController:
    def __init__(self, robot: RobotModel, cfg: Dict[str, Any]):
        self.robot = robot
        c = cfg["control"]
        n = robot.num_joints
        self.kp0 = np.full(n, float(c["kp"]))
        self.kd0 = np.full(n, float(c["kd"]))
        self.kp = self.kp0.copy()
        self.kd = self.kd0.copy()
        self.action_scale = float(c["action_scale"])
        self.use_speed_curve = bool(c["torque_speed_curve"])
        self.damping0 = np.full(n, float(c["joint_damping"]))
        self.damping = self.damping0.copy()
        self.friction = np.full(n, float(c["joint_friction"]))
        self.margin = float(c["joint_limit_margin"])
        self.strength = np.ones(n)                       # motor strength multiplier (domain randomisation)
        self.q_min = robot.lower + self.margin
        self.q_max = robot.upper - self.margin
        self.default_q = robot.default_q.copy()
        # asymmetric action mapping: a=+1 -> `scale` of the way to the upper limit, a=-1 -> to the lower limit
        self._span_up = np.maximum(self.q_max - self.default_q, 0.0)
        self._span_dn = np.maximum(self.default_q - self.q_min, 0.0)
        self.last_tau = np.zeros(n)
        self.last_q_des = self.default_q.copy()

    # ------------------------------------------------------------------ configuration
    def set_gains(self, kp: float, kd: float) -> None:
        self.kp0[:] = kp
        self.kd0[:] = kd
        self.kp, self.kd = self.kp0.copy(), self.kd0.copy()

    def randomize(self, strength: np.ndarray, kp_scale: float = 1.0, damping_scale: float = 1.0) -> None:
        self.strength = np.asarray(strength, dtype=float)
        self.kp = self.kp0 * kp_scale
        self.kd = self.kd0.copy()
        self.damping = self.damping0 * damping_scale

    def reset_randomization(self) -> None:
        self.strength = np.ones_like(self.strength)
        self.kp, self.kd, self.damping = self.kp0.copy(), self.kd0.copy(), self.damping0.copy()

    # ------------------------------------------------------------------ mapping
    def action_to_target(self, action: np.ndarray) -> np.ndarray:
        """Normalised action in [-1, 1] -> joint position target (always inside the URDF limits)."""
        a = np.clip(action, -1.0, 1.0)
        q = self.default_q + self.action_scale * np.where(a >= 0.0, a * self._span_up, a * self._span_dn)
        return np.clip(q, self.q_min, self.q_max)

    def target_to_action(self, q_des: np.ndarray) -> np.ndarray:
        d = q_des - self.default_q
        a = np.where(d >= 0, d / np.maximum(self.action_scale * self._span_up, 1e-9),
                     d / np.maximum(self.action_scale * self._span_dn, 1e-9))
        return np.clip(a, -1.0, 1.0)

    # ------------------------------------------------------------------ state / torque
    def read_state(self) -> Tuple[np.ndarray, np.ndarray]:
        st = p.getJointStates(self.robot.body_id, self.robot.joint_ids, physicsClientId=self.robot.client)
        q = np.fromiter((s[0] for s in st), float, count=len(st))
        qd = np.fromiter((s[1] for s in st), float, count=len(st))
        return q, qd

    def compute_torque(self, q: np.ndarray, qd: np.ndarray, q_des: np.ndarray,
                       qd_des: np.ndarray | None = None) -> np.ndarray:
        qd_des = 0.0 if qd_des is None else qd_des
        tau = self.kp * (q_des - q) + self.kd * (qd_des - qd)
        tau -= self.damping * qd + self.friction * np.tanh(qd / 0.05)
        limit = self.robot.effort * self.strength
        if self.use_speed_curve:
            # torque available falls linearly to 0 at the velocity limit when the motor is driven *along* its motion
            derate = np.clip(1.0 - np.abs(qd) / self.robot.vel_limit, 0.0, 1.0)
            pos_lim = np.where(qd > 0, limit * derate, limit)
            neg_lim = np.where(qd < 0, limit * derate, limit)
            return np.clip(tau, -neg_lim, pos_lim)
        return np.clip(tau, -limit, limit)

    def apply(self, q_des: np.ndarray, q: np.ndarray | None = None, qd: np.ndarray | None = None) -> np.ndarray:
        """Compute and send torques for one physics step. Returns the applied torques."""
        if q is None or qd is None:
            q, qd = self.read_state()
        tau = self.compute_torque(q, qd, q_des)
        p.setJointMotorControlArray(self.robot.body_id, self.robot.joint_ids, p.TORQUE_CONTROL,
                                    forces=tau.tolist(), physicsClientId=self.robot.client)
        self.last_tau = tau
        self.last_q_des = q_des
        return tau
