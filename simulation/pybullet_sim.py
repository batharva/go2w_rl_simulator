"""One PyBullet client = terrain + Go2 + PD actuators + contact/state queries + optional rendering.

``PyBulletSim`` is the only object that talks to PyBullet's global API for a given client id; the Gym environment,
the GUI worker and the validation/evaluation code all go through it, so a physics client is never shared between
components.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from simulation.physics_config import PhysicsConfig
from simulation.robot_controller import JointPDController
from simulation.robot_loader import RobotModel, load_robot
from terrain.terrain_generator import Terrain, TerrainGenerator, TerrainParams
from utils.config import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ContactState:
    foot_force: np.ndarray                      # (4,) summed normal force per foot [N]
    foot_contact: np.ndarray                    # (4,) bool
    base_contact: bool = False                  # base / head touching terrain (fall)
    body_contacts: int = 0                      # hip / thigh / calf touching terrain
    self_collisions: int = 0                    # contacts between non-adjacent robot links


@dataclass
class BaseState:
    pos: np.ndarray                             # URDF base-frame origin in world
    quat: np.ndarray                            # xyzw
    rot: np.ndarray                             # 3x3 body->world
    lin_vel_w: np.ndarray
    ang_vel_w: np.ndarray

    @property
    def lin_vel_b(self) -> np.ndarray:
        return self.rot.T @ self.lin_vel_w

    @property
    def ang_vel_b(self) -> np.ndarray:
        return self.rot.T @ self.ang_vel_w

    @property
    def gravity_b(self) -> np.ndarray:
        """Unit gravity direction expressed in the body frame (== 'projected gravity')."""
        return self.rot.T @ np.array([0.0, 0.0, -1.0])

    @property
    def rpy(self) -> np.ndarray:
        return np.array(p.getEulerFromQuaternion(self.quat.tolist()))


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PyBulletSim:
    def __init__(self, cfg: Dict[str, Any], gui: bool = False, load_visuals: bool = False):
        self.cfg = cfg
        self.gui = gui
        self.load_visuals = load_visuals or gui
        self.cid = p.connect(p.GUI if gui else p.DIRECT)
        if self.cid < 0:
            raise RuntimeError("Could not connect to PyBullet")
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.cid)
        self.physics = PhysicsConfig.from_cfg(cfg)
        self.physics.apply(self.cid)
        cache = resolve_path(cfg, "cache")
        self.terrain_gen = TerrainGenerator(self.cid, with_visual=self.load_visuals, cache_dir=cache)
        self.terrain: Terrain = self.terrain_gen.generate(TerrainParams.from_cfg(cfg["terrain"]))
        self.robot: RobotModel = load_robot(self.cid, cfg, load_visuals=self.load_visuals)
        self.controller = JointPDController(self.robot, cfg)
        self.force_threshold = float(cfg["simulation"]["contact_force_threshold"])
        # inertial frame offset of the base (PyBullet reports the COM pose, the URDF frame is what we want)
        d = p.getDynamicsInfo(self.robot.body_id, -1, physicsClientId=self.cid)
        self._base_com_off = np.array(d[3])
        self._foot_idx = {l: i for i, l in enumerate(self.robot.foot_ids)}
        self._body_set = set(self.robot.thigh_calf_ids) | set(self.robot.hip_ids)
        self._term_set = set(self.robot.terminate_link_ids)
        self._orig_masses = dict(self.robot.masses)
        self._orig_inertia = dict(self.robot.inertias)
        self.closed = False

    # ------------------------------------------------------------------ terrain
    def set_terrain(self, params: TerrainParams) -> Terrain:
        self.terrain = self.terrain_gen.generate(params)
        return self.terrain

    # ------------------------------------------------------------------ stepping
    def step(self, q_target: np.ndarray, n_substeps: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance physics ``n`` steps while the PD controller tracks ``q_target``.

        Returns the mean applied torque, mean |tau*qdot| power and the last torque (for acceleration/energy terms).
        """
        n = n_substeps or self.physics.substeps
        power = np.zeros(self.robot.num_joints)
        tau_sum = np.zeros_like(power)
        for _ in range(n):
            q, qd = self.controller.read_state()
            tau = self.controller.apply(q_target, q, qd)
            p.stepSimulation(physicsClientId=self.cid)
            power += np.abs(tau * qd)
            tau_sum += tau
        return tau_sum / n, power / n, tau

    # ------------------------------------------------------------------ reset / state
    def reset_robot(self, base_pos, yaw: float, q: np.ndarray) -> None:
        orn = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        rot = quat_to_rot(np.array(orn))
        com_pos = np.asarray(base_pos) + rot @ self._base_com_off
        b, c = self.robot.body_id, self.cid
        p.resetBasePositionAndOrientation(b, com_pos.tolist(), orn, physicsClientId=c)
        p.resetBaseVelocity(b, [0, 0, 0], [0, 0, 0], physicsClientId=c)
        for j, qi in zip(self.robot.joint_ids, q):
            p.resetJointState(b, j, float(qi), 0.0, physicsClientId=c)
        self.controller.last_tau[:] = 0.0

    def base_state(self) -> BaseState:
        b, c = self.robot.body_id, self.cid
        pos, orn = p.getBasePositionAndOrientation(b, physicsClientId=c)
        lin, ang = p.getBaseVelocity(b, physicsClientId=c)
        rot = quat_to_rot(np.array(orn))
        return BaseState(np.array(pos) - rot @ self._base_com_off, np.array(orn), rot, np.array(lin), np.array(ang))

    def foot_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """World positions (4,3) and world linear velocities (4,3) of the foot links."""
        st = p.getLinkStates(self.robot.body_id, self.robot.foot_ids, computeLinkVelocity=1, physicsClientId=self.cid)
        return np.array([s[4] for s in st]), np.array([s[6] for s in st])

    def contacts(self) -> ContactState:
        r = self.robot
        cs = ContactState(np.zeros(4), np.zeros(4, dtype=bool))
        for cp in p.getContactPoints(bodyA=r.body_id, physicsClientId=self.cid):
            body_b, link_a, link_b, fn = cp[2], cp[3], cp[4], cp[9]
            if body_b == r.body_id:
                cs.self_collisions += 1
            elif link_a in self._foot_idx:
                cs.foot_force[self._foot_idx[link_a]] += fn
            elif link_a in self._term_set:
                cs.base_contact = True
            elif link_a in self._body_set and fn > self.force_threshold:
                cs.body_contacts += 1
        cs.foot_contact = cs.foot_force > self.force_threshold
        return cs

    # ------------------------------------------------------------------ dynamics randomisation hooks
    def scale_masses(self, scale: float) -> None:
        """Scale mass and principal inertia of every robot link (relative to the URDF values)."""
        b, c = self.robot.body_id, self.cid
        for idx, m in self._orig_masses.items():
            I = self._orig_inertia[idx]
            p.changeDynamics(b, idx, mass=m * scale, localInertiaDiagonal=[i * scale for i in I], physicsClientId=c)

    def apply_push(self, delta_v_xy: np.ndarray) -> None:
        """Instantaneous velocity disturbance of the base (external push)."""
        lin, ang = p.getBaseVelocity(self.robot.body_id, physicsClientId=self.cid)
        p.resetBaseVelocity(self.robot.body_id, [lin[0] + delta_v_xy[0], lin[1] + delta_v_xy[1], lin[2]], ang,
                            physicsClientId=self.cid)

    # ------------------------------------------------------------------ rendering (DIRECT mode, software renderer)
    def render(self, width: int, height: int, target: Tuple[float, float, float], distance: float, yaw: float,
               pitch: float) -> np.ndarray:
        view = p.computeViewMatrixFromYawPitchRoll(list(target), distance, yaw, pitch, 0.0, 2,
                                                   physicsClientId=self.cid)
        proj = p.computeProjectionMatrixFOV(60.0, width / float(height), 0.05, 60.0, physicsClientId=self.cid)
        _, _, rgba, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                                            flags=p.ER_NO_SEGMENTATION_MASK, physicsClientId=self.cid)
        return np.reshape(np.asarray(rgba, dtype=np.uint8), (height, width, 4))[:, :, :3]

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            try:
                p.disconnect(self.cid)
            except p.error:
                pass
