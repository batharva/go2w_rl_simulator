"""``Go2QuadrupedEnv`` - Gymnasium environment around the PyBullet Go2 simulation.

* action  : 12 normalised joint-position targets in [-1, 1] (see ``JointPDController.action_to_target``);
* obs     : 52-dim normalised vector (see ``environment/observation.py``);
* control : PD torque control at ~50 Hz (physics sub-stepped, torque re-computed every physics step);
* terrain : procedural heightfields chosen by the curriculum stage or by the ``terrain`` config section;
* DR      : mass / damping / friction / motor strength / sensor noise / pushes / initial conditions.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.domain_randomization import DomainRandomizer, EpisodeRandomization
from environment.observation import OBS_DIM, ObservationBuilder
from environment.reward import COMPONENTS, RewardCalculator
from environment.termination import check_termination, check_truncation
from simulation.pybullet_sim import PyBulletSim
from terrain.terrain_generator import TERRAIN_TYPES, TerrainParams
from utils.config import deep_update, default_config
from utils.logger import get_logger

log = get_logger(__name__)


class Go2QuadrupedEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, config: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None,
                 gui: bool = False, load_visuals: bool = False, seed: Optional[int] = None):
        super().__init__()
        cfg = default_config()
        if config:
            deep_update(cfg, config)
        self.cfg = cfg
        self.render_mode = render_mode
        self.sim = PyBulletSim(cfg, gui=gui, load_visuals=load_visuals or render_mode == "rgb_array")
        rb = self.sim.robot
        self.dt = self.sim.physics.control_dt
        self.max_steps = int(round(cfg["simulation"]["max_episode_seconds"] / self.dt))
        self.settle_steps = int(round(cfg["simulation"]["settle_seconds"] / self.sim.physics.timestep))
        self.obs_builder = ObservationBuilder(cfg, rb.default_q)
        target_h = float(cfg["robot"]["target_base_height"]) or rb.default_height_above_feet
        self.reward_fn = RewardCalculator(cfg, rb.default_q, target_h, self.dt)
        self.randomizer = DomainRandomizer(cfg)

        self.observation_space = spaces.Box(-float(cfg["observation"]["clip"]), float(cfg["observation"]["clip"]),
                                            (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)

        # curriculum / user overrides
        self.stage: Optional[Dict[str, Any]] = None
        self.stage_difficulty = float(cfg["terrain"]["difficulty"])
        self.fixed_command: Optional[np.ndarray] = None
        self._terrain_dirty = True
        self._terrain_counter = 0
        self.episode_count = 0
        self._render_cam = dict(distance=1.6, yaw=45.0, pitch=-25.0)
        self.rand = EpisodeRandomization()
        self.command = np.zeros(3)
        self._cmd_timer = 0.0
        self.prev_action = np.zeros(12, dtype=np.float32)
        self._step = 0
        self._last: Dict[str, Any] = {}
        self._ep: Dict[str, Any] = {}
        if seed is not None:
            self.reset(seed=seed)

    # ================================================================== public configuration API
    def set_stage(self, stage: Optional[Dict[str, Any]], difficulty: Optional[float] = None) -> None:
        """Select a curriculum stage (``None`` -> use plain config).  Takes effect at the next reset."""
        self.stage = copy.deepcopy(stage) if stage else None
        if difficulty is not None:
            self.stage_difficulty = float(difficulty)
        if self.stage is not None:
            self.randomizer.active = bool(self.stage.get("randomization", True))
            self.randomizer.push_scale = float(self.stage.get("push", 0.0))
        else:
            self.randomizer.active = bool(self.cfg["randomization"]["enabled"])
            self.randomizer.push_scale = 1.0
        self._terrain_dirty = True

    def set_difficulty(self, difficulty: float) -> None:
        self.stage_difficulty = float(difficulty)
        self._terrain_dirty = True

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        """Fix the velocity command (GUI / evaluation).  ``clear_command`` returns to random commands."""
        self.fixed_command = np.array([vx, vy, wz], dtype=float)
        self.command = self.fixed_command.copy()

    def clear_command(self) -> None:
        self.fixed_command = None

    def set_terrain_config(self, **kwargs: Any) -> None:
        """Update terrain parameters (type, difficulty, seed, ...); the terrain is rebuilt at the next reset."""
        self.cfg["terrain"].update(kwargs)
        self._terrain_dirty = True

    def set_reward_weights(self, weights: Dict[str, float]) -> None:
        self.reward_fn.set_weights(weights)

    def set_randomization(self, **kwargs: Any) -> None:
        self.cfg["randomization"].update(kwargs)
        self.randomizer.active = bool(self.cfg["randomization"]["enabled"])

    def set_gains(self, kp: float, kd: float) -> None:
        self.sim.controller.set_gains(kp, kd)

    def apply_push(self, vx: float, vy: float) -> None:
        self.sim.apply_push(np.array([vx, vy]))

    # ================================================================== internals
    def _stage_value(self, key: str, default):
        return self.stage[key] if self.stage and key in self.stage else default

    def _choose_terrain_params(self) -> TerrainParams:
        t = self.cfg["terrain"]
        rng = self.np_random
        if self.stage is not None:
            ttype = str(rng.choice(self.stage["terrains"]))
            d_lo, d_hi = self.stage.get("difficulty", [0.0, 1.0])
            level = float(np.clip(self.stage_difficulty, d_lo, d_hi))
            diff = float(level * rng.uniform(0.6, 1.0)) if ttype != "flat" else 0.0
            seed = int(rng.integers(0, 2 ** 31 - 1))
            fr = t["friction"]
            return TerrainParams.from_cfg(t, type=ttype, difficulty=diff, seed=seed, friction=fr)
        self._terrain_counter += 1
        seed = int(t["seed"]) + (self._terrain_counter - 1 if t.get("regen_every_episodes", 0) > 0 else 0)
        ttype = t["type"]
        if t.get("random_type"):
            ttype = str(rng.choice(TERRAIN_TYPES))
        return TerrainParams.from_cfg(t, type=ttype, seed=seed)

    def _maybe_regenerate_terrain(self) -> None:
        every = int(self.cfg["terrain"].get("regen_every_episodes", 0))
        due = every > 0 and self.episode_count % every == 0
        if self._terrain_dirty or due or self.sim.terrain is None:
            self.sim.set_terrain(self._choose_terrain_params())
            self._terrain_dirty = False

    def regenerate_terrain(self, seed: Optional[int] = None, **overrides: Any) -> Dict[str, Any]:
        """Immediately build a new terrain (used by the GUI).  Call ``reset`` afterwards to respawn the robot."""
        if seed is None:
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        self.cfg["terrain"].update(overrides)
        self.cfg["terrain"]["seed"] = int(seed)
        self.stage = None if overrides.get("_use_config", True) else self.stage
        params = TerrainParams.from_cfg(self.cfg["terrain"])
        self.sim.set_terrain(params)
        self._terrain_dirty = False
        return params.to_dict()

    def _sample_command(self) -> None:
        c = self.cfg["commands"]
        rng = self.np_random
        if self.fixed_command is not None:
            self.command = self.fixed_command.copy()
        else:
            if rng.random() < c["standing_probability"]:
                cmd = np.zeros(3)
            else:
                cmd = np.array([rng.uniform(*self._stage_value("vx", c["vx"])),
                                rng.uniform(*self._stage_value("vy", c["vy"])),
                                rng.uniform(*self._stage_value("wz", c["wz"]))])
                cmd[np.abs(cmd) < c["deadband"]] = 0.0
            self.command = cmd
        self._cmd_timer = float(rng.uniform(*c["resample_seconds"]))

    def _observe(self, base, q, qd, contacts) -> np.ndarray:
        return self.obs_builder.build(base.lin_vel_b, base.ang_vel_b, base.gravity_b, self.command, q, qd,
                                      self.prev_action, contacts.foot_contact, self.np_random, self.rand.noise_level)

    def _height_above_feet(self, base, foot_pos, contact) -> float:
        z = foot_pos[contact, 2].mean() if contact.any() else foot_pos[:, 2].mean()
        return float(base.pos[2] - z)

    # ================================================================== Gymnasium API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_count += 1
        rb, sim = self.sim.robot, self.sim
        self._maybe_regenerate_terrain()
        terrain = sim.terrain

        self.rand = self.randomizer.sample(self.np_random, terrain.params.friction)
        self.randomizer.apply(sim, self.rand)

        xy = self.rand.init_xy
        fp = np.array([[dx, dy] for dx in (-0.25, 0.25) for dy in (-0.15, 0.15)]) + xy
        z_ground = float(np.max(terrain.height_at(fp[:, 0], fp[:, 1])))
        q0 = np.clip(rb.default_q + self.rand.init_q_noise, sim.controller.q_min, sim.controller.q_max)
        sim.reset_robot((xy[0], xy[1], z_ground + rb.default_base_z + 0.01), self.rand.init_yaw, q0)
        self.reward_fn.reset()
        self.prev_action = np.zeros(12, dtype=np.float32)
        self._step = 0
        self._sample_command()
        # let the robot settle onto the ground with the PD controller holding the nominal stance
        sim.step(rb.default_q, self.settle_steps)

        base = sim.base_state()
        q, qd = sim.controller.read_state()
        contacts = sim.contacts()
        self._ep = dict(start_xy=base.pos[:2].copy(), comp={k: 0.0 for k in COMPONENTS}, vel_err=0.0, wz_err=0.0,
                        speed=0.0, energy=0.0, slip=0.0, tilt=0.0, ret=0.0, n=0, contact_frames=0,
                        proj_dist=0.0, cmd_sum=np.zeros(3), cmd_norm=0.0, cmd_wz=0.0)
        self._last = {}
        info = {"terrain": terrain.params.type, "terrain_difficulty": terrain.params.difficulty,
                "command": self.command.copy(), "randomization": self.rand.as_dict()}
        return self._observe(base, q, qd, contacts), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        sim, rb, ctrl = self.sim, self.sim.robot, self.sim.controller
        action = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float32)), -1.0, 1.0)
        q_des = ctrl.action_to_target(action.astype(np.float64))
        _, power, _ = sim.step(q_des)
        self._step += 1
        t = self._step * self.dt
        pushed = self.randomizer.maybe_push(sim, self.rand, t, self.np_random)

        base = sim.base_state()
        q, qd = ctrl.read_state()
        contacts = sim.contacts()
        foot_pos, foot_vel = sim.foot_state()
        h_feet = self._height_above_feet(base, foot_pos, contacts.foot_contact)
        rpy = base.rpy
        fell, reason = check_termination(self.cfg, roll=rpy[0], pitch=rpy[1], height_above_feet=h_feet,
                                         base_contact=contacts.base_contact)
        reward, parts = self.reward_fn.compute(
            cmd=self.command, lin_vel_b=base.lin_vel_b, ang_vel_b=base.ang_vel_b, gravity_b=base.gravity_b,
            height_above_feet=h_feet, q=q, qd=qd, action=action, prev_action=self.prev_action, power=power,
            foot_contact=contacts.foot_contact, foot_vel=foot_vel, body_contacts=contacts.body_contacts,
            self_collisions=contacts.self_collisions, fell=fell)
        self.prev_action = action
        terminated = bool(fell)
        trunc, trunc_reason = check_truncation(self.cfg, step=self._step, max_steps=self.max_steps,
                                               x=base.pos[0], y=base.pos[1], half_size=sim.terrain.half_size)
        truncated = bool(trunc) and not terminated

        self._cmd_timer -= self.dt
        if self._cmd_timer <= 0.0 and not (terminated or truncated):
            self._sample_command()

        # ---- episode statistics --------------------------------------------------------
        ep = self._ep
        v_err = float(np.linalg.norm(base.lin_vel_b[:2] - self.command[:2]))
        ep["n"] += 1
        ep["ret"] += reward
        for k, v in parts.items():
            ep["comp"][k] += v
        ep["vel_err"] += v_err
        ep["wz_err"] += abs(float(base.ang_vel_b[2] - self.command[2]))
        ep["speed"] += float(np.linalg.norm(base.lin_vel_b[:2]))
        ep["energy"] += float(power.sum())
        ep["slip"] += float(np.sqrt(np.sum(foot_vel[:, :2] ** 2, axis=1))[contacts.foot_contact].mean()) \
            if contacts.foot_contact.any() else 0.0
        ep["tilt"] += float(np.arccos(np.clip(-base.gravity_b[2], -1, 1)))
        ep["contact_frames"] += int(contacts.foot_contact.any())
        ep["cmd_sum"] += self.command
        ep["cmd_norm"] += float(np.linalg.norm(self.command[:2]))
        ep["cmd_wz"] += abs(float(self.command[2]))

        self._last = dict(reward=reward, parts=parts, speed=float(np.linalg.norm(base.lin_vel_b[:2])),
                          vx=float(base.lin_vel_b[0]), vy=float(base.lin_vel_b[1]), wz=float(base.ang_vel_b[2]),
                          height=float(base.pos[2]), height_above_feet=h_feet, roll=float(rpy[0]),
                          pitch=float(rpy[1]), yaw=float(rpy[2]), contacts=contacts.foot_contact.copy(),
                          terrain=sim.terrain.params.type, pos=base.pos.copy(), command=self.command.copy(),
                          t=t, pushed=pushed, v_err=v_err, tau=ctrl.last_tau.copy(), term_reason=reason or trunc_reason)
        info: Dict[str, Any] = {"command": self.command.copy(), "pushed": pushed}
        if terminated or truncated:
            info["episode_stats"] = self._episode_summary(base, fell, reason or trunc_reason)
        return self._observe(base, q, qd, contacts), float(reward), terminated, truncated, info

    def _episode_summary(self, base, fell: bool, reason: str) -> Dict[str, Any]:
        ep, n = self._ep, max(self._ep["n"], 1)
        disp = base.pos[:2] - ep["start_xy"]
        cmd_mean = ep["cmd_sum"] / n
        cn = float(np.linalg.norm(cmd_mean[:2]))
        yaw = float(base.rpy[2])
        # distance along the commanded direction (in the initial body frame), signed
        along = float(disp @ (cmd_mean[:2] / cn)) if cn > 1e-6 else float(np.linalg.norm(disp))
        return {
            "length": ep["n"], "duration": ep["n"] * self.dt, "return": ep["ret"], "fell": bool(fell), "reason": reason,
            "components": {k: v / n for k, v in ep["comp"].items()},
            "vel_error": ep["vel_err"] / n, "wz_error": ep["wz_err"] / n, "mean_speed": ep["speed"] / n,
            "energy_w": ep["energy"] / n, "foot_slip": ep["slip"] / n, "tilt": ep["tilt"] / n,
            "distance": float(np.linalg.norm(disp)), "distance_along_cmd": along,
            "mean_cmd": cmd_mean.tolist(), "cmd_speed": ep["cmd_norm"] / n, "cmd_wz_abs": ep["cmd_wz"] / n, "terrain": self.sim.terrain.params.type,
            "difficulty": self.sim.terrain.params.difficulty, "final_yaw": yaw,
        }

    # ================================================================== misc
    def telemetry(self) -> Dict[str, Any]:
        """Latest state for the GUI (cheap; no PyBullet calls except joint state)."""
        d = dict(self._last)
        d["episode"] = self.episode_count
        d["step"] = self._step
        return d

    def set_camera(self, distance: float, yaw: float, pitch: float) -> None:
        self._render_cam.update(distance=distance, yaw=yaw, pitch=pitch)

    def render(self, width: int = 480, height: int = 320) -> Optional[np.ndarray]:
        if self.render_mode != "rgb_array":
            return None
        pos = self.sim.base_state().pos
        return self.sim.render(width, height, tuple(pos), self._render_cam["distance"], self._render_cam["yaw"],
                               self._render_cam["pitch"])

    def close(self) -> None:
        self.sim.close()
