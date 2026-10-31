"""Configuration handling.

The configuration is a plain nested ``dict`` (picklable, trivially sent to worker
processes, trivially edited by the GUI).  ``DEFAULT_CONFIG`` below is the single
source of truth; ``config.yaml`` overrides it (deep-merge) and the command line /
GUI override that again.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]
FOOT_LINKS = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

# Reward terms. Value = signed weight (positive = reward, negative = penalty).
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "track_vx": 1.0,            # forward/backward velocity tracking (exp kernel)
    "track_vy": 0.6,            # lateral velocity tracking (exp kernel)
    "track_wz": 0.5,            # yaw-rate tracking (exp kernel)
    "alive": 0.2,               # small survival bonus (keeps total reward > 0 while upright)
    "foot_contact": 0.1,        # >=2 feet on ground (stand: 3+) -> stable support polygon
    "feet_air_time": 1.0,       # swing-phase reward when walking (prevents shuffling / stalling)
    "orientation": -2.0,        # |projected gravity xy|^2
    "base_height": -20.0,       # (height above feet - target)^2
    "lin_vel_z": -2.0,          # vertical bouncing
    "ang_vel_xy": -0.05,        # roll / pitch rate
    "joint_vel": -5e-4,         # smooth joint motion
    "joint_acc": -2.5e-7,
    "action_rate": -0.02,
    "energy": -2e-4,            # sum |tau * qdot|
    "foot_slip": -0.2,          # tangential foot speed while in contact
    "default_pose": -0.05,      # regulariser toward the nominal stance
    "stall": -1.0,              # commanded to move but not moving (anti reward-hacking)
    "body_contact": -1.0,       # thigh / calf / hip touching the ground
    "self_collision": -1.0,
    "fall": -20.0,              # one-off at termination by falling
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "urdf": "assets/go2_description.urdf",
        "mesh_dirs": ["assets/dae", "assets"],
        "models": "models",
        "checkpoints": "checkpoints",
        "logs": "logs",
        "results": "results",
        "cache": ".cache",
    },
    "simulation": {
        "timestep": 1.0 / 240.0,
        "control_freq": 50.0,          # substeps = round(1 / (freq * timestep)) -> 5 (=48 Hz)
        "gravity": -9.81,
        "solver_iterations": 50,
        "max_episode_seconds": 20.0,
        "self_collision": True,
        "ground_friction": 0.9,
        "ground_restitution": 0.0,
        "foot_friction": 1.0,
        "contact_force_threshold": 1.0,   # N, a foot counts as touching above this
        "settle_seconds": 0.25,           # PD hold after every reset (not rewarded)
    },
    "robot": {
        "base_link": "base",
        "joint_order": JOINT_ORDER,
        "foot_links": FOOT_LINKS,
        "stance_height": 0.30,        # hip-to-foot vertical distance used for the IK of the default pose
        "hip_abduction": 0.1,         # rad; + for left legs, - for right legs
        "front_foot_x": 0.0,          # foot x relative to hip (m) in the default pose
        "rear_foot_x": -0.02,
        "target_base_height": 0.0,    # 0 -> derived from the default pose by forward kinematics
        "terminate_on_links": ["base", "Head_upper", "Head_lower"],   # contact with terrain => fall
        "fix_massless_links": True,   # give inertia-less links a tiny mass (see robot_loader.py)
    },
    "control": {
        "kp": 40.0,
        "kd": 1.0,
        "action_scale": 0.25,
        "torque_speed_curve": True,   # linear torque derating towards the URDF velocity limit
        "joint_damping": 0.01,        # passive viscous damping  [N m s/rad]
        "joint_friction": 0.03,       # passive Coulomb friction [N m]
        "joint_limit_margin": 0.02,   # rad kept away from the hard limits
    },
    "observation": {
        "scales": {"lin_vel": 2.0, "ang_vel": 0.25, "joint_vel": 0.05, "command": [2.0, 2.0, 0.25]},
        "clip": 10.0,
        "noise": {"lin_vel": 0.1, "ang_vel": 0.2, "gravity": 0.05, "joint_pos": 0.01, "joint_vel": 0.5},
    },
    "reward": {
        "weights": DEFAULT_REWARD_WEIGHTS,
        "sigma_velocity": 0.15,
        "sigma_angular": 0.25,
        "air_time_threshold": 0.4,
        "target_base_height": 0.0,    # 0 -> use robot.target_base_height
    },
    "termination": {
        "max_roll": 1.0,              # rad
        "max_pitch": 1.0,             # rad
        "min_height_above_feet": 0.12,
        "arena_margin": 1.0,          # m before the terrain border -> truncate
    },
    "terrain": {
        "type": "flat",               # flat|uneven|stepping|stairs|slopes|rough|obstacles|mixed
        "size": 16.0,                 # edge length of the square arena (m)
        "resolution": 0.08,           # heightfield cell size (m)
        "difficulty": 0.3,
        "max_height": 0.15,
        "min_height": 0.0,
        "roughness": 0.5,
        "obstacle_density": 0.6,      # obstacles per m^2 (scaled by difficulty)
        "obstacle_size": 0.25,        # nominal edge length (m)
        "slope_angle": 15.0,          # deg, reached at difficulty 1.0
        "stair_height": 0.12,         # max step height at difficulty 1.0
        "stair_width": 0.32,          # nominal tread depth
        "friction": 0.9,
        "restitution": 0.0,
        "seed": 0,
        "spawn_pad": 1.5,             # flat radius around the spawn point (m)
        "regen_every_episodes": 4,    # a fresh random terrain every N episodes (training)
        "random_type": False,         # pick the type randomly each regeneration (from the stage list)
    },
    "randomization": {
        "enabled": True,
        "mass_scale": [0.85, 1.15],
        "joint_damping_scale": [0.5, 2.0],
        "friction": [0.5, 1.3],
        "motor_strength": [0.9, 1.1],
        "kp_scale": [0.9, 1.1],
        "sensor_noise": True,
        "push_enabled": True,
        "push_interval_s": [4.0, 9.0],
        "push_velocity": 0.5,         # max delta-v (m/s) at full curriculum strength
        "init_pos_xy": 0.3,           # m
        "init_yaw": 0.4,              # rad
        "init_joint_noise": 0.1,      # rad
    },
    "commands": {
        "resample_seconds": [3.0, 6.0],
        "vx": [-0.4, 0.8],
        "vy": [-0.4, 0.4],
        "wz": [-1.0, 1.0],
        "standing_probability": 0.1,
        "deadband": 0.1,
    },
    "curriculum": {
        "enabled": True,
        "eval_freq": 50_000,           # timesteps between curriculum evaluations
        "eval_episodes": 5,
        "eval_episode_seconds": 10.0,
        "difficulty_step": 0.15,
        "min_steps_per_stage": 100_000,
        "stages": [
            {"name": "1_standing_balance", "enabled": True, "terrains": ["flat"],
             "difficulty": [0.0, 0.0], "vx": [0, 0], "vy": [0, 0], "wz": [0, 0],
             "randomization": False, "push": 0.0, "min_survival": 0.9, "max_vel_error": 0.12},
            {"name": "2_basic_walking", "enabled": True, "terrains": ["flat"],
             "difficulty": [0.0, 0.0], "vx": [-0.5, 0.8], "vy": [0, 0], "wz": [0, 0],
             "randomization": False, "push": 0.0, "min_survival": 0.85, "max_vel_error": 0.25},
            {"name": "3_multidirectional", "enabled": True, "terrains": ["flat"],
             "difficulty": [0.0, 0.0], "vx": [-0.6, 0.9], "vy": [-0.5, 0.5], "wz": [-1.2, 1.2],
             "randomization": True, "push": 0.0, "min_survival": 0.85, "max_vel_error": 0.3},
            {"name": "4_terrain_adaptation", "enabled": True,
             "terrains": ["uneven", "slopes", "obstacles", "rough", "flat"],
             "difficulty": [0.1, 0.6], "vx": [-0.5, 0.8], "vy": [-0.4, 0.4], "wz": [-1.0, 1.0],
             "randomization": True, "push": 0.0, "min_survival": 0.8, "max_vel_error": 0.35},
            {"name": "5_robust_locomotion", "enabled": True,
             "terrains": ["mixed", "stairs", "stepping", "uneven", "slopes", "obstacles", "rough"],
             "difficulty": [0.3, 1.0], "vx": [-0.5, 0.8], "vy": [-0.4, 0.4], "wz": [-1.0, 1.0],
             "randomization": True, "push": 1.0, "min_survival": 0.75, "max_vel_error": 0.4},
        ],
    },
    "ppo": {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 1024,
        "n_epochs": 5,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 1.0,
        "max_grad_norm": 1.0,
        "total_timesteps": 5_000_000,
        "net_arch_pi": [256, 128, 64],
        "net_arch_vf": [256, 128, 64],
        "activation": "elu",           # elu|tanh|relu
        "log_std_init": -0.7,
        "seed": 42,
        "n_envs": 4,
        "device": "cpu",
        "checkpoint_freq": 100_000,
        "snapshot_freq": 20_000,       # light-weight "latest policy" file for the GUI preview
    },
    "evaluation": {
        "episodes_per_scenario": 3,
        "episode_seconds": 10.0,
        "seed": 1234,
        "pass_fall_rate": 0.2,
        "pass_vel_error": 0.3,
        "pass_completion": 0.7,
    },
    "gui": {
        "render_width": 480,
        "render_height": 300,
        "render_fps": 15,
        "realtime_factor": 1.0,
    },
}


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``new`` into ``base`` (in place) and return ``base``."""
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def default_config() -> Dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load defaults, then ``config.yaml`` (if present), then ``overrides``."""
    cfg = default_config()
    p = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        deep_update(cfg, user)
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Resolve ``cfg['paths'][key]`` relative to the project root and create dirs for output paths."""
    p = Path(cfg["paths"][key])
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if key not in ("urdf", "mesh_dirs"):
        p.mkdir(parents=True, exist_ok=True)
    return p


def config_to_json(cfg: Dict[str, Any]) -> str:
    return json.dumps(cfg, indent=2, default=str)
