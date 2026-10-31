"""PPO training pipeline (Stable-Baselines3) with curriculum, checkpoints and GUI/console reporting."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch.nn as nn

from environment.go2_env import Go2QuadrupedEnv
from rl.callbacks import CheckpointCallback, ControlCallback, CurriculumEvalCallback, MetricsCallback
from rl.curriculum import CurriculumManager
from rl.model_manager import ModelManager
from utils.config import resolve_path
from utils.logger import get_logger, setup_logging

log = get_logger(__name__)

ACTIVATIONS = {"elu": nn.ELU, "tanh": nn.Tanh, "relu": nn.ReLU}


def validate_ppo_config(cfg: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty = valid)."""
    p, errs = cfg["ppo"], []
    def need(cond: bool, msg: str) -> None:
        if not cond:
            errs.append(msg)
    need(0 < p["learning_rate"] < 1, "learning_rate must be in (0, 1)")
    need(0 < p["gamma"] <= 1, "gamma must be in (0, 1]")
    need(0 < p["gae_lambda"] <= 1, "gae_lambda must be in (0, 1]")
    need(p["n_steps"] >= 16, "n_steps (rollout steps) must be >= 16")
    need(p["n_envs"] >= 1, "n_envs must be >= 1")
    need(p["batch_size"] >= 2, "batch_size must be >= 2")
    need(p["batch_size"] <= p["n_steps"] * p["n_envs"],
         f"batch_size ({p['batch_size']}) must not exceed n_steps*n_envs ({p['n_steps'] * p['n_envs']})")
    need(p["n_epochs"] >= 1, "n_epochs must be >= 1")
    need(0 < p["clip_range"] < 1, "clip_range must be in (0, 1)")
    need(p["ent_coef"] >= 0, "ent_coef must be >= 0")
    need(p["vf_coef"] >= 0, "vf_coef must be >= 0")
    need(p["max_grad_norm"] > 0, "max_grad_norm must be > 0")
    need(p["total_timesteps"] >= p["n_steps"] * p["n_envs"], "total_timesteps must cover at least one rollout")
    need(p["activation"] in ACTIVATIONS, f"activation must be one of {list(ACTIVATIONS)}")
    for k in ("net_arch_pi", "net_arch_vf"):
        need(len(p[k]) >= 1 and all(int(u) >= 1 for u in p[k]), f"{k} needs at least one positive layer size")
    t = cfg["terrain"]
    need(0 <= t["difficulty"] <= 1, "terrain difficulty must be in [0, 1]")
    need(t["min_height"] <= t["max_height"], "terrain min_height must be <= max_height")
    need(cfg["curriculum"]["enabled"] is False or any(s.get("enabled", True) for s in cfg["curriculum"]["stages"]),
         "curriculum enabled but all stages disabled")
    return errs


def make_env_fn(cfg: Dict[str, Any], rank: int, seed: int) -> Callable[[], Go2QuadrupedEnv]:
    def _init() -> Go2QuadrupedEnv:
        setup_logging()
        return Go2QuadrupedEnv(cfg, seed=seed + rank)
    return _init


class PPOTrainer:
    def __init__(self, cfg: Dict[str, Any], sink: Optional[Callable[[Dict[str, Any]], None]] = None,
                 stop_event=None, pause_event=None, save_event=None):
        self.cfg = cfg
        self.sink = sink or (lambda m: None)
        self.stop_event = stop_event or mp.Event()
        self.pause_event = pause_event or mp.Event()
        self.save_event = save_event
        self.mm = ModelManager(cfg)
        self.curriculum = CurriculumManager(cfg)
        self.model = None
        self.venv = None

    # ------------------------------------------------------------------ construction
    def _build_vec_env(self):
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
        p = self.cfg["ppo"]
        fns = [make_env_fn(self.cfg, i, int(p["seed"])) for i in range(int(p["n_envs"]))]
        venv = DummyVecEnv(fns) if len(fns) == 1 else SubprocVecEnv(fns, start_method="spawn")
        return VecMonitor(venv)

    def _build_model(self, venv, resume: Optional[Path]):
        from stable_baselines3 import PPO
        p = self.cfg["ppo"]
        tb = None
        try:
            import tensorboard  # noqa: F401
            tb = str(resolve_path(self.cfg, "logs") / "tensorboard")
        except ImportError:
            log.warning("tensorboard not installed - TensorBoard logging disabled")
        hyper = dict(learning_rate=float(p["learning_rate"]), n_steps=int(p["n_steps"]),
                     batch_size=int(p["batch_size"]), n_epochs=int(p["n_epochs"]), gamma=float(p["gamma"]),
                     gae_lambda=float(p["gae_lambda"]), clip_range=float(p["clip_range"]),
                     ent_coef=float(p["ent_coef"]), vf_coef=float(p["vf_coef"]),
                     max_grad_norm=float(p["max_grad_norm"]))
        if resume is not None:
            log.info("Resuming from %s", resume)
            # hyper-parameters edited in the GUI override the stored ones
            return PPO.load(str(resume), env=venv, device=p["device"], custom_objects=dict(hyper),
                            tensorboard_log=tb)
        policy_kwargs = dict(net_arch=dict(pi=[int(u) for u in p["net_arch_pi"]], vf=[int(u) for u in p["net_arch_vf"]]),
                             activation_fn=ACTIVATIONS[p["activation"]], log_std_init=float(p["log_std_init"]))
        return PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs, tensorboard_log=tb, seed=int(p["seed"]),
                   device=p["device"], verbose=0, **hyper)

    def _manual_save(self, model, ckpt) -> None:
        path = self.mm.save(model, self.mm.models_dir / f"manual_{int(model.num_timesteps):010d}.zip", ckpt.meta())
        self.sink({"type": "checkpoint", "path": str(path), "timesteps": int(model.num_timesteps), "manual": True})

    # ------------------------------------------------------------------ main entry
    def train(self, resume: Optional[str | Path] = None, total_timesteps: Optional[int] = None) -> Path:
        errs = validate_ppo_config(self.cfg)
        if errs:
            raise ValueError("Invalid configuration:\n  - " + "\n  - ".join(errs))
        from stable_baselines3.common.callbacks import CallbackList
        from stable_baselines3.common.utils import set_random_seed
        p = self.cfg["ppo"]
        set_random_seed(int(p["seed"]))
        resume_path = Path(resume) if resume else None
        if resume_path is not None:
            meta = self.mm.read_meta(resume_path)
            if "curriculum" in meta:
                self.curriculum.load_state_dict(meta["curriculum"])
        self.sink({"type": "status", "state": "starting"})
        self.venv = self._build_vec_env()
        try:
            self.model = self._build_model(self.venv, resume_path)
            target = int(total_timesteps or p["total_timesteps"])
            remaining = target - int(self.model.num_timesteps)
            if remaining <= 0:
                log.warning("Model already at %d >= target %d steps; training %d additional steps",
                            self.model.num_timesteps, target, target)
                remaining = target
            n_params = sum(x.numel() for x in self.model.policy.parameters())
            log.info("PPO ready: %d envs, %d parameters, obs=%s, act=%s, %d steps to go", p["n_envs"], n_params,
                     self.venv.observation_space.shape, self.venv.action_space.shape, remaining)
            self.sink({"type": "info", "parameters": n_params, "remaining": remaining,
                       "obs_dim": int(self.venv.observation_space.shape[0]),
                       "start_timesteps": int(self.model.num_timesteps)})
            state: Dict[str, Any] = {}
            ckpt = CheckpointCallback(self.cfg, self.mm, self.curriculum, self.sink, state)
            cbs = CallbackList([
                ControlCallback(self.stop_event, self.pause_event, self.sink, self.save_event,
                                lambda m: self._manual_save(m, ckpt)),
                MetricsCallback(self.sink),
                ckpt,
                CurriculumEvalCallback(self.cfg, self.mm, self.curriculum, self.sink, ckpt, state, self.stop_event),
            ])
            self.sink({"type": "status", "state": "running"})
            self.model.learn(total_timesteps=remaining, callback=cbs, reset_num_timesteps=False,
                             tb_log_name="ppo", progress_bar=False)
            stopped = self.stop_event.is_set()
            last = self.mm.save_last(self.model, ckpt.meta())
            self.sink({"type": "checkpoint", "path": str(last), "timesteps": int(self.model.num_timesteps),
                       "final": True, "stopped_by_user": stopped})
            log.info("Training %s at %d steps; model saved to %s", "stopped" if stopped else "finished",
                     self.model.num_timesteps, last)
            return last
        finally:
            if self.venv is not None:
                self.venv.close()
                self.venv = None


class QueueSink:
    """Non-blocking sink that drops *progress* messages (never status/errors) when the GUI is slow."""

    def __init__(self, q):
        self.q = q

    def __call__(self, msg: Dict[str, Any]) -> None:
        try:
            self.q.put(msg, block=msg["type"] in ("status", "error", "checkpoint", "curriculum", "info"),
                       timeout=2.0)
        except queue.Full:
            pass


def training_worker_main(cfg: Dict[str, Any], stop_event, pause_event, out_queue, resume: Optional[str] = None,
                         save_event=None) -> None:
    """Process entry point used by the GUI (and by ``main.py --headless``)."""
    setup_logging(log_file=resolve_path(cfg, "logs") / "training.log")
    sink = QueueSink(out_queue)
    try:
        PPOTrainer(cfg, sink, stop_event, pause_event, save_event).train(resume)
    except Exception as e:  # noqa: BLE001 - surfaced to the GUI
        log.exception("Training failed")
        sink({"type": "error", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    finally:
        sink({"type": "status", "state": "finished"})
