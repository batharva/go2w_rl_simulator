"""Stable-Baselines3 callbacks: pause/stop control, live metrics, checkpoints, curriculum evaluation."""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from environment.go2_env import Go2QuadrupedEnv
from rl.curriculum import CurriculumManager
from rl.evaluation import rollout_episodes, summarize
from rl.model_manager import ModelManager, atomic_save
from utils.config import deep_update, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)
Sink = Callable[[Dict[str, Any]], None]


class ControlCallback(BaseCallback):
    """Honours the GUI's pause / stop requests (events shared with the controlling process)."""

    def __init__(self, stop_event, pause_event, sink: Sink, save_event=None, save_fn=None):
        super().__init__()
        self.stop_event, self.pause_event, self.sink = stop_event, pause_event, sink
        self.save_event, self.save_fn = save_event, save_fn
        self._announced_pause = False

    def _on_step(self) -> bool:
        if self.save_event is not None and self.save_event.is_set():     # "Save Model" while training
            self.save_event.clear()
            if self.save_fn is not None:
                self.save_fn(self.model)
        if self.pause_event is not None and self.pause_event.is_set():
            if not self._announced_pause:
                self.sink({"type": "status", "state": "paused"})
                self._announced_pause = True
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.1)
            if not self.stop_event.is_set():
                self.sink({"type": "status", "state": "running"})
            self._announced_pause = False
        if self.stop_event.is_set():
            log.info("Stop requested - leaving the training loop")
            return False
        return True


class MetricsCallback(BaseCallback):
    """Forwards per-episode statistics and per-update losses to the sink (GUI / console) and TensorBoard."""

    def __init__(self, sink: Sink, window: int = 100):
        super().__init__()
        self.sink = sink
        self.returns: Deque[float] = deque(maxlen=window)
        self.vel_err: Deque[float] = deque(maxlen=window)
        self.t0 = time.time()
        self.t_start_steps = 0
        self.episodes = 0
        self._last_update = 0.0

    def _on_training_start(self) -> None:
        self.t0 = time.time()
        self.t_start_steps = self.num_timesteps

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        for info, done in zip(infos, dones):
            st = info.get("episode_stats")
            if done and st is not None:
                self.episodes += 1
                self.returns.append(st["return"])
                self.vel_err.append(st["vel_error"])
                self.sink({"type": "episode", "timesteps": self.num_timesteps, "episode": self.episodes,
                           "return": st["return"], "length": st["length"], "vel_error": st["vel_error"],
                           "mean_speed": st["mean_speed"], "fell": st["fell"], "terrain": st["terrain"],
                           "difficulty": st["difficulty"], "components": st["components"],
                           "reason": st["reason"], "mean_return": float(np.mean(self.returns))})
                self.logger.record_mean("custom/vel_error", st["vel_error"])
                for k, v in st["components"].items():
                    self.logger.record_mean(f"reward/{k}", v)
        return True

    def _on_rollout_end(self) -> None:
        nv = self.logger.name_to_value
        fps = (self.num_timesteps - self.t_start_steps) / max(time.time() - self.t0, 1e-6)
        self.sink({"type": "update", "timesteps": self.num_timesteps, "fps": fps,
                   "policy_loss": nv.get("train/policy_gradient_loss"), "value_loss": nv.get("train/value_loss"),
                   "entropy_loss": nv.get("train/entropy_loss"), "approx_kl": nv.get("train/approx_kl"),
                   "explained_variance": nv.get("train/explained_variance"),
                   "mean_return": float(np.mean(self.returns)) if self.returns else None,
                   "mean_vel_error": float(np.mean(self.vel_err)) if self.vel_err else None})


class CheckpointCallback(BaseCallback):
    """Periodic full checkpoints (+ side-car) and a light 'latest policy' snapshot for the GUI preview."""

    def __init__(self, cfg: Dict[str, Any], mm: ModelManager, curriculum: CurriculumManager, sink: Sink,
                 state: Dict[str, Any]):
        super().__init__()
        p = cfg["ppo"]
        self.freq, self.snap_freq = int(p["checkpoint_freq"]), int(p["snapshot_freq"])
        self.mm, self.curr, self.sink, self.state = mm, curriculum, sink, state
        self.snap_path = resolve_path(cfg, "models") / "latest_policy.zip"
        self._last_ckpt = self._last_snap = 0

    def meta(self) -> Dict[str, Any]:
        st = self.curr.stage
        return {"stage": st["name"] if st else None, "difficulty": self.curr.difficulty,
                "curriculum": self.curr.state_dict(), "eval_score": self.state.get("last_score")}

    def _on_training_start(self) -> None:
        self._last_ckpt = self._last_snap = self.num_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_snap >= self.snap_freq:
            self._last_snap = self.num_timesteps
            atomic_save(self.model, self.snap_path)
        if self.num_timesteps - self._last_ckpt >= self.freq:
            self._last_ckpt = self.num_timesteps
            path = self.mm.save_checkpoint(self.model, self.meta())
            self.sink({"type": "checkpoint", "path": str(path), "timesteps": self.num_timesteps})
        return True


class CurriculumEvalCallback(BaseCallback):
    """Periodic deterministic evaluation -> curriculum decision -> best-model saving.

    The evaluation env runs in this process with the stage's own conditions (terrain types, difficulty, command ranges,
    randomisation).  Training environments are updated through ``env_method('set_stage', ...)``.
    """

    def __init__(self, cfg: Dict[str, Any], mm: ModelManager, curriculum: CurriculumManager, sink: Sink,
                 ckpt_cb: CheckpointCallback, state: Dict[str, Any], stop_event):
        super().__init__()
        self.cfg, self.mm, self.curr, self.sink, self.ckpt_cb = cfg, mm, curriculum, sink, ckpt_cb
        self.state, self.stop_event = state, stop_event
        c = cfg["curriculum"]
        self.freq, self.n_eps = int(c["eval_freq"]), int(c["eval_episodes"])
        ecfg = json.loads(json.dumps(cfg))
        deep_update(ecfg, {"simulation": {"max_episode_seconds": float(c["eval_episode_seconds"])},
                           "terrain": {"regen_every_episodes": 1}})
        self.eval_cfg = ecfg
        self.env: Optional[Go2QuadrupedEnv] = None
        self._last_eval = 0
        self.best_key = tuple(state.get("best_key", (-1, -1.0, -1.0)))

    def _on_training_start(self) -> None:
        self._last_eval = self.num_timesteps
        self.env = Go2QuadrupedEnv(self.eval_cfg, seed=int(self.cfg["ppo"]["seed"]) + 10_000)
        self._apply_to_training_envs()

    def _apply_to_training_envs(self) -> None:
        self.training_env.env_method("set_stage", self.curr.stage, self.curr.difficulty)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.freq:
            self._last_eval = self.num_timesteps
            self.evaluate()
        return not self.stop_event.is_set()

    def evaluate(self) -> None:
        assert self.env is not None
        self.env.set_stage(self.curr.stage, self.curr.difficulty)
        stats = rollout_episodes(self.model, self.env, self.n_eps, seed=int(self.cfg["evaluation"]["seed"]),
                                 stop_fn=self.stop_event.is_set)
        if not stats:
            return
        m = summarize(stats)
        score = float(m["survival_rate"] * (1.0 - min(m["vel_error"] / 0.5, 1.0)))
        self.state["last_score"] = score
        decision = self.curr.update(m, self.num_timesteps) if self.curr.enabled else None
        key_level = self.curr.level
        # best model: furthest curriculum progress first, then the evaluation score
        key = (key_level[0], key_level[1], score)
        improved = key > self.best_key
        if improved:
            self.best_key = key
            self.state["best_key"] = list(key)
            self.mm.save_best(self.model, {**self.ckpt_cb.meta(), "eval_metrics": m, "eval_score": score})
        self.logger.record("curriculum/stage_index", self.curr.idx)
        self.logger.record("curriculum/difficulty", self.curr.difficulty)
        self.logger.record("eval/survival_rate", m["survival_rate"])
        self.logger.record("eval/vel_error", m["vel_error"])
        self.logger.record("eval/score", score)
        stage = self.curr.stage
        self.sink({"type": "curriculum", "timesteps": self.num_timesteps, "metrics": m, "score": score,
                   "stage": stage["name"] if stage else "disabled", "stage_index": self.curr.idx,
                   "difficulty": self.curr.difficulty, "decision": decision.action if decision else "n/a",
                   "reason": decision.reason if decision else "", "best_saved": improved})
        if decision is not None and decision.action in ("harder", "advance"):
            self._apply_to_training_envs()

    def _on_training_end(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
