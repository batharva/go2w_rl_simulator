"""Performance-gated curriculum.

The curriculum advances based on *evaluation results* (deterministic-policy episodes under the current stage's
conditions), never on elapsed time alone.  Within a stage the terrain difficulty is raised step by step; when the stage's
maximum difficulty is reached and the policy still passes, the next enabled stage starts.

A stage is "passed" when, in an evaluation of N episodes,
    survival_rate >= stage['min_survival']   and   mean velocity-tracking error <= stage['max_vel_error'].
``curriculum.min_steps_per_stage`` only prevents promoting a policy after a lucky early evaluation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CurriculumDecision:
    action: str                 # "hold" | "harder" | "advance" | "finished"
    passed: bool
    stage_index: int
    stage_name: str
    difficulty: float
    reason: str = ""


class CurriculumManager:
    def __init__(self, cfg: Dict[str, Any]):
        c = cfg["curriculum"]
        self.enabled = bool(c["enabled"])
        self.stages: List[Dict[str, Any]] = [copy.deepcopy(s) for s in c["stages"] if s.get("enabled", True)]
        if self.enabled and not self.stages:
            raise ValueError("Curriculum is enabled but no stage is enabled")
        self.step = float(c["difficulty_step"])
        self.min_steps = int(c["min_steps_per_stage"])
        self.idx = 0
        self.difficulty = self.stages[0]["difficulty"][0] if self.stages else 0.0
        self.stage_start_step = 0
        self.finished = False
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ access
    @property
    def stage(self) -> Optional[Dict[str, Any]]:
        return self.stages[self.idx] if self.enabled and self.stages else None

    @property
    def level(self) -> tuple:
        """Sortable progress indicator used to pick the 'best' model."""
        return (self.idx, round(self.difficulty, 4))

    def apply(self, set_stage_fn) -> None:
        """``set_stage_fn(stage, difficulty)`` is called with the current stage (e.g. ``vec_env.env_method``)."""
        set_stage_fn(self.stage, self.difficulty)

    # ------------------------------------------------------------------ update
    def update(self, metrics: Dict[str, float], timesteps: int) -> CurriculumDecision:
        st = self.stage
        if st is None:
            return CurriculumDecision("finished", True, 0, "disabled", 0.0, "curriculum disabled")
        cmd_speed = float(metrics.get("cmd_speed", 0.0))
        # standing still gives vel_error ~= mean commanded speed; require beating that baseline clearly (<= 60 %)
        beats_still = cmd_speed < 0.15 or metrics["vel_error"] <= 0.6 * cmd_speed
        yaw_ok = metrics.get("yaw_rate_error", 0.0) <= 0.6 * max(float(metrics.get("cmd_wz_abs", 0.0)), 0.25) + 0.15
        passed = bool(metrics["survival_rate"] >= st["min_survival"] and metrics["vel_error"] <= st["max_vel_error"]
                      and beats_still and yaw_ok)
        enough = timesteps - self.stage_start_step >= self.min_steps
        rec = dict(timesteps=timesteps, stage=st["name"], difficulty=self.difficulty, passed=passed, **metrics)
        self.history.append(rec)
        if not passed:
            return CurriculumDecision("hold", False, self.idx, st["name"], self.difficulty,
                                      f"survival {metrics['survival_rate']:.2f}/{st['min_survival']}, "
                                      f"vel_err {metrics['vel_error']:.2f}/{st['max_vel_error']} (must also be <= 0.6x commanded speed "
                                      f"{cmd_speed:.2f} and track yaw)")
        if not enough:
            return CurriculumDecision("hold", True, self.idx, st["name"], self.difficulty, "passed, min steps not reached")
        d_max = st["difficulty"][1]
        if self.difficulty + 1e-9 < d_max:
            self.difficulty = min(d_max, self.difficulty + self.step)
            return CurriculumDecision("harder", True, self.idx, st["name"], self.difficulty, "difficulty increased")
        if self.idx + 1 < len(self.stages):
            self.idx += 1
            self.difficulty = self.stages[self.idx]["difficulty"][0]
            self.stage_start_step = timesteps
            log.info("Curriculum: advancing to stage %s", self.stages[self.idx]["name"])
            return CurriculumDecision("advance", True, self.idx, self.stages[self.idx]["name"], self.difficulty,
                                      "stage passed")
        self.finished = True
        return CurriculumDecision("finished", True, self.idx, st["name"], self.difficulty, "final stage passed")

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return {"idx": self.idx, "difficulty": self.difficulty, "stage_start_step": self.stage_start_step,
                "finished": self.finished, "history": self.history[-50:]}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.idx = min(int(d.get("idx", 0)), max(len(self.stages) - 1, 0))
        self.difficulty = float(d.get("difficulty", self.difficulty))
        self.stage_start_step = int(d.get("stage_start_step", 0))
        self.finished = bool(d.get("finished", False))
        self.history = list(d.get("history", []))
