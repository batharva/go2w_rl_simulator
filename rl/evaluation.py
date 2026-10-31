"""Policy evaluation on fixed scenarios + checkpoint comparison + CSV/JSON reports.

Evaluation uses the *deterministic* policy, fixed seeds that differ from training seeds (so terrains are unseen), and by
default no domain randomisation / pushes (``robust=True`` enables them).  Nothing here is "claimed": the verdict is
computed from the measured metrics against the thresholds in ``evaluation`` (config) and reported as PASS / NOT PASSED.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from environment.go2_env import Go2QuadrupedEnv
from utils.config import deep_update, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Scenario:
    name: str
    terrain: str
    difficulty: float
    command: Tuple[float, float, float]
    flat_reference: bool = False      # counts for the strict overall verdict


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario("flat_stand", "flat", 0.0, (0.0, 0.0, 0.0), True),
    Scenario("flat_forward", "flat", 0.0, (0.5, 0.0, 0.0), True),
    Scenario("flat_backward", "flat", 0.0, (-0.3, 0.0, 0.0), True),
    Scenario("flat_lateral", "flat", 0.0, (0.0, 0.3, 0.0), True),
    Scenario("flat_turn", "flat", 0.0, (0.0, 0.0, 0.8), True),
    Scenario("uneven", "uneven", 0.5, (0.4, 0.0, 0.0)),
    Scenario("slopes", "slopes", 0.5, (0.4, 0.0, 0.0)),
    Scenario("stairs", "stairs", 0.4, (0.3, 0.0, 0.0)),
    Scenario("obstacles", "obstacles", 0.5, (0.4, 0.0, 0.0)),
    Scenario("mixed", "mixed", 0.5, (0.4, 0.0, 0.0)),
]


def summarize(stats: Sequence[Dict[str, Any]], expected_cmd_distance: Optional[Sequence[float]] = None) -> Dict[str, float]:
    """Aggregate per-episode statistics (as produced by ``Go2QuadrupedEnv``) into scenario metrics."""
    n = max(len(stats), 1)
    fell = np.array([s["fell"] for s in stats], dtype=float)
    out = {
        "episodes": len(stats),
        "avg_episode_reward": float(np.mean([s["return"] for s in stats])),
        "distance_travelled": float(np.mean([s["distance"] for s in stats])),
        "distance_along_command": float(np.mean([s["distance_along_cmd"] for s in stats])),
        "avg_velocity": float(np.mean([s["mean_speed"] for s in stats])),
        "vel_error": float(np.mean([s["vel_error"] for s in stats])),
        "yaw_rate_error": float(np.mean([s["wz_error"] for s in stats])),
        "cmd_speed": float(np.mean([s.get("cmd_speed", 0.0) for s in stats])),
        "cmd_wz_abs": float(np.mean([s.get("cmd_wz_abs", 0.0) for s in stats])),
        "fall_rate": float(fell.mean()),
        "survival_rate": float(1.0 - fell.mean()),
        "completion_rate": float(np.mean([(not s["fell"]) and s["reason"] in ("time_limit", "arena_border")
                                          for s in stats])),
        "orientation_deviation_rad": float(np.mean([s["tilt"] for s in stats])),
        "energy_w": float(np.mean([s["energy_w"] for s in stats])),
        "foot_slip_mps": float(np.mean([s["foot_slip"] for s in stats])),
        "episode_length_s": float(np.mean([s["duration"] for s in stats])),
    }
    return out


def rollout_episodes(model, env: Go2QuadrupedEnv, n_episodes: int, seed: Optional[int] = None,
                     deterministic: bool = True, stop_fn: Optional[Callable[[], bool]] = None) -> List[Dict[str, Any]]:
    """Run ``n_episodes`` full episodes and return the per-episode statistics."""
    stats: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        if stop_fn and stop_fn():
            break
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        stats.append(info["episode_stats"])
    return stats


def terrain_completed(stat: Dict[str, Any], cmd: Tuple[float, float, float]) -> bool:
    """No fall AND the commanded motion was actually achieved (standing still never counts as completing a walk)."""
    if stat["fell"]:
        return False
    speed = float(np.hypot(cmd[0], cmd[1]))
    if speed > 0.05:                                  # at least 40 % of the commanded travel along the command
        return stat["distance_along_cmd"] >= 0.4 * speed * stat["duration"]
    if abs(cmd[2]) > 0.05:                            # turning in place: yaw-rate error below half the command
        return stat["wz_error"] <= 0.5 * abs(cmd[2]) and stat["mean_speed"] < 0.3
    return stat["distance"] < 0.5                     # stand: must not drift away


class Evaluator:
    def __init__(self, cfg: Dict[str, Any], robust: bool = False):
        ev = cfg["evaluation"]
        c = json.loads(json.dumps(cfg))                      # deep copy of plain data
        deep_update(c, {"simulation": {"max_episode_seconds": ev["episode_seconds"]},
                        "randomization": {"enabled": bool(robust), "push_enabled": bool(robust)},
                        "terrain": {"regen_every_episodes": 0}})
        self.cfg = c
        self.eval_cfg = ev
        self.robust = robust
        self.env: Optional[Go2QuadrupedEnv] = None

    def _env(self) -> Go2QuadrupedEnv:
        if self.env is None:
            self.env = Go2QuadrupedEnv(self.cfg)
        return self.env

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def run_scenario(self, model, sc: Scenario, seed: int, episodes: int,
                     stop_fn: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        env = self._env()
        env.set_stage(None)
        env.randomizer.active = self.robust
        env.randomizer.push_scale = 1.0 if self.robust else 0.0
        env.set_command(*sc.command)
        stats = []
        for ep in range(episodes):
            if stop_fn and stop_fn():
                break
            env.set_terrain_config(type=sc.terrain, difficulty=sc.difficulty, seed=seed + 101 * ep)
            stats.extend(rollout_episodes(model, env, 1, seed=seed + ep, stop_fn=stop_fn))
        res = summarize(stats)
        res["terrain_completion_rate"] = float(np.mean([terrain_completed(s, sc.command) for s in stats])) if stats else 0.0
        res.update(scenario=sc.name, terrain=sc.terrain, difficulty=sc.difficulty, command=list(sc.command))
        e = self.eval_cfg
        scale = 1.0 if sc.flat_reference else 1.5
        moving = np.hypot(sc.command[0], sc.command[1]) > 0.05 or abs(sc.command[2]) > 0.05
        res["passed"] = bool(res["fall_rate"] <= e["pass_fall_rate"] * scale
                             and res["vel_error"] <= e["pass_vel_error"] * scale
                             and res["yaw_rate_error"] <= e["pass_vel_error"] * scale * 2.0
                             and res["completion_rate"] >= e["pass_completion"] / scale
                             # a moving command additionally requires real progress (cannot pass by standing still)
                             and (not moving or res["terrain_completion_rate"] >= e["pass_completion"] / scale))
        return res

    def evaluate(self, model, scenarios: Optional[Sequence[Scenario]] = None, episodes: Optional[int] = None,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 stop_fn: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        scenarios = list(scenarios or DEFAULT_SCENARIOS)
        episodes = episodes or int(self.eval_cfg["episodes_per_scenario"])
        seed = int(self.eval_cfg["seed"])
        rows = []
        for i, sc in enumerate(scenarios):
            if stop_fn and stop_fn():
                break
            if progress:
                progress(i, len(scenarios), sc.name)
            rows.append(self.run_scenario(model, sc, seed, episodes, stop_fn))
            log.info("eval %-14s fall=%.2f vel_err=%.3f complete=%.2f", sc.name, rows[-1]["fall_rate"],
                     rows[-1]["vel_error"], rows[-1]["completion_rate"])
        if progress:
            progress(len(scenarios), len(scenarios), "done")
        flat = [r for r, s in zip(rows, scenarios) if s.flat_reference]
        keys = ["avg_episode_reward", "distance_travelled", "avg_velocity", "vel_error", "fall_rate",
                "completion_rate", "orientation_deviation_rad", "energy_w", "foot_slip_mps",
                "terrain_completion_rate"]
        overall = {k: float(np.mean([r[k] for r in rows])) for k in keys} if rows else {}
        passed_flat = bool(flat) and all(r["passed"] for r in flat)
        overall["passed_all_flat_scenarios"] = passed_flat
        overall["scenarios_passed"] = int(sum(r["passed"] for r in rows))
        overall["scenarios_total"] = len(rows)
        overall["verdict"] = ("PASS: all flat-ground scenarios meet the thresholds" if passed_flat
                              else "NOT PASSED: the policy does not yet meet the flat-ground thresholds")
        return {"scenarios": rows, "overall": overall, "robust": self.robust, "episodes_per_scenario": episodes,
                "seed": seed, "thresholds": {k: self.eval_cfg[k] for k in ("pass_fall_rate", "pass_vel_error",
                                                                              "pass_completion")}}


def write_report(report: Dict[str, Any], out_dir: Path, tag: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    jp, cp = out_dir / f"eval_{tag}_{stamp}.json", out_dir / f"eval_{tag}_{stamp}.csv"
    jp.write_text(json.dumps(report, indent=2, default=float))
    rows = report["scenarios"]
    cols = ["scenario", "terrain", "difficulty", "command", "episodes", "avg_episode_reward", "distance_travelled",
            "distance_along_command", "avg_velocity", "vel_error", "yaw_rate_error", "fall_rate", "completion_rate",
            "terrain_completion_rate", "orientation_deviation_rad", "energy_w", "foot_slip_mps", "passed"]
    with open(cp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "command": " ".join(f"{c:g}" for c in r["command"])})
        w.writerow({"scenario": "OVERALL_MEAN", **{k: v for k, v in report["overall"].items() if k in cols}})
    return jp, cp


def evaluate_model_file(model_path: str | Path, cfg: Dict[str, Any], episodes: Optional[int] = None,
                        robust: bool = False, progress=None, stop_fn=None,
                        scenarios: Optional[Sequence[Scenario]] = None) -> Dict[str, Any]:
    from rl.model_manager import ModelManager
    model = ModelManager.load_ppo(model_path, device="cpu")
    ev = Evaluator(cfg, robust=robust)
    try:
        rep = ev.evaluate(model, scenarios=scenarios, episodes=episodes, progress=progress, stop_fn=stop_fn)
    finally:
        ev.close()
    rep["model"] = str(model_path)
    rep["model_timesteps"] = int(model.num_timesteps)
    jp, cp = write_report(rep, resolve_path(cfg, "results"), Path(model_path).stem)
    rep["report_json"], rep["report_csv"] = str(jp), str(cp)
    return rep


def compare_models(model_paths: Sequence[str | Path], cfg: Dict[str, Any], episodes: Optional[int] = None,
                   robust: bool = False, progress=None) -> Dict[str, Any]:
    """Evaluate several checkpoints with identical scenarios/seeds and write a side-by-side CSV + JSON."""
    reports = {}
    for i, mp in enumerate(model_paths):
        if progress:
            progress(i, len(model_paths), Path(mp).name)
        reports[str(mp)] = evaluate_model_file(mp, cfg, episodes, robust)
    out = resolve_path(cfg, "results")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    cp, jp = out / f"comparison_{stamp}.csv", out / f"comparison_{stamp}.json"
    keys = ["avg_episode_reward", "distance_travelled", "avg_velocity", "vel_error", "fall_rate", "completion_rate",
            "orientation_deviation_rad", "energy_w", "foot_slip_mps", "terrain_completion_rate"]
    with open(cp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "timesteps", "scenario"] + keys + ["passed"])
        for mp, rep in reports.items():
            for r in rep["scenarios"]:
                w.writerow([Path(mp).name, rep["model_timesteps"], r["scenario"]] + [f"{r[k]:.4f}" for k in keys]
                           + [r["passed"]])
            o = rep["overall"]
            w.writerow([Path(mp).name, rep["model_timesteps"], "OVERALL_MEAN"] + [f"{o[k]:.4f}" for k in keys]
                       + [o["passed_all_flat_scenarios"]])
    jp.write_text(json.dumps({k: {"overall": v["overall"], "timesteps": v["model_timesteps"]}
                              for k, v in reports.items()}, indent=2, default=float))
    return {"reports": reports, "csv": str(cp), "json": str(jp)}


def evaluation_worker_main(cfg: Dict[str, Any], model_paths: List[str], episodes: int, robust: bool, out_q,
                           stop_event) -> None:
    """Process entry point used by the GUI ('Evaluate Model' / 'Compare')."""
    import queue as _q
    import traceback
    from utils.logger import setup_logging
    setup_logging()

    def emit(m: Dict[str, Any]) -> None:
        try:
            out_q.put(m, timeout=2.0)
        except _q.Full:
            pass

    try:
        def prog(i: int, n: int, name: str) -> None:
            emit({"type": "progress", "done": i, "total": n, "name": name})
        if len(model_paths) == 1:
            rep = evaluate_model_file(model_paths[0], cfg, episodes, robust, progress=prog, stop_fn=stop_event.is_set)
            emit({"type": "result", "reports": {model_paths[0]: rep}, "csv": rep["report_csv"],
                  "json": rep["report_json"]})
        else:
            comp = compare_models(model_paths, cfg, episodes, robust, progress=prog)
            emit({"type": "result", "reports": comp["reports"], "csv": comp["csv"], "json": comp["json"]})
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    finally:
        emit({"type": "finished"})
