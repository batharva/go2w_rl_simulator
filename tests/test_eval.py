"""Evaluation: metrics, scenarios, CSV/JSON reports, checkpoint comparison."""
import csv
import json
from pathlib import Path

import numpy as np

from rl.evaluation import DEFAULT_SCENARIOS, Evaluator, Scenario, compare_models, evaluate_model_file, summarize
from rl.ppo_trainer import PPOTrainer


def _make_models(cfg):
    PPOTrainer(cfg).train()
    return Path(cfg["paths"]["models"]) / "last_model.zip"


def test_summarize_metrics():
    stats = [dict(**{"return": 10.0, "distance": 1.0, "distance_along_cmd": 0.9, "mean_speed": 0.3, "vel_error": 0.2,
                     "wz_error": 0.1, "fell": False, "reason": "time_limit", "tilt": 0.05, "energy_w": 50.0,
                     "foot_slip": 0.02, "duration": 10.0}),
             dict(**{"return": -5.0, "distance": 0.1, "distance_along_cmd": 0.0, "mean_speed": 0.1, "vel_error": 0.6,
                     "wz_error": 0.3, "fell": True, "reason": "roll", "tilt": 0.5, "energy_w": 90.0,
                     "foot_slip": 0.2, "duration": 2.0})]
    m = summarize(stats)
    assert m["fall_rate"] == 0.5 and m["completion_rate"] == 0.5 and abs(m["vel_error"] - 0.4) < 1e-9
    assert m["avg_episode_reward"] == 2.5 and m["survival_rate"] == 0.5


def test_evaluate_model_writes_reports_and_no_false_success_claim(cfg):
    model = _make_models(cfg)
    scen = [Scenario("flat_forward", "flat", 0.0, (0.5, 0, 0), True), Scenario("slopes", "slopes", 0.4, (0.4, 0, 0))]
    rep = evaluate_model_file(model, cfg, episodes=1, scenarios=scen)
    assert [r["scenario"] for r in rep["scenarios"]] == ["flat_forward", "slopes"]
    for k in ("avg_episode_reward", "distance_travelled", "vel_error", "fall_rate", "completion_rate",
              "orientation_deviation_rad", "energy_w", "foot_slip_mps", "terrain_completion_rate"):
        assert np.isfinite(rep["scenarios"][0][k])
    assert Path(rep["report_json"]).exists() and Path(rep["report_csv"]).exists()
    rows = list(csv.DictReader(open(rep["report_csv"])))
    assert rows[-1]["scenario"] == "OVERALL_MEAN" and len(rows) == 3
    data = json.load(open(rep["report_json"]))
    # a ~1000-step policy cannot walk: the verdict must say so rather than claim success
    assert data["overall"]["verdict"].startswith("NOT PASSED")


def test_default_scenarios_cover_required_terrains():
    names = {s.terrain for s in DEFAULT_SCENARIOS}
    assert {"flat", "uneven", "slopes", "stairs", "obstacles", "mixed"} <= names
    cmds = {s.command for s in DEFAULT_SCENARIOS if s.terrain == "flat"}
    assert (0.0, 0.0, 0.0) in cmds and any(c[0] < 0 for c in cmds) and any(c[1] != 0 for c in cmds) and any(c[2] != 0 for c in cmds)


def test_compare_checkpoints(cfg):
    _make_models(cfg)
    ck = sorted(Path(cfg["paths"]["checkpoints"]).glob("ckpt_*.zip"))
    assert len(ck) >= 2
    import rl.evaluation as ev
    old = ev.DEFAULT_SCENARIOS[:]
    ev.DEFAULT_SCENARIOS[:] = [Scenario("flat_forward", "flat", 0.0, (0.5, 0, 0), True)]
    try:
        res = compare_models(ck[:2], cfg, episodes=1)
    finally:
        ev.DEFAULT_SCENARIOS[:] = old
    assert Path(res["csv"]).exists() and len(res["reports"]) == 2
    assert len(list(csv.reader(open(res["csv"])))) == 1 + 2 * 2      # header + (scenario + overall) per model


def test_evaluation_is_deterministic(cfg):
    model = _make_models(cfg)
    from rl.model_manager import ModelManager
    m = ModelManager.load_ppo(model)
    s = Scenario("flat_forward", "flat", 0.0, (0.5, 0, 0), True)
    a = Evaluator(cfg); ra = a.run_scenario(m, s, 123, 1); a.close()
    b = Evaluator(cfg); rb = b.run_scenario(m, s, 123, 1); b.close()
    assert abs(ra["avg_episode_reward"] - rb["avg_episode_reward"]) < 1e-3


def test_stationary_policy_never_passes_moving_scenarios(cfg):
    """Regression: a policy that just stands still must not be reported as walking / turning."""
    class Still:                                   # outputs the zero action -> robot holds the default stance
        def predict(self, obs, deterministic=True):
            return np.zeros(12, dtype=np.float32), None
    ev = Evaluator(cfg)
    for name, terr, cmd in (("fwd", "flat", (0.5, 0, 0)), ("turn", "flat", (0, 0, 0.8)), ("slope", "slopes", (0.4, 0, 0)),
                            ("lat", "flat", (0, 0.3, 0))):
        r = ev.run_scenario(Still(), Scenario(name, terr, 0.3, cmd, terr == "flat"), 5, 1)
        assert not r["passed"], (name, r)
        assert r["terrain_completion_rate"] == 0.0
    r = ev.run_scenario(Still(), Scenario("stand", "flat", 0.0, (0, 0, 0), True), 5, 1)
    assert r["passed"]                             # standing still IS the right behaviour for a zero command
    ev.close()
