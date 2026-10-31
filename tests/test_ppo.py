"""Tests 8-10: PPO initialisation, short training, checkpoint save/reload, resume, curriculum, config validation."""
import copy
from pathlib import Path

import numpy as np
import pytest

from rl.curriculum import CurriculumManager
from rl.model_manager import ModelManager
from rl.ppo_trainer import PPOTrainer, validate_ppo_config


def test_config_validation_catches_errors(cfg):
    assert validate_ppo_config(cfg) == []
    bad = copy.deepcopy(cfg)
    bad["ppo"].update(batch_size=10_000_000, gamma=1.5, learning_rate=-1, activation="swish")
    errs = validate_ppo_config(bad)
    assert len(errs) >= 4


def test8_ppo_initialises(cfg):
    t = PPOTrainer(cfg)
    venv = t._build_vec_env()
    model = t._build_model(venv, None)
    pol = model.policy
    assert venv.observation_space.shape == (52,) and venv.action_space.shape == (12,)
    # separate actor / critic networks
    assert len(pol.mlp_extractor.policy_net) > 0 and len(pol.mlp_extractor.value_net) > 0
    assert pol.mlp_extractor.policy_net is not pol.mlp_extractor.value_net
    assert model.n_steps == 128 and model.batch_size == 64 and model.gamma == cfg["ppo"]["gamma"]
    obs = venv.reset()
    a, _ = model.predict(obs, deterministic=True)
    assert a.shape == (1, 12) and np.all(np.abs(a) <= 1.0)
    venv.close()


def test9_10_short_training_save_reload_resume(cfg):
    msgs = []
    t = PPOTrainer(cfg, sink=msgs.append)
    last = t.train()
    assert Path(last).exists() and Path(last).with_suffix(".json").exists()
    types = {m["type"] for m in msgs}
    assert {"episode", "update", "checkpoint", "curriculum"} <= types
    assert any(Path(m["path"]).name.startswith("ckpt_") for m in msgs if m["type"] == "checkpoint")
    # reload gives identical deterministic actions
    mm = ModelManager(cfg)
    m1, m2 = mm.load_ppo(last), mm.load_ppo(last)
    obs = np.random.default_rng(0).normal(size=(1, 52)).astype(np.float32)
    assert np.allclose(m1.predict(obs, deterministic=True)[0], m2.predict(obs, deterministic=True)[0])
    assert m1.num_timesteps >= 1024
    # resume continues the step counter and restores the curriculum state
    cfg2 = copy.deepcopy(cfg)
    msgs2 = []
    PPOTrainer(cfg2, sink=msgs2.append).train(resume=last, total_timesteps=1024 + 256)
    info = [m for m in msgs2 if m["type"] == "info"][0]
    assert info["start_timesteps"] == m1.num_timesteps
    assert mm.load_ppo(Path(cfg["paths"]["models"]) / "last_model.zip").num_timesteps > m1.num_timesteps
    assert (Path(cfg["paths"]["models"]) / "best_model.zip").exists()
    assert len(mm.list_models()) >= 3


def test_training_losses_are_real_numbers(cfg):
    msgs = []
    PPOTrainer(cfg, sink=msgs.append).train()
    upd = [m for m in msgs if m["type"] == "update" and m["value_loss"] is not None]
    assert upd and all(np.isfinite(m["value_loss"]) for m in upd)


def test_stop_event_ends_training_and_saves(cfg):
    import multiprocessing as mp
    stop = mp.Event()
    cfg["ppo"]["total_timesteps"] = 10_000_000
    msgs = []
    def sink(m):
        msgs.append(m)
        if m["type"] == "update":
            stop.set()
    t = PPOTrainer(cfg, sink=sink, stop_event=stop)
    last = t.train()
    assert Path(last).exists()
    assert [m for m in msgs if m["type"] == "checkpoint" and m.get("final")][0]["stopped_by_user"]


def _curr(cfg):
    c = copy.deepcopy(cfg)
    c["curriculum"]["min_steps_per_stage"] = 100
    return CurriculumManager(c)


def test_curriculum_is_performance_gated(cfg):
    cm = _curr(cfg)
    bad = {"survival_rate": 0.2, "vel_error": 0.9}
    good = {"survival_rate": 1.0, "vel_error": 0.01}
    for step in (1_000, 10_000_000):                         # lots of time, bad performance -> never advances
        assert cm.update(bad, step).action == "hold" and cm.idx == 0
    assert cm.update(good, 10_000_050).action == "advance" and cm.idx == 1


def test_curriculum_min_steps_and_difficulty_ladder(cfg):
    cm = _curr(cfg)
    good = {"survival_rate": 1.0, "vel_error": 0.01}
    assert cm.update(good, 10).action == "hold"             # passed but too early
    cm.idx = 3
    cm.difficulty = cm.stage["difficulty"][0]
    cm.stage_start_step = 0
    d0 = cm.difficulty
    assert cm.update(good, 1000).action == "harder" and cm.difficulty > d0


def test_curriculum_disabled_stages_are_skipped(cfg):
    c = copy.deepcopy(cfg)
    c["curriculum"]["stages"][0]["enabled"] = False
    cm = CurriculumManager(c)
    assert cm.stage["name"].startswith("2_")
    sd = cm.state_dict(); cm2 = CurriculumManager(c); cm2.load_state_dict(sd)
    assert cm2.idx == cm.idx


def test_curriculum_reaches_final_stage_and_finishes(cfg):
    cm = _curr(cfg)
    good = {"survival_rate": 1.0, "vel_error": 0.01}
    t, seen = 1000, set()
    for _ in range(100):
        d = cm.update(good, t); t += 1000; seen.add(d.action)
        if d.action == "finished":
            break
    assert cm.finished and {"advance", "harder", "finished"} <= seen and cm.idx == len(cm.stages) - 1


def test_curriculum_does_not_pass_a_robot_that_just_stands_still(cfg):
    cm = _curr(cfg)
    cm.idx = 3                                                   # stage with moving commands and a loose vel threshold
    cm.stage_start_step = 0
    standing = {"survival_rate": 1.0, "vel_error": 0.33, "cmd_speed": 0.40, "yaw_rate_error": 0.2, "cmd_wz_abs": 0.2}
    assert cm.update(standing, 10**7).action == "hold"
    walking = {"survival_rate": 1.0, "vel_error": 0.15, "cmd_speed": 0.40, "yaw_rate_error": 0.2, "cmd_wz_abs": 0.2}
    assert cm.update(walking, 10**7 + 1).action in ("harder", "advance")
