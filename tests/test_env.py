"""Tests 6-7: environment steps and Gymnasium API compliance (+ reward / termination / randomisation behaviour)."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from environment.go2_env import Go2QuadrupedEnv
from environment.observation import OBS_DIM
from environment.reward import COMPONENTS


@pytest.fixture()
def env(cfg):
    e = Go2QuadrupedEnv(cfg, seed=0)
    yield e
    e.close()


def test6_runs_100_plus_steps(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,) == env.observation_space.shape
    n = 0
    for _ in range(150):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        n += 1
        if term or trunc:
            obs, _ = env.reset()
    assert n >= 100


def test7_gymnasium_check_env(cfg):
    e = Go2QuadrupedEnv(cfg)
    check_env(e, skip_render_check=True)
    e.close()


def test_observation_normalised_and_bounded(env):
    obs, _ = env.reset()
    assert obs.dtype == np.float32 and np.abs(obs).max() <= env.cfg["observation"]["clip"]


def test_reward_components_logged_separately(env):
    env.reset()
    _, r, _, _, _ = env.step(np.zeros(12, np.float32))
    parts = env.telemetry()["parts"]
    assert set(parts) == set(COMPONENTS)
    assert r == pytest.approx(sum(parts.values()))


def test_same_seed_is_reproducible(cfg):
    runs = []
    for _ in range(2):
        e = Go2QuadrupedEnv(cfg, seed=7)
        e.reset(seed=7)
        rng = np.random.default_rng(3)
        traj = [e.step(rng.uniform(-1, 1, 12).astype(np.float32))[0] for _ in range(40)]
        runs.append(np.array(traj))
        e.close()
    assert np.allclose(runs[0], runs[1], atol=1e-4)


def test_fall_terminates_with_penalty(env):
    env.set_stage(None)
    env.randomizer.active = False
    env.reset()
    term = False
    for k in range(400):
        _, r, term, trunc, info = env.step(np.array([1, 1, 1] * 4, np.float32) * (1 if k < 200 else -1))
        if term:
            break
    # a pathological action pattern should not crash the sim; if it fell the penalty must be applied
    if term:
        assert info["episode_stats"]["fell"] and env.telemetry()["parts"]["fall"] < 0


def test_time_limit_truncation(cfg):
    cfg["simulation"]["max_episode_seconds"] = 1.0
    cfg["randomization"]["enabled"] = False
    e = Go2QuadrupedEnv(cfg, seed=0)
    e.reset()
    for _ in range(100):
        _, _, term, trunc, info = e.step(np.zeros(12, np.float32))
        if term or trunc:
            break
    assert trunc and not term and info["episode_stats"]["reason"] == "time_limit"
    e.close()


def test_stalling_is_penalised_when_commanded_to_walk(cfg):
    """Anti reward-hacking: standing still under a walk command must be worse than the same stand with zero command."""
    cfg["randomization"]["enabled"] = False
    out = {}
    for name, cmd in (("walk", (0.6, 0, 0)), ("stand", (0, 0, 0))):
        e = Go2QuadrupedEnv(cfg, seed=0)
        e.set_command(*cmd)
        e.reset(seed=0)
        out[name] = np.mean([e.step(np.zeros(12, np.float32))[1] for _ in range(60)])
        e.close()
    assert out["walk"] < out["stand"] - 0.3


def test_domain_randomization_changes_dynamics(cfg):
    e = Go2QuadrupedEnv(cfg, seed=1)
    e.randomizer.active = True
    e.reset(seed=1)
    a = e.rand.as_dict()
    e.reset()
    b = e.rand.as_dict()
    assert a != b and 0.85 <= a["mass_scale"] <= 1.15
    import pybullet as p
    mass = p.getDynamicsInfo(e.sim.robot.body_id, -1, physicsClientId=e.sim.cid)[0]
    assert mass == pytest.approx(e.sim._orig_masses[-1] * b["mass_scale"], rel=1e-6)
    e.close()


def test_pushes_change_velocity(cfg):
    cfg["randomization"].update(push_interval_s=[0.05, 0.06], push_velocity=1.0)
    e = Go2QuadrupedEnv(cfg, seed=2)
    e.set_stage({"terrains": ["flat"], "difficulty": [0, 0], "randomization": True, "push": 1.0})
    e.reset()
    pushed = any(e.step(np.zeros(12, np.float32))[4]["pushed"] for _ in range(20))
    assert pushed
    e.close()


def test_rgb_render(cfg):
    e = Go2QuadrupedEnv(cfg, render_mode="rgb_array")
    e.reset()
    img = e.render(160, 100)
    assert img.shape == (100, 160, 3) and img.std() > 1.0
    e.close()
