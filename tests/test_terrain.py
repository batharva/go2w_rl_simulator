"""Tests 11-12: every terrain type generates, has real collision geometry, respects parameters and is deterministic."""
import numpy as np
import pybullet as p
import pytest

from environment.go2_env import Go2QuadrupedEnv
from terrain.terrain_generator import TERRAIN_TYPES, TerrainGenerator, TerrainParams


@pytest.fixture()
def gen():
    cid = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    yield cid, TerrainGenerator(cid)
    p.disconnect(cid)


@pytest.mark.parametrize("ttype", TERRAIN_TYPES)
def test11_generate_all_types(gen, ttype):
    cid, g = gen
    t = g.generate(TerrainParams(type=ttype, difficulty=0.8, seed=11))
    assert t.body_id >= 0 and np.all(np.isfinite(t.heightmap))
    if ttype != "flat":
        assert np.ptp(t.heightmap) > 0.02, "non-flat terrain must have relief"
    # the spawn pad is flat
    c = t.heightmap.shape[0] // 2
    assert np.abs(t.heightmap[c - 5:c + 6, c - 5:c + 6]).max() < 0.02


@pytest.mark.parametrize("ttype", TERRAIN_TYPES)
def test12_collision_geometry_matches_heightmap(gen, ttype):
    """Ray-cast against the *physics* body and compare with the height map used to build it."""
    cid, g = gen
    t = g.generate(TerrainParams(type=ttype, difficulty=0.7, seed=5))
    pts = np.random.default_rng(0).uniform(-6.5, 6.5, (200, 2))
    res = p.rayTestBatch([[x, y, 5.0] for x, y in pts], [[x, y, -5.0] for x, y in pts], physicsClientId=cid)
    assert all(r[0] == t.body_id for r in res)
    z = np.array([r[3][2] for r in res])
    err = np.abs(z - t.height_at(pts[:, 0], pts[:, 1]))
    # Inside one cell the heightfield uses triangle interpolation, my lookup is bilinear; the two differ by at most the
    # cell's "twist" |h00 - h10 - h01 + h11|, so every point must satisfy err <= twist/2 (+2 mm numerical slack).
    hm, n = t.heightmap, t.heightmap.shape[0]
    g = np.clip(pts / t.resolution + (n - 1) / 2, 0, n - 1.001)
    i0, j0 = np.floor(g[:, 0]).astype(int), np.floor(g[:, 1]).astype(int)
    twist = np.abs(hm[i0, j0] - hm[i0 + 1, j0] - hm[i0, j0 + 1] + hm[i0 + 1, j0 + 1])
    assert np.all(err <= twist / 2 + 2e-3 + 1e-9), (err - twist / 2).max()
    assert np.median(err) < 1e-3


def test_a_dropped_ball_rests_on_the_terrain(gen):
    cid, g = gen
    t = g.generate(TerrainParams(type="stairs", difficulty=0.8, seed=3))
    x, y = 3.0, 0.5
    h = float(t.height_at(x, y))
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05, physicsClientId=cid)
    ball = p.createMultiBody(0.5, col, basePosition=[x, y, h + 0.6], physicsClientId=cid)
    for _ in range(480):
        p.stepSimulation(physicsClientId=cid)
    (bx, by, z), _ = p.getBasePositionAndOrientation(ball, physicsClientId=cid)
    h_final = float(t.height_at(bx, by))            # the ball may roll down a step; it must rest ON the surface
    assert h_final - 0.02 < z - 0.05 < h_final + 0.15, (z, h_final)
    assert z < h + 0.6 - 0.05                        # and it did fall (gravity + collision active)


def test_same_seed_same_terrain_different_seed_different(gen):
    _, g = gen
    a = g.build_heightmap(TerrainParams(type="uneven", seed=1, difficulty=0.8))
    b = g.build_heightmap(TerrainParams(type="uneven", seed=1, difficulty=0.8))
    c = g.build_heightmap(TerrainParams(type="uneven", seed=2, difficulty=0.8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_difficulty_scales_height(gen):
    _, g = gen
    easy = np.ptp(g.build_heightmap(TerrainParams(type="uneven", difficulty=0.2, seed=4)))
    hard = np.ptp(g.build_heightmap(TerrainParams(type="uneven", difficulty=1.0, seed=4)))
    assert hard > 2 * easy


def test_height_limits_are_respected(gen):
    _, g = gen
    hm = g.build_heightmap(TerrainParams(type="uneven", difficulty=1.0, max_height=0.05, min_height=-0.05, seed=4))
    assert hm.max() <= 0.0501 and hm.min() >= -0.0501


def test_friction_and_restitution_applied(gen):
    cid, g = gen
    t = g.generate(TerrainParams(type="rough", friction=0.33, restitution=0.4))
    d = p.getDynamicsInfo(t.body_id, -1, physicsClientId=cid)
    assert d[1] == pytest.approx(0.33) and d[5] == pytest.approx(0.4)


def test_invalid_parameters_rejected():
    for bad in (dict(type="lava"), dict(difficulty=1.5), dict(min_height=0.5, max_height=0.1), dict(slope_angle=80)):
        with pytest.raises(ValueError):
            TerrainParams(**bad).validate()


@pytest.mark.parametrize("ttype", ["uneven", "stairs", "slopes", "obstacles", "mixed", "stepping", "rough"])
def test_robot_spawns_and_stands_on_every_terrain(cfg, ttype):
    cfg["randomization"]["enabled"] = False
    cfg["terrain"].update(type=ttype, difficulty=0.8, regen_every_episodes=0)
    e = Go2QuadrupedEnv(cfg, seed=0)
    e.reset(seed=0)
    for _ in range(60):
        _, _, term, _, info = e.step(np.zeros(12, np.float32))
        assert not term, info
    assert e.telemetry()["height"] > 0.2
    e.close()


def test_environment_applies_gui_terrain_changes(cfg):
    e = Go2QuadrupedEnv(cfg, seed=0)
    e.reset()
    info = e.regenerate_terrain(seed=77, type="slopes", difficulty=0.9)
    assert info["type"] == "slopes" and e.sim.terrain.params.seed == 77
    e.close()
