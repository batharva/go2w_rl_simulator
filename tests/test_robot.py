"""Tests 1-5: URDF loading, joint identification, reset, standing stability, actuator limits."""
import numpy as np
import pybullet as p
import pytest

from simulation.physics_config import PhysicsConfig
from simulation.robot_controller import JointPDController
from simulation.robot_loader import leg_ik, load_robot
from terrain.terrain_generator import TerrainGenerator, TerrainParams
from utils.config import JOINT_ORDER
from utils.robot_validation import validate_robot


@pytest.fixture()
def world(cfg):
    cid = p.connect(p.DIRECT)
    PhysicsConfig.from_cfg(cfg).apply(cid)
    TerrainGenerator(cid).generate(TerrainParams.from_cfg(cfg["terrain"], type="flat"))
    rb = load_robot(cid, cfg, load_visuals=False)
    yield cid, rb, JointPDController(rb, cfg), cfg
    p.disconnect(cid)


def test1_urdf_loads(world):
    cid, rb, _, _ = world
    assert rb.body_id >= 0
    assert p.getBodyInfo(rb.body_id, physicsClientId=cid)[0].decode() == "base"
    assert abs(rb.masses[-1] - 6.921) < 1e-3                      # base mass from the user's URDF


def test_no_ghost_mass(world):
    """PyBullet would add 1 kg to every inertia-less link; the loader must prevent that."""
    _, rb, _, _ = world
    assert 16.0 < rb.total_mass < 16.2


def test2_twelve_actuated_joints_by_name(world):
    cid, rb, _, _ = world
    assert rb.num_joints == 12
    names = [p.getJointInfo(rb.body_id, j, physicsClientId=cid)[1].decode() for j in rb.joint_ids]
    assert names == JOINT_ORDER
    assert len(set(rb.joint_ids)) == 12
    assert all(p.getJointInfo(rb.body_id, j, physicsClientId=cid)[2] == p.JOINT_REVOLUTE for j in rb.joint_ids)
    assert rb.joint_ids != list(range(12)), "indices should come from getJointInfo, not be assumed sequential"
    assert len(rb.foot_ids) == 4
    assert np.all(rb.effort > 0) and np.all(rb.vel_limit > 0) and np.all(rb.upper > rb.lower)


def test_urdf_limits_are_used(world):
    _, rb, _, _ = world
    by = dict(zip(JOINT_ORDER, zip(rb.lower, rb.upper, rb.effort, rb.vel_limit)))
    assert by["FL_hip_joint"][:2] == pytest.approx((-1.0472, 1.0472), abs=1e-4)
    assert by["RL_thigh_joint"][1] == pytest.approx(4.5379, abs=1e-3)      # rear thigh limits differ from front
    assert by["FL_calf_joint"][2] == pytest.approx(45.43, abs=0.5)


def test3_reset_without_errors(world):
    cid, rb, ctrl, _ = world
    for _ in range(3):
        p.resetBasePositionAndOrientation(rb.body_id, [0, 0, rb.default_base_z + 0.02], [0, 0, 0, 1], physicsClientId=cid)
        for j, q in zip(rb.joint_ids, rb.default_q):
            p.resetJointState(rb.body_id, j, float(q), 0.0, physicsClientId=cid)
        for _ in range(10):
            ctrl.apply(rb.default_q)
            p.stepSimulation(physicsClientId=cid)
    q, _ = ctrl.read_state()
    assert np.all(np.isfinite(q))


def test_default_pose_ik_matches_fk(world):
    cid, rb, _, cfg = world
    p.resetBasePositionAndOrientation(rb.body_id, [0, 0, 1], [0, 0, 0, 1], physicsClientId=cid)
    for j, q in zip(rb.joint_ids, rb.default_q):
        p.resetJointState(rb.body_id, j, float(q), 0, physicsClientId=cid)
    z = np.array([s[4][2] for s in p.getLinkStates(rb.body_id, rb.foot_ids, physicsClientId=cid)])
    assert np.ptp(z) < 0.01                       # all four feet at the same height in the nominal stance
    th, ca = leg_ik(0.0, 0.3, 0.213, 0.213, 0.0)
    assert 0.7 < th < 0.9 and -1.7 < ca < -1.4


def test4_robot_stands_and_does_not_fall(world):
    cid, rb, ctrl, _ = world
    p.resetBasePositionAndOrientation(rb.body_id, [0, 0, rb.default_base_z + 0.01], [0, 0, 0, 1], physicsClientId=cid)
    for j, q in zip(rb.joint_ids, rb.default_q):
        p.resetJointState(rb.body_id, j, float(q), 0.0, physicsClientId=cid)
    for _ in range(int(3.0 / PhysicsConfig().timestep)):
        ctrl.apply(rb.default_q)
        p.stepSimulation(physicsClientId=cid)
    pos, orn = p.getBasePositionAndOrientation(rb.body_id, physicsClientId=cid)
    rpy = p.getEulerFromQuaternion(orn)
    assert pos[2] > 0.85 * rb.default_base_z
    assert abs(rpy[0]) < 0.1 and abs(rpy[1]) < 0.1
    assert np.linalg.norm(pos[:2]) < 0.1          # no drifting while standing


def test5_joint_control_and_torque_limits(world):
    cid, rb, ctrl, _ = world
    p.resetBasePositionAndOrientation(rb.body_id, [0, 0, rb.default_base_z + 0.02], [0, 0, 0, 1], physicsClientId=cid)
    rng = np.random.default_rng(0)
    peak = np.zeros(12)
    for k in range(400):
        a = rng.uniform(-1, 1, 12) if k % 20 == 0 else a
        q_des = ctrl.action_to_target(a)
        assert np.all(q_des >= rb.lower) and np.all(q_des <= rb.upper)      # targets always inside URDF limits
        tau = ctrl.apply(q_des)
        peak = np.maximum(peak, np.abs(tau))
        p.stepSimulation(physicsClientId=cid)
    assert np.all(peak <= rb.effort + 1e-9), "torque exceeded the URDF effort limit"
    assert peak.max() > 5.0                       # the controller actually produced torque
    q, _ = ctrl.read_state()
    assert np.all(q >= rb.lower - 0.05) and np.all(q <= rb.upper + 0.05)


def test_action_mapping_roundtrip(world):
    _, _, ctrl, _ = world
    a = np.random.default_rng(1).uniform(-1, 1, 12)
    assert np.allclose(ctrl.target_to_action(ctrl.action_to_target(a)), a, atol=1e-6)
    assert np.allclose(ctrl.action_to_target(np.zeros(12)), ctrl.default_q)


def test_validation_report(cfg):
    rep = validate_robot(cfg, load_visuals=True, save=False, verbose=False)
    assert rep["ok"], rep["errors"]
    assert rep["num_actuated_joints"] == 12 and rep["num_links"] > 12
    assert len(rep["missing_meshes"]) == len(set(rep["missing_meshes"]))
    assert "FL_calflower" in rep["links_without_inertial"]
    assert rep["pybullet"]["ghost_mass_check"]
