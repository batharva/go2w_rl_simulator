"""Robot validation: parses the URDF, loads it into PyBullet and reports everything the training depends on."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.config import PROJECT_ROOT, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def _vec(s: Optional[str]) -> List[float]:
    return [float(v) for v in s.split()] if s else []


def parse_urdf(urdf: Path, mesh_dirs: List[Path]) -> Dict[str, Any]:
    """Static (XML only) analysis of the URDF."""
    from simulation.robot_loader import AssetResolver
    root = ET.parse(urdf).getroot()
    resolver = AssetResolver(urdf, mesh_dirs)
    links, joints, missing, found = {}, {}, [], {}
    for l in root.findall("link"):
        inert = l.find("inertial")
        mass = float(inert.find("mass").get("value")) if inert is not None and inert.find("mass") is not None else None
        inertia = dict(inert.find("inertia").attrib) if inert is not None and inert.find("inertia") is not None else None
        com = _vec(inert.find("origin").get("xyz")) if inert is not None and inert.find("origin") is not None else None
        cols = [c.find("geometry")[0].tag for c in l.findall("collision")]
        links[l.get("name")] = {"mass": mass, "com": com, "inertia": {k: float(v) for k, v in (inertia or {}).items()},
                                "collision": cols, "has_visual": l.find("visual") is not None}
        for m in l.findall("visual/geometry/mesh"):
            fn = m.get("filename", "")
            r = resolver.resolve(fn)
            (found.__setitem__(fn, str(r)) if r else (missing.append(fn) if fn not in missing else None))
    for j in root.findall("joint"):
        lim = j.find("limit")
        joints[j.get("name")] = {
            "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
            "axis": _vec(j.find("axis").get("xyz")) if j.find("axis") is not None else None,
            "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
            "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            "effort": float(lim.get("effort")) if lim is not None and lim.get("effort") else None,
            "velocity": float(lim.get("velocity")) if lim is not None and lim.get("velocity") else None}
    return {"links": links, "joints": joints, "missing_meshes": missing, "resolved_meshes": found}


def validate_robot(cfg: Dict[str, Any], load_visuals: bool = True, hold_seconds: float = 1.5,
                   save: bool = True, verbose: bool = True) -> Dict[str, Any]:
    """Run the full validation. Returns a JSON-serialisable report with ``ok`` and ``errors`` / ``warnings``."""
    import pybullet as p
    from simulation.physics_config import PhysicsConfig
    from simulation.robot_controller import JointPDController
    from simulation.robot_loader import load_robot
    from terrain.terrain_generator import TerrainGenerator, TerrainParams

    errors: List[str] = []
    warnings: List[str] = []
    urdf = resolve_path(cfg, "urdf")
    mesh_dirs = [PROJECT_ROOT / d if not Path(d).is_absolute() else Path(d) for d in cfg["paths"]["mesh_dirs"]]
    static = parse_urdf(urdf, mesh_dirs)
    links, joints = static["links"], static["joints"]
    revolute = {n: j for n, j in joints.items() if j["type"] == "revolute"}
    report: Dict[str, Any] = {
        "urdf": str(urdf), "num_links": len(links), "num_joints": len(joints), "num_actuated_joints": len(revolute),
        "missing_meshes": static["missing_meshes"], "resolved_meshes": static["resolved_meshes"],
        "links_without_collision": [n for n, l in links.items() if not l["collision"] and (l["mass"] or 0) > 0],
        "links_without_inertial": [n for n, l in links.items() if not l["mass"]],
        "urdf_total_mass": float(sum((l["mass"] or 0.0) for l in links.values())),
        "base_mass": links.get(cfg["robot"]["base_link"], {}).get("mass"),
    }
    if len(revolute) != 12:
        errors.append(f"Expected 12 revolute joints, found {len(revolute)}")
    if static["missing_meshes"]:
        warnings.append(f"{len(static['missing_meshes'])} mesh file(s) missing -> collision-geometry visual fallback "
                        f"({', '.join(Path(m).name for m in static['missing_meshes'])})")
    if report["links_without_inertial"]:
        warnings.append("Links without inertial data get a negligible mass (otherwise PyBullet would assign 1 kg): "
                        + ", ".join(report["links_without_inertial"]))

    # ---- PyBullet side ----------------------------------------------------------------
    cid = p.connect(p.DIRECT)
    try:
        PhysicsConfig.from_cfg(cfg).apply(cid)
        tg = TerrainGenerator(cid, with_visual=False)
        tg.generate(TerrainParams.from_cfg(cfg["terrain"], type="flat"))
        rb = load_robot(cid, cfg, load_visuals=load_visuals)
        ctrl = JointPDController(rb, cfg)
        report["pybullet"] = {
            "num_links_pybullet": p.getNumJoints(rb.body_id, physicsClientId=cid) + 1,
            "total_mass": rb.total_mass,
            "ghost_mass_check": abs(rb.total_mass - report["urdf_total_mass"]) < 0.05,
            "joints": [{"name": j.name, "pybullet_index": j.index, "lower": j.lower, "upper": j.upper,
                        "max_effort": j.effort, "max_velocity": j.velocity, "axis": list(j.axis)} for j in rb.joints],
            "foot_links": {rb.link_names[i]: i for i in rb.foot_ids},
            "default_pose": dict(zip([j.name for j in rb.joints], [round(float(v), 4) for v in rb.default_q])),
            "default_base_height": rb.default_base_z,
            "designed_overlap_pairs_disabled": [f"{rb.link_names[a]}<->{rb.link_names[b]}" for a, b in rb.designed_overlaps],
            "inertial": {rb.link_names[i]: {"mass": rb.masses[i], "inertia_diag": list(rb.inertias[i])}
                         for i in rb.masses},
        }
        if not report["pybullet"]["ghost_mass_check"]:
            errors.append(f"PyBullet total mass {rb.total_mass:.3f} kg differs from URDF {report['urdf_total_mass']:.3f} kg")
        if sorted(j.index for j in rb.joints) != sorted(set(j.index for j in rb.joints)):
            errors.append("duplicate joint indices")

        # ---- stand test: PD hold of the default pose for ``hold_seconds`` ------------------
        pc = PhysicsConfig.from_cfg(cfg)
        p.resetBasePositionAndOrientation(rb.body_id, [0, 0, rb.default_base_z + 0.01], [0, 0, 0, 1], physicsClientId=cid)
        for j, q in zip(rb.joint_ids, rb.default_q):
            p.resetJointState(rb.body_id, j, float(q), 0.0, physicsClientId=cid)
        hmax, taumax, selfc = 0.0, np.zeros(12), 0
        n = int(hold_seconds / pc.timestep)
        for k in range(n):
            q, qd = ctrl.read_state()
            tau = ctrl.apply(rb.default_q, q, qd)
            p.stepSimulation(physicsClientId=cid)
            taumax = np.maximum(taumax, np.abs(tau))
        pos, orn = p.getBasePositionAndOrientation(rb.body_id, physicsClientId=cid)
        rpy = p.getEulerFromQuaternion(orn)
        for cp in p.getContactPoints(bodyA=rb.body_id, bodyB=rb.body_id, physicsClientId=cid):
            selfc += 1
        report["stand_test"] = {"seconds": hold_seconds, "final_base_z": pos[2], "roll": rpy[0], "pitch": rpy[1],
                                "self_contacts": selfc, "max_torque_per_joint": taumax.round(2).tolist(),
                                "torque_within_limits": bool(np.all(taumax <= rb.effort + 1e-6))}
        if pos[2] < 0.6 * rb.default_base_z or abs(rpy[0]) > 0.3 or abs(rpy[1]) > 0.3:
            errors.append(f"Robot does not hold the default stance (z={pos[2]:.3f}, roll={rpy[0]:.2f}, pitch={rpy[1]:.2f})")
        if selfc:
            warnings.append(f"{selfc} self-contact(s) in the nominal stance")
    except Exception as e:   # noqa: BLE001 - report instead of crashing the validator
        errors.append(f"PyBullet load failed: {type(e).__name__}: {e}")
        log.exception("Robot validation failed")
    finally:
        p.disconnect(cid)

    report["errors"], report["warnings"], report["ok"] = errors, warnings, not errors
    report["static"] = {"links": links, "joints": joints}
    if save:
        out = resolve_path(cfg, "results") / "robot_validation.json"
        out.write_text(json.dumps(report, indent=2, default=float))
    if verbose:
        print_report(report)
    return report


def print_report(r: Dict[str, Any]) -> None:
    pb = r.get("pybullet", {})
    print("=" * 78)
    print(f"ROBOT VALIDATION  ({r['urdf']})")
    print("=" * 78)
    print(f"links: {r['num_links']}   joints: {r['num_joints']}   actuated (revolute): {r['num_actuated_joints']}")
    print(f"URDF total mass: {r['urdf_total_mass']:.3f} kg   base mass: {r['base_mass']} kg   "
          f"PyBullet total mass: {pb.get('total_mass', float('nan')):.3f} kg")
    print(f"{'joint':<16}{'pyb idx':>8}{'lower':>9}{'upper':>9}{'effort':>9}{'vel':>8}")
    for j in pb.get("joints", []):
        print(f"{j['name']:<16}{j['pybullet_index']:>8}{j['lower']:>9.3f}{j['upper']:>9.3f}{j['max_effort']:>9.2f}"
              f"{j['max_velocity']:>8.2f}")
    print(f"missing meshes ({len(r['missing_meshes'])}): {[Path(m).name for m in r['missing_meshes']] or 'none'}")
    print(f"links with mass but no collision geometry: {r['links_without_collision'] or 'none'}")
    print(f"links without inertial data (patched): {r['links_without_inertial'] or 'none'}")
    if "default_pose" in pb:
        print(f"default base height: {pb['default_base_height']:.3f} m;  stand test: {r['stand_test']}")
    for w in r["warnings"]:
        print("WARNING:", w)
    for e in r["errors"]:
        print("ERROR:", e)
    print("RESULT:", "OK" if r["ok"] else "FAILED")
