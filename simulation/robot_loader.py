"""Go2 URDF loading for PyBullet.

Responsibilities
----------------
* Resolve ``package://go2_description/...`` mesh references (search paths, recursive search).
* Build a *derived* URDF in a cache directory (the user's file is never modified):
    - mesh paths are rewritten to absolute paths;
    - if a mesh is missing, the visual falls back to that link's own collision geometry;
    - links without inertial data (``imu``, ``radar``, ``*_calflower``, ``front_camera``) are given a
      negligible mass.  PyBullet would otherwise silently assign **1 kg** to every link that has no
      <inertial> (9 kg of ghost mass on a 16 kg robot) and treat explicit ``mass=0`` links as static.
  Every such change is logged and reported by the validation routine.
* Load the body, identify the 12 actuated joints *by name* with ``getJointInfo`` (never by index).
* Compute a physically valid default standing pose by analytic 2-link IK + forward kinematics.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from utils.config import PROJECT_ROOT, resolve_path
from utils.logger import get_logger

log = get_logger(__name__)

MIN_LINK_MASS = 1e-3
MIN_LINK_INERTIA = 9.6e-6   # same values the URDF author uses for the Head_* helper links


# --------------------------------------------------------------------------------------
# Asset resolution
# --------------------------------------------------------------------------------------
class AssetResolver:
    """Finds mesh files referenced by ``package://`` / relative URDF filenames."""

    _SKIP_DIRS = {".git", "__pycache__", "node_modules", ".cache", "logs", "checkpoints", "models", "results"}

    def __init__(self, urdf_path: Path, mesh_dirs: List[Path], search_root: Path = PROJECT_ROOT):
        self.urdf_path = Path(urdf_path)
        self.mesh_dirs = [Path(d) for d in mesh_dirs]
        self.search_root = Path(search_root)
        self._cache: Dict[str, Optional[Path]] = {}
        self._index: Optional[Dict[str, Path]] = None

    def _build_index(self) -> Dict[str, Path]:
        idx: Dict[str, Path] = {}
        roots = [self.search_root, self.urdf_path.parent] + self.mesh_dirs
        for root in roots:
            if not root.exists():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in self._SKIP_DIRS]
                for fn in filenames:
                    if fn.lower().endswith((".dae", ".obj", ".stl", ".ply")):
                        idx.setdefault(fn, Path(dirpath) / fn)
        return idx

    def resolve(self, filename: str) -> Optional[Path]:
        if filename in self._cache:
            return self._cache[filename]
        rel = filename
        if filename.startswith("package://"):
            rel = filename[len("package://"):]
            rel = rel.split("/", 1)[1] if "/" in rel else rel   # drop the package name
        elif filename.startswith("file://"):
            rel = filename[len("file://"):]
        base = Path(rel).name
        candidates: List[Path] = [Path(rel)] if Path(rel).is_absolute() else []
        for d in self.mesh_dirs:
            candidates += [d / rel, d / base, d.parent / rel]
        candidates += [self.urdf_path.parent / rel, self.urdf_path.parent / base]
        found: Optional[Path] = next((c for c in candidates if c.is_file()), None)
        if found is None:
            if self._index is None:
                self._index = self._build_index()
            found = self._index.get(base)
            if found is not None:
                log.info("Mesh %s located by recursive search: %s", base, found)
        self._cache[filename] = found.resolve() if found else None
        return self._cache[filename]


@dataclass
class PreparedURDF:
    """Result of :func:`prepare_urdf`."""
    path: Path
    missing_meshes: List[str] = field(default_factory=list)
    resolved_meshes: Dict[str, str] = field(default_factory=dict)
    fallback_visual_links: List[str] = field(default_factory=list)
    inertia_patched_links: List[str] = field(default_factory=list)
    links_without_collision: List[str] = field(default_factory=list)
    visuals_loaded: bool = True


def _geometry_copy(col: ET.Element) -> ET.Element:
    """Build a <visual> element that mirrors a <collision> element."""
    vis = ET.Element("visual")
    for child in col:
        if child.tag in ("origin", "geometry"):
            vis.append(copy.deepcopy(child))
    mat = ET.SubElement(vis, "material", {"name": "collision_fallback"})
    ET.SubElement(mat, "color", {"rgba": "0.55 0.58 0.65 1"})
    return vis


def prepare_urdf(urdf_path: Path, mesh_dirs: List[Path], cache_dir: Path, load_visuals: bool = True,
                 fix_massless: bool = True) -> PreparedURDF:
    """Create the derived URDF used by PyBullet (see module docstring)."""
    urdf_path = Path(urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(
            f"URDF not found: {urdf_path}. Place go2_description.urdf in assets/ or set paths.urdf in config.yaml")
    resolver = AssetResolver(urdf_path, mesh_dirs)
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    out = PreparedURDF(path=urdf_path, visuals_loaded=load_visuals)

    for link in root.findall("link"):
        name = link.get("name", "?")
        cols = link.findall("collision")
        if not cols and link.find("inertial") is not None and name != "base":
            out.links_without_collision.append(name)
        # ---- visuals -------------------------------------------------------------------
        visuals = link.findall("visual")
        if not load_visuals:
            for v in visuals:
                link.remove(v)
        else:
            for v in visuals:
                mesh = v.find("geometry/mesh")
                if mesh is None:
                    continue
                fn = mesh.get("filename", "")
                found = resolver.resolve(fn)
                if found is not None:
                    mesh.set("filename", str(found))
                    out.resolved_meshes[fn] = str(found)
                    # The URDF carries many vendor-specific <material> children; the DAE already has its
                    # own materials, so drop them to avoid colour overrides / parser ambiguity.
                    for m in v.findall("material"):
                        v.remove(m)
                else:
                    if fn not in out.missing_meshes:
                        out.missing_meshes.append(fn)
                    link.remove(v)
                    if cols:
                        link.append(_geometry_copy(cols[0]))
                        out.fallback_visual_links.append(name)
            if link.find("visual") is None and cols:       # links that only have collision geometry
                for c in cols:
                    link.append(_geometry_copy(c))
        # ---- inertial sanity -----------------------------------------------------------
        inertial = link.find("inertial")
        mass_el = inertial.find("mass") if inertial is not None else None
        mass = float(mass_el.get("value", "0")) if mass_el is not None else 0.0
        if fix_massless and (inertial is None or mass <= 0.0):
            if inertial is not None:
                link.remove(inertial)
            inertial = ET.Element("inertial")
            ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
            ET.SubElement(inertial, "mass", {"value": str(MIN_LINK_MASS)})
            ET.SubElement(inertial, "inertia", {k: (str(MIN_LINK_INERTIA) if k in ("ixx", "iyy", "izz") else "0")
                                                for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")})
            link.insert(0, inertial)
            out.inertia_patched_links.append(name)

    if out.missing_meshes:
        log.warning("%d mesh file(s) not found (%s). Falling back to collision-geometry visuals for %d link(s). "
                    "Put the .dae files into assets/dae/ or set paths.mesh_dirs.",
                    len(out.missing_meshes), ", ".join(Path(m).name for m in out.missing_meshes),
                    len(out.fallback_visual_links))
    if out.inertia_patched_links:
        log.info("Assigned negligible mass (%g kg) to %d link(s) without inertial data: %s",
                 MIN_LINK_MASS, len(out.inertia_patched_links), ", ".join(out.inertia_patched_links))

    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = hashlib.md5((str(urdf_path.resolve()) + str(urdf_path.stat().st_mtime_ns) + str(load_visuals)
                       + str(sorted(out.resolved_meshes.items())) + str(fix_massless)).encode()).hexdigest()[:10]
    dst = cache_dir / f"go2_resolved_{tag}.urdf"
    if not dst.exists():
        fd, tmp = tempfile.mkstemp(suffix=".urdf", dir=cache_dir)   # atomic write: workers may race here
        os.close(fd)
        tree.write(tmp, encoding="utf-8", xml_declaration=True)
        os.replace(tmp, dst)
    out.path = dst
    return out


# --------------------------------------------------------------------------------------
# Robot model
# --------------------------------------------------------------------------------------
@dataclass
class JointSpec:
    name: str
    index: int
    lower: float
    upper: float
    effort: float
    velocity: float
    axis: Tuple[float, float, float]


@dataclass
class RobotModel:
    """Everything the environment needs to know about the loaded robot (indices come from PyBullet)."""
    client: int
    body_id: int
    joints: List[JointSpec]
    joint_ids: List[int]
    lower: np.ndarray
    upper: np.ndarray
    effort: np.ndarray
    vel_limit: np.ndarray
    foot_ids: List[int]
    foot_radius: float
    link_names: Dict[int, str]
    link_index: Dict[str, int]
    masses: Dict[int, float]
    inertias: Dict[int, Tuple[float, float, float]]
    default_q: np.ndarray
    default_base_z: float                 # base height of the nominal stance (feet touching z=0)
    default_height_above_feet: float
    prepared: PreparedURDF
    thigh_len: float
    calf_len: float
    leg_ids: Dict[str, List[int]] = field(default_factory=dict)       # leg -> [hip, thigh, calf] pybullet idx
    terminate_link_ids: List[int] = field(default_factory=list)
    hip_ids: List[int] = field(default_factory=list)                  # links that should never touch ground
    thigh_calf_ids: List[int] = field(default_factory=list)
    designed_overlaps: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def num_joints(self) -> int:
        return len(self.joint_ids)

    @property
    def total_mass(self) -> float:
        return float(sum(self.masses.values()))


def leg_ik(foot_x: float, stance_h: float, l1: float, l2: float, abduction: float) -> Tuple[float, float]:
    """Sagittal-plane 2-link IK for the Go2 leg (thigh/calf both rotate about +y, foot below hip).

    Returns (thigh, calf) angles such that the foot is ``stance_h`` below and ``foot_x`` in front of the thigh joint.
    """
    h_leg = stance_h / max(math.cos(abduction), 1e-3)
    d2 = foot_x ** 2 + h_leg ** 2
    cos_phi = np.clip((d2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2), -1.0, 1.0)
    phi = -math.acos(cos_phi)                                # knee bends backwards (calf joint < 0)
    psi = math.atan2(-foot_x, h_leg)
    delta = math.atan2(l2 * math.sin(phi), l1 + l2 * math.cos(phi))
    return psi - delta, phi


def load_robot(client: int, cfg: Dict[str, Any], load_visuals: bool = True,
               base_pos: Tuple[float, float, float] = (0, 0, 0.5)) -> RobotModel:
    """Load the Go2 into ``client`` and return a :class:`RobotModel`."""
    rc = cfg["robot"]
    urdf = resolve_path(cfg, "urdf")
    mesh_dirs = [PROJECT_ROOT / d if not Path(d).is_absolute() else Path(d) for d in cfg["paths"]["mesh_dirs"]]
    cache = resolve_path(cfg, "cache")
    prepared = prepare_urdf(urdf, mesh_dirs, cache, load_visuals=load_visuals, fix_massless=rc["fix_massless_links"])

    flags = p.URDF_USE_INERTIA_FROM_FILE
    if cfg["simulation"]["self_collision"]:
        flags |= p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    body = p.loadURDF(str(prepared.path), basePosition=list(base_pos), baseOrientation=[0, 0, 0, 1],
                      useFixedBase=False, flags=flags, physicsClientId=client)

    # ---- enumerate joints / links by name (indices are NOT assumed) --------------------
    n = p.getNumJoints(body, physicsClientId=client)
    link_names: Dict[int, str] = {-1: p.getBodyInfo(body, physicsClientId=client)[0].decode()}
    by_name: Dict[str, JointSpec] = {}
    for j in range(n):
        info = p.getJointInfo(body, j, physicsClientId=client)
        link_names[j] = info[12].decode()
        if info[2] == p.JOINT_REVOLUTE:
            by_name[info[1].decode()] = JointSpec(info[1].decode(), j, info[8], info[9], info[10], info[11],
                                                  tuple(info[13]))
    link_index = {v: k for k, v in link_names.items()}
    if link_names[-1] != rc["base_link"]:
        log.warning("Base link is '%s' (config says '%s')", link_names[-1], rc["base_link"])

    missing = [jn for jn in rc["joint_order"] if jn not in by_name]
    if missing:
        raise RuntimeError(f"URDF is missing expected actuated joints: {missing}. Found: {sorted(by_name)}")
    joints = [by_name[jn] for jn in rc["joint_order"]]
    foot_ids = []
    for fl in rc["foot_links"]:
        if fl not in link_index:
            raise RuntimeError(f"Foot link '{fl}' not found in URDF")
        foot_ids.append(link_index[fl])

    masses, inertias = {}, {}
    for idx in range(-1, n):
        d = p.getDynamicsInfo(body, idx, physicsClientId=client)
        masses[idx], inertias[idx] = d[0], tuple(d[2])

    shapes = p.getCollisionShapeData(body, foot_ids[0], physicsClientId=client)
    foot_radius = float(shapes[0][3][0]) if shapes and shapes[0][2] == p.GEOM_SPHERE else 0.022

    # In PyBullet the joint index equals the index of its child link.
    leg_ids = {leg: [by_name[f"{leg}_{s}_joint"].index for s in ("hip", "thigh", "calf")]
               for leg in ("FL", "FR", "RL", "RR")}

    # Segment lengths from forward kinematics of the URDF link *frames* at q = 0 (note: getJointInfo's parentFramePos is
    # relative to the parent's centre of mass, not its URDF frame, so it must not be used for this).
    for j in range(n):
        p.resetJointState(body, j, 0.0, 0.0, physicsClientId=client)
    fl = leg_ids["FL"]
    ls = p.getLinkStates(body, [fl[1], fl[2], foot_ids[0]], physicsClientId=client)
    l1 = float(np.linalg.norm(np.array(ls[1][4]) - np.array(ls[0][4])))
    l2 = float(np.linalg.norm(np.array(ls[2][4]) - np.array(ls[1][4])))

    # ---- default stance from IK -----------------------------------------------------------
    q0 = np.zeros(12)
    for li, leg in enumerate(("FL", "FR", "RL", "RR")):
        ab = rc["hip_abduction"] * (1 if leg.endswith("L") else -1)
        fx = rc["front_foot_x"] if leg.startswith("F") else rc["rear_foot_x"]
        th, ca = leg_ik(fx, rc["stance_height"], l1, l2, ab)
        q0[3 * li:3 * li + 3] = (ab, th, ca)
    lower = np.array([j.lower for j in joints]); upper = np.array([j.upper for j in joints])
    if np.any(q0 < lower) or np.any(q0 > upper):
        raise RuntimeError(f"Default pose {q0} violates joint limits")

    model = RobotModel(
        client=client, body_id=body, joints=joints, joint_ids=[j.index for j in joints], lower=lower, upper=upper,
        effort=np.array([j.effort for j in joints]), vel_limit=np.array([j.velocity for j in joints]),
        foot_ids=foot_ids, foot_radius=foot_radius, link_names=link_names, link_index=link_index,
        masses=masses, inertias=inertias, default_q=q0, default_base_z=0.0, default_height_above_feet=0.0,
        prepared=prepared, thigh_len=l1, calf_len=l2, leg_ids=leg_ids)
    model.terminate_link_ids = [link_index[n_] for n_ in rc["terminate_on_links"] if n_ in link_index] + [-1]
    model.terminate_link_ids = sorted(set(model.terminate_link_ids))
    model.hip_ids = [leg_ids[l][0] for l in leg_ids]
    model.thigh_calf_ids = [leg_ids[l][k] for l in leg_ids for k in (1, 2)]

    _configure_dynamics(model, cfg)
    _compute_standing_height(model)
    _disable_designed_overlaps(model)
    return model


def _configure_dynamics(m: RobotModel, cfg: Dict[str, Any]) -> None:
    """Dynamics parameters; joint motors are disabled (we apply torques ourselves)."""
    c, body = m.client, m.body_id
    for idx in range(-1, p.getNumJoints(body, physicsClientId=c)):
        p.changeDynamics(body, idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0,
                         restitution=0.0, lateralFriction=1.0, physicsClientId=c)
    for fid in m.foot_ids:
        p.changeDynamics(body, fid, lateralFriction=cfg["simulation"]["foot_friction"],
                         spinningFriction=0.01, rollingFriction=0.0, physicsClientId=c)
    for j, spec in zip(m.joint_ids, m.joints):
        p.changeDynamics(body, j, maxJointVelocity=spec.velocity * 1.5, physicsClientId=c)
    # disable the default velocity motors (they act as strong friction otherwise)
    p.setJointMotorControlArray(body, m.joint_ids, p.VELOCITY_CONTROL, forces=[0.0] * 12, physicsClientId=c)


def _compute_standing_height(m: RobotModel) -> None:
    """Forward kinematics of the default pose -> base height at which the lowest foot just touches z=0."""
    c = m.client
    p.resetBasePositionAndOrientation(m.body_id, [0, 0, 1.0], [0, 0, 0, 1], physicsClientId=c)
    for j, q in zip(m.joint_ids, m.default_q):
        p.resetJointState(m.body_id, j, float(q), 0.0, physicsClientId=c)
    st = p.getLinkStates(m.body_id, m.foot_ids, physicsClientId=c)
    foot_z = np.array([s[4][2] for s in st])
    m.default_base_z = float(1.0 - (foot_z.min() - m.foot_radius))
    m.default_height_above_feet = float(1.0 - foot_z.mean())
    log.info("Default stance: base z=%.3f m, base-to-foot-centre=%.3f m, foot z spread=%.4f m",
             m.default_base_z, m.default_height_above_feet, float(np.ptp(foot_z)))


def _disable_designed_overlaps(m: RobotModel) -> None:
    """Disable collisions for link pairs whose collision shapes overlap in the nominal stance.

    The Go2 URDF contains small helper collision shapes at the lower leg (``*_calflower`` ...) that interpenetrate
    the foot/calf by design. Keeping those pairs enabled would create permanent bogus "self-collisions".
    Pairs that only touch at other configurations stay enabled.
    """
    c, b = m.client, m.body_id
    p.resetBasePositionAndOrientation(b, [0, 0, 1.0], [0, 0, 0, 1], physicsClientId=c)
    for j, q in zip(m.joint_ids, m.default_q):
        p.resetJointState(b, j, float(q), 0.0, physicsClientId=c)
    p.performCollisionDetection(physicsClientId=c)
    pairs = set()
    for cp in p.getContactPoints(bodyA=b, bodyB=b, physicsClientId=c):
        a, bb = cp[3], cp[4]
        if a != bb:
            pairs.add((min(a, bb), max(a, bb)))
    for a, bb in sorted(pairs):
        p.setCollisionFilterPair(b, b, a, bb, 0, physicsClientId=c)
    m.designed_overlaps = sorted(pairs)
    if pairs:
        log.info("Disabled %d link pairs overlapping by design in the nominal stance: %s", len(pairs),
                 ", ".join(f"{m.link_names[a]}<->{m.link_names[bb]}" for a, bb in sorted(pairs)))
