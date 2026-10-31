"""Terrain generation for PyBullet.

Every terrain is a *real collision* heightfield (``GEOM_HEIGHTFIELD``), not only a visual mesh.  Flat terrain uses a
large box.  A matching visual mesh (with checker texture) is created only when ``with_visual`` is set (GUI / rendering),
so headless training does not pay for it.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pybullet as p

from terrain.common import pad_mask
from terrain.flat import generate_flat
from terrain.mixed import generate_mixed
from terrain.obstacles import generate_obstacles, generate_stepping
from terrain.slopes import generate_slopes
from terrain.stairs import generate_stairs
from terrain.uneven import generate_rough, generate_uneven
from utils.logger import get_logger

log = get_logger(__name__)

TERRAIN_TYPES = ["flat", "uneven", "stepping", "stairs", "slopes", "rough", "obstacles", "mixed"]

_GENERATORS: Dict[str, Callable] = {
    "flat": generate_flat, "uneven": generate_uneven, "stepping": generate_stepping, "stairs": generate_stairs,
    "slopes": generate_slopes, "rough": generate_rough, "obstacles": generate_obstacles, "mixed": generate_mixed,
}
_SELF_PADDED = {"flat", "stairs", "slopes"}      # these generators already keep the spawn pad flat


@dataclass
class TerrainParams:
    type: str = "flat"
    size: float = 16.0
    resolution: float = 0.08
    difficulty: float = 0.3
    max_height: float = 0.15
    min_height: float = 0.0
    roughness: float = 0.5
    obstacle_density: float = 0.6
    obstacle_size: float = 0.25
    slope_angle: float = 15.0
    stair_height: float = 0.12
    stair_width: float = 0.32
    friction: float = 0.9
    restitution: float = 0.0
    seed: int = 0
    spawn_pad: float = 1.5

    @classmethod
    def from_cfg(cls, tcfg: Dict[str, Any], **overrides: Any) -> "TerrainParams":
        fields = cls.__dataclass_fields__
        d = {k: v for k, v in tcfg.items() if k in fields}
        d.update({k: v for k, v in overrides.items() if k in fields})
        return cls(**d)

    def validate(self) -> None:
        if self.type not in TERRAIN_TYPES:
            raise ValueError(f"Unknown terrain type '{self.type}'. Valid: {TERRAIN_TYPES}")
        if not 0.0 <= self.difficulty <= 1.0:
            raise ValueError("terrain difficulty must be in [0, 1]")
        if self.min_height > self.max_height:
            raise ValueError("terrain min_height must be <= max_height")
        if self.resolution < 0.02 or self.size / self.resolution > 600:
            raise ValueError("terrain resolution too fine for the arena size (max 600 cells per side)")
        if not 0.0 <= self.friction <= 5.0 or not 0.0 <= self.restitution <= 1.0:
            raise ValueError("friction must be in [0, 5] and restitution in [0, 1]")
        if not 0.0 <= self.slope_angle <= 40.0:
            raise ValueError("slope_angle must be in [0, 40] degrees")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Terrain:
    """A generated terrain: PyBullet body + the height map used to build it."""
    body_id: int
    params: TerrainParams
    heightmap: np.ndarray             # [ix, iy] (m); the map centre is the world origin
    resolution: float
    z_offset: float
    visual_id: int = -1

    @property
    def half_size(self) -> float:
        return (self.heightmap.shape[0] - 1) * self.resolution / 2.0

    def height_at(self, x, y) -> np.ndarray:
        """Bilinear terrain height at world (x, y) (clamped at the border)."""
        hm = self.heightmap
        nx, ny = hm.shape
        gx = np.clip(np.asarray(x, dtype=float) / self.resolution + (nx - 1) / 2.0, 0, nx - 1 - 1e-9)
        gy = np.clip(np.asarray(y, dtype=float) / self.resolution + (ny - 1) / 2.0, 0, ny - 1 - 1e-9)
        x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
        fx, fy = gx - x0, gy - y0
        return (hm[x0, y0] * (1 - fx) * (1 - fy) + hm[x0 + 1, y0] * fx * (1 - fy)
                + hm[x0, y0 + 1] * (1 - fx) * fy + hm[x0 + 1, y0 + 1] * fx * fy)

    def summary(self) -> Dict[str, float]:
        return {"min": float(self.heightmap.min()), "max": float(self.heightmap.max()),
                "std": float(self.heightmap.std())}


def _checker_texture(cache_dir: Path, size_m: float) -> str:
    """One texture covering the whole arena (UVs stay in [0, 1]; the software renderer cannot wrap UVs).

    1 m checker squares with thin 0.25 m grid lines, so relief and robot speed are easy to judge visually.
    """
    px_per_m = 8
    n = int(round(size_m * px_per_m))
    path = cache_dir / f"terrain_checker_{int(round(size_m))}m_v2.png"
    if not path.exists():
        import matplotlib.image as mpimg
        m = np.arange(n) // px_per_m
        base = np.where((m[:, None] + m[None, :]) % 2 == 0, 0.74, 0.58)
        rgb = np.stack([base * 0.80, base * 0.90, base * 0.84], -1)
        cache_dir.mkdir(parents=True, exist_ok=True)
        mpimg.imsave(str(path), np.clip(rgb, 0, 1))
    return str(path)


class TerrainGenerator:
    TEX_SIZE = 16.0
    """Creates / replaces the terrain body of one PyBullet client."""

    def __init__(self, client: int, with_visual: bool = False, cache_dir: Optional[Path] = None):
        self.client = client
        self.with_visual = with_visual
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache")
        self.current: Optional[Terrain] = None

    def clear(self) -> None:
        self.clear_visual_body()
        if self.current is not None:
            try:
                p.removeBody(self.current.body_id, physicsClientId=self.client)
            except p.error:
                pass
            self.current = None

    def build_heightmap(self, params: TerrainParams) -> np.ndarray:
        params.validate()
        rng = np.random.default_rng(int(params.seed) & 0x7FFFFFFF)
        n = int(round(params.size / params.resolution))
        n += 1 - (n % 2)                                   # odd => a cell sits exactly at (0, 0)
        hm = _GENERATORS[params.type]((n, n), params.resolution, params, rng).astype(np.float64)
        if params.type not in _SELF_PADDED:
            hm = hm * pad_mask(hm.shape, params.resolution, params.spawn_pad)
        return hm

    def generate(self, params: TerrainParams) -> Terrain:
        hm = self.build_heightmap(params)
        self.clear()
        c = self.client
        res = params.resolution
        nx, ny = hm.shape
        visual_id = -1
        if float(np.ptp(hm)) < 1e-9:                      # flat: big box, top surface at z = 0
            half = params.size / 2.0 + 1.0
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, 0.5], physicsClientId=c)
            z_off = -0.5
            body = p.createMultiBody(0, col, basePosition=[0, 0, z_off], physicsClientId=c)
        else:
            z_off = float((hm.max() + hm.min()) / 2.0)    # Bullet centres the heightfield on its AABB
            col = p.createCollisionShape(
                p.GEOM_HEIGHTFIELD, meshScale=[res, res, 1.0], heightfieldData=hm.T.ravel().tolist(), heightfieldTextureScaling=params.size / 2.0,
                numHeightfieldRows=nx, numHeightfieldColumns=ny, flags=p.GEOM_CONCAVE_INTERNAL_EDGE,
                physicsClientId=c)
            body = p.createMultiBody(0, col, basePosition=[0, 0, z_off], physicsClientId=c)
        p.changeDynamics(body, -1, lateralFriction=params.friction, restitution=params.restitution,
                         physicsClientId=c)
        if self.with_visual:
            visual_id = self._add_visual(body, hm, res, z_off)
        self.current = Terrain(body, params, hm, res, z_off, visual_id)
        return self.current

    def set_surface(self, friction: float, restitution: Optional[float] = None) -> None:
        """Change friction/restitution of the current terrain without regenerating it."""
        if self.current is None:
            return
        kw = {"lateralFriction": friction}
        if restitution is not None:
            kw["restitution"] = restitution
        p.changeDynamics(self.current.body_id, -1, physicsClientId=self.client, **kw)
        self.current.params.friction = friction

    # ------------------------------------------------------------------ visuals
    def _add_visual(self, body: int, hm: np.ndarray, res: float, z_off: float) -> int:
        """PyBullet already renders heightfield / box collision shapes; here we only colour and texture them."""
        c = self.client
        try:
            tex = p.loadTexture(_checker_texture(self.cache_dir, self.TEX_SIZE), physicsClientId=c)
            p.changeVisualShape(body, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1],
                                specularColor=[0.02, 0.02, 0.02], physicsClientId=c)
        except p.error as e:                                # pragma: no cover
            log.warning("Terrain texture could not be applied: %s", e)
        return body

    def clear_visual_body(self) -> None:   # kept for API compatibility (no companion body any more)
        return None
