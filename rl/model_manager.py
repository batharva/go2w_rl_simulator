"""Checkpoint / model files with JSON side-cars (training config, metrics, curriculum state)."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def sidecar(path: Path) -> Path:
    return Path(path).with_suffix(".json")


def atomic_save(model, path: Path) -> None:
    """Save an SB3 model without ever leaving a half-written zip (the GUI polls these files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=path.parent)
    os.close(fd)
    model.save(tmp)
    os.replace(tmp, path)


class ModelManager:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.models_dir = resolve_path(cfg, "models")
        self.ckpt_dir = resolve_path(cfg, "checkpoints")

    # ------------------------------------------------------------------ saving
    def save(self, model, path: Path, meta: Optional[Dict[str, Any]] = None) -> Path:
        path = Path(path)
        atomic_save(model, path)
        info = {"file": path.name, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_timesteps": int(model.num_timesteps), "config": self.cfg}
        if meta:
            info.update(meta)
        sidecar(path).write_text(json.dumps(info, indent=2, default=str))
        return path

    def save_checkpoint(self, model, meta: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(model, self.ckpt_dir / f"ckpt_{int(model.num_timesteps):010d}.zip", meta)

    def save_best(self, model, meta: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(model, self.models_dir / "best_model.zip", meta)

    def save_last(self, model, meta: Optional[Dict[str, Any]] = None) -> Path:
        return self.save(model, self.models_dir / "last_model.zip", meta)

    # ------------------------------------------------------------------ queries
    def list_models(self) -> List[Dict[str, Any]]:
        """All checkpoints and models, newest first, with the metadata from their side-cars."""
        out = []
        for folder, kind in ((self.ckpt_dir, "checkpoint"), (self.models_dir, "model")):
            for z in sorted(folder.glob("*.zip")):
                meta: Dict[str, Any] = {}
                sc = sidecar(z)
                if sc.exists():
                    try:
                        meta = json.loads(sc.read_text())
                    except json.JSONDecodeError:
                        pass
                out.append({"path": str(z), "name": z.name, "kind": kind, "mtime": z.stat().st_mtime,
                            "timesteps": meta.get("num_timesteps"), "stage": meta.get("stage"),
                            "difficulty": meta.get("difficulty"), "score": meta.get("eval_score"),
                            "saved_at": meta.get("saved_at"), "meta": meta})
        return sorted(out, key=lambda m: m["mtime"], reverse=True)

    def latest_checkpoint(self) -> Optional[Path]:
        cks = sorted(self.ckpt_dir.glob("ckpt_*.zip"))
        return cks[-1] if cks else None

    @staticmethod
    def read_meta(path: str | Path) -> Dict[str, Any]:
        sc = sidecar(Path(path))
        return json.loads(sc.read_text()) if sc.exists() else {}

    @staticmethod
    def load_ppo(path: str | Path, device: str = "cpu", env=None):
        from stable_baselines3 import PPO
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        return PPO.load(str(p), env=env, device=device)

    def copy_to(self, src: str | Path, dst: str | Path) -> None:
        shutil.copy2(src, dst)
        if sidecar(Path(src)).exists():
            shutil.copy2(sidecar(Path(src)), sidecar(Path(dst)))
