"""Shared fixtures. Tests use small configs and temporary output directories."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from utils.config import load_config  # noqa: E402


@pytest.fixture()
def cfg(tmp_path):
    return load_config(overrides={
        "paths": {"models": str(tmp_path / "models"), "checkpoints": str(tmp_path / "ckpt"),
                  "logs": str(tmp_path / "logs"), "results": str(tmp_path / "results")},
        "simulation": {"max_episode_seconds": 4.0},
        "ppo": {"n_envs": 1, "n_steps": 128, "batch_size": 64, "n_epochs": 2, "total_timesteps": 1024,
                "checkpoint_freq": 512, "snapshot_freq": 256},
        "curriculum": {"eval_freq": 512, "eval_episodes": 1, "eval_episode_seconds": 2.0, "min_steps_per_stage": 0},
        "evaluation": {"episodes_per_scenario": 1, "episode_seconds": 2.0},
    })
