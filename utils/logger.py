"""Central logging setup used by every module and process."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO, log_file: Optional[str | Path] = None) -> None:
    """Configure the root logger (idempotent; safe to call in every process)."""
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if called repeatedly (spawned workers call it too).
    if not any(getattr(h, "_go2", False) for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
        sh._go2 = True  # type: ignore[attr-defined]
        root.addHandler(sh)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(getattr(h, "_go2_file", None) == str(path) for h in root.handlers):
            fh = logging.FileHandler(path)
            fh.setFormatter(logging.Formatter(_FMT))
            fh._go2_file = str(path)  # type: ignore[attr-defined]
            root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
