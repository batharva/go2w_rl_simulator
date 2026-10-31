"""Live training analytics.

Matplotlib figures are rendered with the Agg backend in a *worker thread* from a snapshot of the data; the GUI thread
only blits the finished image.  Plot updates therefore never block the Qt event loop (rendering 8 axes takes
0.3-0.5 s on a slow CPU).
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QTabWidget, QVBoxLayout, QWidget

from gui.styles import ACCENT, BAD, GOOD, MPL_RC, PANEL, WARN

MAX_POINTS = 1500
HISTORY = 60_000          # newest points kept per series


def _thin(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) <= MAX_POINTS:
        return x, y
    s = int(np.ceil(len(x) / MAX_POINTS))
    return x[::s], y[::s]


def _smooth(y: np.ndarray, w: int) -> np.ndarray:
    if len(y) < 3:
        return y
    w = min(w, len(y))
    return np.convolve(np.pad(y, (w - 1, 0), mode="edge"), np.ones(w) / w, mode="valid")


def draw_curves(snap: Dict[str, Any], w: int, h: int) -> np.ndarray:
    """Render the 2x4 training-curve figure into an RGBA array (thread-safe: private Figure)."""
    with matplotlib.rc_context(MPL_RC):
        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        a = fig.subplots(2, 4).ravel()
        e, u = snap["ep"], snap["upd"]
        if len(e["t"]):
            t, r = _thin(e["t"], e["ret"]); a[0].plot(t, r, color=ACCENT, lw=0.6, alpha=0.8); a[0].set_title("Episode reward")
            t, m = _thin(e["t"], e["mean"]); a[1].plot(t, m, color=GOOD, lw=1.2); a[1].set_title("Mean episode reward (100)")
            t, l = _thin(e["t"], e["len"]); a[2].plot(t, l, color=WARN, lw=0.6); a[2].set_title("Episode length [steps]")
            t, v = _thin(e["t"], e["vel"]); a[6].plot(t, v, color=BAD, lw=0.5, alpha=0.5)
            a[6].plot(t, _smooth(v, 30), color=BAD, lw=1.2); a[6].set_title("Velocity tracking error [m/s]")
        if len(u["t"]):
            a[3].plot(u["t"], u["policy_loss"], color=ACCENT); a[3].set_title("Policy loss")
            a[4].plot(u["t"], u["value_loss"], color=WARN); a[4].set_title("Value loss"); a[4].set_yscale("symlog")
            a[5].plot(u["t"], u["entropy_loss"], color=GOOD); a[5].set_title("Entropy loss")
        a[7].set_title("Training progress / curriculum")
        if snap["stage_t"]:
            a[7].step(snap["stage_t"], snap["stage_i"], where="post", color=ACCENT, label="stage")
            a[7].plot(snap["stage_t"], np.array(snap["stage_d"]) * 5, color=WARN, lw=1, label="difficulty×5")
            a[7].legend(loc="upper left")
        if len(e["t"]) and snap["target"]:
            a[7].text(0.98, 0.05, f"{e['t'][-1] / snap['target']:.1%} of target", transform=a[7].transAxes, ha="right", fontsize=8)
        for ax in a:
            ax.set_xlabel("timesteps")
        fig.subplots_adjust(left=0.05, right=0.99, top=0.9, bottom=0.2, wspace=0.28, hspace=0.95)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()


def draw_breakdown(snap: Dict[str, Any], w: int, h: int) -> np.ndarray:
    with matplotlib.rc_context(MPL_RC):
        fig = Figure(figsize=(w / 100, h / 100), dpi=100)
        canvas = FigureCanvasAgg(fig)
        axs = fig.subplots(1, 2)
        comps = snap["comps"]
        if comps:
            names = sorted(comps, key=lambda k: -abs(np.mean(comps[k][-50:])))
            groups = (([k for k in names if np.mean(comps[k][-50:]) >= 0][:6], "Reward terms (per step, smoothed)"),
                      ([k for k in names if np.mean(comps[k][-50:]) < 0][:8], "Penalty terms (per step, smoothed)"))
            for ax, (ks, title) in zip(axs, groups):
                for k in ks:
                    x, y = _thin(np.arange(len(comps[k])), comps[k])
                    ax.plot(x, _smooth(y, 20), label=k, lw=1.1)
                ax.set_title(title); ax.set_xlabel("episode")
                if ks:
                    ax.legend(loc="best", ncol=2)
        fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.2, wspace=0.2)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba()).copy()


class _Bridge(QObject):
    done = Signal(int, object)          # tab index, rgba array


class AnalyticsPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.labels = [QLabel("waiting for data…"), QLabel("waiting for data…")]
        for l in self.labels:
            l.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            l.setMinimumSize(200, 120)
            l.setStyleSheet(f"background:{PANEL};color:#8794a7;")
        self.tabs = QTabWidget()
        self.tabs.addTab(self.labels[0], "Training curves")
        self.tabs.addTab(self.labels[1], "Reward breakdown")
        self.tabs.currentChanged.connect(lambda _: self._mark())
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.tabs)
        self.setMinimumHeight(250)
        self.target_steps = 0
        self.bridge = _Bridge()
        self.bridge.done.connect(self._show)
        self._busy = False
        self.clear()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1500)

    def clear(self) -> None:
        self.ep = defaultdict(list)
        self.upd = defaultdict(list)
        self.comps: Dict[str, List[float]] = defaultdict(list)
        self.stage_t: List[float] = []
        self.stage_i: List[float] = []
        self.stage_d: List[float] = []
        self.dirty = True

    def _mark(self) -> None:
        self.dirty = True

    # ------------------------------------------------------------------ data in
    def add_episode(self, m: Dict[str, Any]) -> None:
        e = self.ep
        e["t"].append(m["timesteps"]); e["ret"].append(m["return"]); e["len"].append(m["length"])
        e["vel"].append(m["vel_error"]); e["mean"].append(m["mean_return"])
        for k, v in m["components"].items():
            self.comps[k].append(v)
        self.dirty = True

    def add_update(self, m: Dict[str, Any]) -> None:
        u = self.upd
        u["t"].append(m["timesteps"])
        for k in ("policy_loss", "value_loss", "entropy_loss", "fps"):
            u[k].append(m[k] if m[k] is not None else np.nan)
        self.dirty = True

    def add_curriculum(self, m: Dict[str, Any]) -> None:
        self.stage_t.append(m["timesteps"]); self.stage_i.append(m["stage_index"] + 1); self.stage_d.append(m["difficulty"])
        self.dirty = True

    # ------------------------------------------------------------------ rendering (worker thread)
    def _snapshot(self) -> Dict[str, Any]:
        cut = lambda d: {k: np.asarray(v[-HISTORY:], dtype=float) for k, v in d.items()}   # noqa: E731
        return {"ep": cut(self.ep) or {"t": []}, "upd": cut(self.upd) or {"t": []}, "comps": cut(self.comps),
                "stage_t": list(self.stage_t), "stage_i": list(self.stage_i), "stage_d": list(self.stage_d),
                "target": self.target_steps}

    def refresh(self) -> None:
        if not self.dirty or self._busy or not self.isVisible():
            return
        self.dirty = False
        idx = self.tabs.currentIndex()
        lbl = self.labels[idx]
        w, h = max(lbl.width(), 200), max(lbl.height(), 120)
        snap = self._snapshot()
        for k in ("ep", "upd"):                       # draw functions index these keys directly
            snap[k].setdefault("t", np.array([]))
        fn = draw_curves if idx == 0 else draw_breakdown
        self._busy = True

        def work() -> None:
            try:
                img = fn(snap, w, h)
            except Exception:       # noqa: BLE001 - never kill the GUI because of a plotting problem
                import logging
                logging.getLogger(__name__).exception("plot rendering failed")
                img = None
            self.bridge.done.emit(idx, img)
        threading.Thread(target=work, daemon=True, name="plot-render").start()

    def _show(self, idx: int, rgba: Optional[np.ndarray]) -> None:
        self._busy = False
        if rgba is None:
            return
        h, w, _ = rgba.shape
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        self.labels[idx].setPixmap(QPixmap.fromImage(img))
