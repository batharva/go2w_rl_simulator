"""Reward weight editor."""
from __future__ import annotations

from typing import Any, Callable, Dict

from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.widgets import dspin
from utils.config import DEFAULT_REWARD_WEIGHTS

DESCR = {
    "track_vx": "forward/backward velocity tracking  exp(-err²/σ)", "track_vy": "lateral velocity tracking",
    "track_wz": "yaw-rate tracking", "alive": "survival bonus", "foot_contact": "≥2 feet down (stand: ≥3)",
    "feet_air_time": "swing-phase reward at touchdown (walking only)", "orientation": "|projected gravity xy|²",
    "base_height": "(height above feet − target)²", "lin_vel_z": "vertical velocity²", "ang_vel_xy": "roll/pitch rate²",
    "joint_vel": "Σ qdot²", "joint_acc": "Σ qddot²", "action_rate": "Σ (a − a_prev)²", "energy": "Σ |τ·qdot|",
    "foot_slip": "tangential foot speed² in contact", "default_pose": "Σ (q − q_default)²",
    "stall": "commanded to move but not moving", "body_contact": "hip/thigh/calf touching ground",
    "self_collision": "self-collision contacts", "fall": "one-off penalty at a fall"}


class RewardPanel(QWidget):
    def __init__(self, cfg: Dict[str, Any], send: Callable[..., None]):
        super().__init__()
        self.send = send
        self.spins: Dict[str, QDoubleSpinBox] = {}
        box = QGroupBox("Reward weights (+ reward / − penalty)")
        grid = QGridLayout(box)
        for i, (k, v) in enumerate(cfg["reward"]["weights"].items()):
            s = dspin(-1000.0, 1000.0, v, 0.1 if abs(v) >= 0.1 else 0.001, 7)
            s.setMaximumWidth(120)
            self.spins[k] = s
            grid.addWidget(QLabel(k), i, 0)
            grid.addWidget(s, i, 1)
            d = QLabel(DESCR.get(k, "")); d.setStyleSheet("color:#8794a7;font-size:10px;")
            grid.addWidget(d, i, 2)
        self.sig_v = dspin(0.01, 2.0, cfg["reward"]["sigma_velocity"], 0.01, 3, "σ of the linear-velocity tracking kernel")
        self.sig_w = dspin(0.01, 2.0, cfg["reward"]["sigma_angular"], 0.01, 3, "σ of the yaw-rate tracking kernel")
        n = len(self.spins)
        grid.addWidget(QLabel("sigma_velocity"), n, 0); grid.addWidget(self.sig_v, n, 1)
        grid.addWidget(QLabel("sigma_angular"), n + 1, 0); grid.addWidget(self.sig_w, n + 1, 1)
        lay = QVBoxLayout(self)
        lay.addWidget(box)
        row = QHBoxLayout()
        apply_b = QPushButton("Apply to live simulation"); apply_b.clicked.connect(lambda: self.send("reward_weights", self.weights()))
        reset_b = QPushButton("Restore defaults"); reset_b.clicked.connect(self.restore_defaults)
        row.addWidget(apply_b); row.addWidget(reset_b)
        lay.addLayout(row)
        note = QLabel("Weights are used by the next training run (Start / Resume). Changing them while training "
                      "requires a restart because the worker processes hold their own copy.")
        note.setWordWrap(True); note.setStyleSheet("color:#8794a7;")
        lay.addWidget(note)
        lay.addStretch(1)

    def weights(self) -> Dict[str, float]:
        return {k: s.value() for k, s in self.spins.items()}

    def restore_defaults(self) -> None:
        for k, s in self.spins.items():
            s.setValue(DEFAULT_REWARD_WEIGHTS[k])
        self.sig_v.setValue(0.15); self.sig_w.setValue(0.25)
        self.send("reward_weights", self.weights())

    def read_config(self, cfg: Dict[str, Any]) -> None:
        cfg["reward"]["weights"] = self.weights()
        cfg["reward"]["sigma_velocity"] = self.sig_v.value()
        cfg["reward"]["sigma_angular"] = self.sig_w.value()
