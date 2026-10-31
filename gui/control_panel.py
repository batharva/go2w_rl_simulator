"""Robot control: velocity command sliders/buttons, mode selection, manual joint test, PD gains, pushes."""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from PySide6.QtWidgets import (QButtonGroup, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton, QRadioButton,
                               QVBoxLayout, QWidget)

from gui.widgets import LabeledSlider, dspin, group

JOINT_LABELS = ["FL hip", "FL thigh", "FL calf", "FR hip", "FR thigh", "FR calf",
                "RL hip", "RL thigh", "RL calf", "RR hip", "RR thigh", "RR calf"]


class ControlPanel(QWidget):
    def __init__(self, cfg: Dict[str, Any], send: Callable[..., None]):
        super().__init__()
        self.send = send
        c = cfg["commands"]
        self.vx = LabeledSlider(-1.0, 1.5, 0.0, 2, lambda _: self._cmd())
        self.vy = LabeledSlider(-1.0, 1.0, 0.0, 2, lambda _: self._cmd())
        self.wz = LabeledSlider(-2.0, 2.0, 0.0, 2, lambda _: self._cmd())
        self.step = dspin(0.05, 1.5, 0.5, 0.05, 2, "Magnitude used by the direction buttons")

        lay = QVBoxLayout(self)
        mode = QGroupBox("Control mode")
        ml = QHBoxLayout(mode)
        self.mode_group = QButtonGroup(self)
        self.modes = {}
        for key, text in (("hold", "Hold stance"), ("policy", "PPO policy"), ("manual", "Manual joints")):
            rb = QRadioButton(text)
            self.mode_group.addButton(rb)
            ml.addWidget(rb)
            self.modes[key] = rb
            rb.toggled.connect(lambda on, k=key: on and self._mode(k))
        self.modes["hold"].setChecked(True)
        lay.addWidget(mode)

        lay.addWidget(group("Velocity command (body frame; used by the PPO policy)", [
            ("Forward vx [m/s]", self.vx), ("Lateral vy [m/s]", self.vy), ("Yaw rate wz [rad/s]", self.wz),
            ("Button magnitude", self.step)]))
        btns = QGridLayout()
        spec = [("Forward", 0, 1, lambda: (1, 0, 0)), ("Left", 1, 0, lambda: (0, 1, 0)), ("Stop", 1, 1, lambda: (0, 0, 0)),
                ("Right", 1, 2, lambda: (0, -1, 0)), ("Backward", 2, 1, lambda: (-1, 0, 0)),
                ("Turn left", 3, 0, lambda: (0, 0, 2)), ("Turn right", 3, 2, lambda: (0, 0, -2))]
        for text, r, col, fn in spec:
            b = QPushButton(text)
            if text == "Stop":
                b.setObjectName("danger")
            b.clicked.connect(lambda _=False, f=fn: self._direction(f()))
            btns.addWidget(b, r, col)
        box = QGroupBox("Command buttons")
        box.setLayout(btns)
        lay.addWidget(box)

        self.manual_sliders: List[LabeledSlider] = []
        mg = QGroupBox("Manual joint targets (normalised action -1 … +1, 0 = default stance)")
        mgl = QGridLayout(mg)
        for i, name in enumerate(JOINT_LABELS):
            s = LabeledSlider(-1.0, 1.0, 0.0, 2, lambda _: self._manual())
            self.manual_sliders.append(s)
            mgl.addWidget(QLabel(name), i // 3, (i % 3) * 2)
            mgl.addWidget(s, i // 3, (i % 3) * 2 + 1)
        zero = QPushButton("Zero all joints")
        zero.clicked.connect(self._zero)
        mgl.addWidget(zero, 4, 0, 1, 6)
        lay.addWidget(mg)

        self.kp = dspin(0.0, 400.0, cfg["control"]["kp"], 1.0, 1, "PD stiffness [N·m/rad]")
        self.kd = dspin(0.0, 20.0, cfg["control"]["kd"], 0.1, 2, "PD damping [N·m·s/rad]")
        self.rtf = dspin(0.05, 5.0, cfg["gui"]["realtime_factor"], 0.25, 2, "Simulation speed vs real time")
        self.push_v = dspin(0.0, 3.0, 0.8, 0.1, 2, "Velocity impulse of the push [m/s]")
        pb = QPushButton("Push robot (random direction)")
        pb.clicked.connect(self._push)
        for w in (self.kp, self.kd):
            w.valueChanged.connect(lambda _: self.send("gains", self.kp.value(), self.kd.value()))
        self.rtf.valueChanged.connect(lambda v: self.send("rtf", v))
        lay.addWidget(group("Actuators / simulation", [("PD Kp", self.kp), ("PD Kd", self.kd), ("Real-time factor", self.rtf),
                                                       ("Push strength", self.push_v), ("", pb)]))
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _cmd(self) -> None:
        self.send("command", self.vx.value(), self.vy.value(), self.wz.value())

    def _direction(self, d) -> None:
        m = self.step.value()
        self.vx.setValue(d[0] * m); self.vy.setValue(d[1] * m); self.wz.setValue(d[2] * m)
        if "policy" in self.modes and not self.modes["policy"].isChecked() and any(d):
            pass                                               # command is stored; shown in telemetry
        self._cmd()

    def _mode(self, key: str) -> None:
        self.send("mode", key)
        if key == "manual":
            self._manual()

    def _manual(self) -> None:
        self.send("manual_action", [s.value() for s in self.manual_sliders])

    def _zero(self) -> None:
        for s in self.manual_sliders:
            s.setValue(0.0)
        self._manual()

    def _push(self) -> None:
        import math, random
        a = random.uniform(0, 2 * math.pi)
        v = self.push_v.value()
        self.send("push", v * math.cos(a), v * math.sin(a))

    def set_mode(self, key: str) -> None:
        self.modes[key].setChecked(True)
