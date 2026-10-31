"""Simulation panel: rendered PyBullet frames with mouse camera control + live telemetry."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                               QSizePolicy, QVBoxLayout, QWidget)

from gui.styles import BAD, BORDER, GOOD, MUTED

DEFAULT_CAM = dict(distance=1.6, yaw=45.0, pitch=-25.0, follow=True, target=[0.0, 0.0, 0.2])
PRESETS = {"Front": (0.0, -15.0), "Rear": (180.0, -15.0), "Left": (90.0, -15.0), "Right": (-90.0, -15.0),
           "Top": (0.0, -88.0), "Iso": (45.0, -25.0)}


class FrameLabel(QLabel):
    """Shows frames; left-drag = orbit, wheel = zoom, right/middle-drag = pan (disables follow)."""

    def __init__(self, on_camera: Callable[[Dict[str, Any]], None]):
        super().__init__("Simulation not running\n(press  Start Simulation)")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background:#0b0e13;border:1px solid {BORDER};border-radius:8px;color:{MUTED};")
        self.cam = dict(DEFAULT_CAM)
        self.on_camera = on_camera
        self._last: Optional[QPoint] = None
        self._pix: Optional[QPixmap] = None
        self.setMouseTracking(False)

    def show_frame(self, w: int, h: int, data: bytes) -> None:
        img = QImage(data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def mousePressEvent(self, e) -> None:  # noqa: N802
        self._last = e.position().toPoint()

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        self._last = None

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._last is None:
            return
        pos = e.position().toPoint()
        dx, dy = pos.x() - self._last.x(), pos.y() - self._last.y()
        self._last = pos
        if e.buttons() & Qt.LeftButton:
            self.cam["yaw"] = (self.cam["yaw"] - dx * 0.4) % 360.0
            self.cam["pitch"] = max(-89.0, min(-2.0, self.cam["pitch"] - dy * 0.3))
        elif e.buttons() & (Qt.RightButton | Qt.MiddleButton):
            import math
            s = self.cam["distance"] * 0.002
            yaw = math.radians(self.cam["yaw"])
            t = self.cam["target"]
            t[0] += (-dx * math.cos(yaw) + dy * math.sin(yaw)) * s
            t[1] += (-dx * math.sin(yaw) - dy * math.cos(yaw)) * s
            self.cam["follow"] = False
        self.on_camera(self.cam)

    def wheelEvent(self, e) -> None:  # noqa: N802
        steps = e.angleDelta().y() / 120.0
        self.cam["distance"] = max(0.4, min(25.0, self.cam["distance"] * (0.88 ** steps)))
        self.on_camera(self.cam)


class SimulationView(QWidget):
    """Frame area on the left, telemetry on the right."""

    def __init__(self, send: Callable[..., None]):
        super().__init__()
        self.send = send
        self.frame = FrameLabel(self._camera_changed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.addWidget(self.frame, 1)
        bar = QHBoxLayout()
        for name in PRESETS:
            b = QPushButton(name)
            b.clicked.connect(lambda _=False, n=name: self.preset(n))
            bar.addWidget(b)
        self.follow = QCheckBox("Follow robot")
        self.follow.setChecked(True)
        self.follow.toggled.connect(self._follow_toggled)
        reset = QPushButton("Reset camera")
        reset.clicked.connect(self.reset_camera)
        bar.addStretch(1)
        bar.addWidget(self.follow)
        bar.addWidget(reset)
        left.addLayout(bar)
        lay.addLayout(left, 1)

        self.labels: Dict[str, QLabel] = {}
        box = QGroupBox("Telemetry")
        box.setFixedWidth(230)
        form = QFormLayout(box)
        for key, title in [("mode", "Mode"), ("model", "Policy"), ("terrain", "Terrain"), ("time", "Episode time"),
                           ("command", "Command vx/vy/wz"), ("vel", "Velocity vx/vy/wz"), ("speed", "Speed"),
                           ("height", "Base height"), ("rp", "Roll / pitch"), ("reward", "Step reward"),
                           ("ep_reward", "Episode reward"), ("torque", "Max |torque|"), ("fps", "Sim steps/s")]:
            l = QLabel("-")
            l.setTextInteractionFlags(Qt.TextSelectableByMouse)
            form.addRow(title, l)
            self.labels[key] = l
        self.contact_lbl = QLabel()
        form.addRow("Foot contact", self.contact_lbl)
        self.parts = QLabel("")
        self.parts.setStyleSheet(f"color:{MUTED};font-size:10px;")
        self.parts.setWordWrap(True)
        form.addRow(self.parts)
        lay.addWidget(box)

    def _camera_changed(self, cam: Dict[str, Any]) -> None:
        self.send("camera", dict(cam))

    def _follow_toggled(self, on: bool) -> None:
        self.frame.cam["follow"] = on
        self._camera_changed(self.frame.cam)

    def preset(self, name: str) -> None:
        yaw, pitch = PRESETS[name]
        self.frame.cam.update(yaw=yaw, pitch=pitch)
        self._camera_changed(self.frame.cam)

    def reset_camera(self) -> None:
        self.frame.cam = dict(DEFAULT_CAM, target=[0.0, 0.0, 0.2])
        self.follow.setChecked(True)
        self._camera_changed(self.frame.cam)

    def set_stopped(self) -> None:
        self.frame.setPixmap(QPixmap())
        self.frame._pix = None
        self.frame.setText("Simulation not running\n(press  Start Simulation)")

    def update_frame(self, msg: Dict[str, Any]) -> None:
        self.frame.show_frame(msg["w"], msg["h"], msg["data"])
        t = msg.get("telemetry") or {}
        if not t:
            return
        L = self.labels
        L["mode"].setText(t["mode"])
        L["model"].setText(t["model"] or "none loaded")
        L["terrain"].setText(f"{t['terrain']} (d={t['difficulty']:.2f})")
        L["time"].setText(f"{t['time']:.1f} s   (ep {t['episode']})")
        c = t["command"]
        L["command"].setText(f"{c[0]:+.2f} {c[1]:+.2f} {c[2]:+.2f}")
        L["vel"].setText(f"{t['vx']:+.2f} {t['vy']:+.2f} {t['wz']:+.2f}")
        L["speed"].setText(f"{t['speed']:.2f} m/s")
        L["height"].setText(f"{t['height']:.3f} m")
        L["rp"].setText(f"{t['roll'] * 57.3:+.1f}° / {t['pitch'] * 57.3:+.1f}°")
        L["reward"].setText(f"{t['reward']:+.3f}")
        L["ep_reward"].setText(f"{t['ep_reward']:.1f}")
        L["torque"].setText(f"{t['torque_max']:.1f} N·m")
        L["fps"].setText(f"{t['sim_fps']:.0f}")
        names = ["FL", "FR", "RL", "RR"]
        self.contact_lbl.setText("  ".join(f"<span style='color:{GOOD if c else BAD}'>{n}</span>"
                                          for n, c in zip(names, t["contacts"])))
        top = sorted(t["parts"].items(), key=lambda kv: -abs(kv[1]))[:5]
        self.parts.setText("top terms: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
