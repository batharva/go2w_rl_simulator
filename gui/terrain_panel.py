"""Terrain + domain-randomisation configuration. Terrain changes are applied to the live simulation immediately."""
from __future__ import annotations

import random
from typing import Any, Callable, Dict

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from gui.widgets import dspin, group, ispin
from terrain.terrain_generator import TERRAIN_TYPES


class TerrainPanel(QWidget):
    def __init__(self, cfg: Dict[str, Any], send: Callable[..., None]):
        super().__init__()
        self.send = send
        t, r = cfg["terrain"], cfg["randomization"]
        self.type = QComboBox(); self.type.addItems(TERRAIN_TYPES + ["random"])
        self.type.setCurrentText(t["type"])
        self.diff = dspin(0.0, 1.0, t["difficulty"], 0.05, 2, "0 = easiest, 1 = hardest")
        self.hmax = dspin(0.0, 1.0, t["max_height"], 0.01, 3, "Maximum terrain height variation [m]")
        self.hmin = dspin(-1.0, 1.0, t["min_height"], 0.01, 3, "Minimum terrain height (negative = pits) [m]")
        self.rough = dspin(0.0, 1.0, t["roughness"], 0.05, 2)
        self.dens = dspin(0.0, 3.0, t["obstacle_density"], 0.1, 2, "Obstacles per m² (scaled by difficulty)")
        self.osize = dspin(0.05, 1.5, t["obstacle_size"], 0.05, 2, "Obstacle edge length [m]")
        self.slope = dspin(0.0, 40.0, t["slope_angle"], 1.0, 1, "Slope angle at difficulty 1 [deg]")
        self.fric = dspin(0.05, 3.0, t["friction"], 0.05, 2, "Ground friction coefficient")
        self.rest = dspin(0.0, 1.0, t["restitution"], 0.05, 2, "Ground restitution")
        self.seed = ispin(0, 2_000_000_000, t["seed"], 1, "Terrain random seed (same seed = same terrain)")
        self.regen = ispin(0, 1000, t["regen_every_episodes"], 1, "Training: new random terrain every N episodes (0 = never)")

        apply_btn = QPushButton("Apply terrain"); apply_btn.setObjectName("primary")
        apply_btn.clicked.connect(lambda: self.apply(new_seed=False))
        rand_btn = QPushButton("Generate Random Terrain")
        rand_btn.clicked.connect(self.generate_random)
        lay = QVBoxLayout(self)
        lay.addWidget(group("Terrain", [("Type", self.type), ("Difficulty", self.diff), ("Max height [m]", self.hmax),
                                        ("Min height [m]", self.hmin), ("Roughness", self.rough),
                                        ("Obstacle density", self.dens), ("Obstacle size [m]", self.osize),
                                        ("Slope angle [°]", self.slope), ("Friction", self.fric),
                                        ("Restitution", self.rest), ("Seed", self.seed),
                                        ("Regenerate every (train)", self.regen)]))
        row = QHBoxLayout(); row.addWidget(apply_btn); row.addWidget(rand_btn)
        lay.addLayout(row)
        note = QLabel("Applied immediately to the live simulation. For training, the curriculum chooses terrains per "
                      "stage; these values are used when the curriculum is disabled (and for height/roughness limits).")
        note.setWordWrap(True); note.setStyleSheet("color:#8794a7;")
        lay.addWidget(note)

        self.dr_enabled = QCheckBox("Enable domain randomisation"); self.dr_enabled.setChecked(r["enabled"])
        self.mass_lo = dspin(0.5, 1.0, r["mass_scale"][0], 0.05, 2); self.mass_hi = dspin(1.0, 2.0, r["mass_scale"][1], 0.05, 2)
        self.fr_lo = dspin(0.05, 1.0, r["friction"][0], 0.05, 2); self.fr_hi = dspin(1.0, 3.0, r["friction"][1], 0.05, 2)
        self.mot_lo = dspin(0.5, 1.0, r["motor_strength"][0], 0.05, 2); self.mot_hi = dspin(1.0, 1.5, r["motor_strength"][1], 0.05, 2)
        self.damp_lo = dspin(0.1, 1.0, r["joint_damping_scale"][0], 0.1, 2); self.damp_hi = dspin(1.0, 5.0, r["joint_damping_scale"][1], 0.1, 2)
        self.noise = QCheckBox("Sensor noise"); self.noise.setChecked(r["sensor_noise"])
        self.push_on = QCheckBox("External pushes"); self.push_on.setChecked(r["push_enabled"])
        self.push_v = dspin(0.0, 3.0, r["push_velocity"], 0.1, 2, "Max push velocity change [m/s]")
        self.init_xy = dspin(0.0, 2.0, r["init_pos_xy"], 0.05, 2); self.init_q = dspin(0.0, 0.5, r["init_joint_noise"], 0.01, 3)
        lay.addWidget(group("Domain randomisation", [
            ("", self.dr_enabled), ("Mass scale min / max", self._pair(self.mass_lo, self.mass_hi)),
            ("Ground friction min / max", self._pair(self.fr_lo, self.fr_hi)),
            ("Motor strength min / max", self._pair(self.mot_lo, self.mot_hi)),
            ("Joint damping scale min / max", self._pair(self.damp_lo, self.damp_hi)),
            ("", self.noise), ("", self.push_on), ("Max push [m/s]", self.push_v),
            ("Initial xy offset [m]", self.init_xy), ("Initial joint noise [rad]", self.init_q)]))
        ap = QPushButton("Apply randomisation to live simulation")
        ap.clicked.connect(lambda: self.send("randomization", self.randomization_dict()))
        lay.addWidget(ap)
        lay.addStretch(1)

    @staticmethod
    def _pair(a, b) -> QWidget:
        w = QWidget(); l = QHBoxLayout(w); l.setContentsMargins(0, 0, 0, 0); l.addWidget(a); l.addWidget(b)
        return w

    def terrain_dict(self) -> Dict[str, Any]:
        return dict(type=self.type.currentText(), difficulty=self.diff.value(), max_height=self.hmax.value(),
                    min_height=self.hmin.value(), roughness=self.rough.value(), obstacle_density=self.dens.value(),
                    obstacle_size=self.osize.value(), slope_angle=self.slope.value(), friction=self.fric.value(),
                    restitution=self.rest.value(), seed=self.seed.value(), regen_every_episodes=self.regen.value())

    def randomization_dict(self) -> Dict[str, Any]:
        return dict(enabled=self.dr_enabled.isChecked(), mass_scale=[self.mass_lo.value(), self.mass_hi.value()],
                    friction=[self.fr_lo.value(), self.fr_hi.value()],
                    motor_strength=[self.mot_lo.value(), self.mot_hi.value()],
                    joint_damping_scale=[self.damp_lo.value(), self.damp_hi.value()],
                    sensor_noise=self.noise.isChecked(), push_enabled=self.push_on.isChecked(),
                    push_velocity=self.push_v.value(), init_pos_xy=self.init_xy.value(),
                    init_joint_noise=self.init_q.value())

    def validate(self) -> None:
        if self.hmin.value() > self.hmax.value():
            raise ValueError("Terrain: minimum height must not exceed maximum height")
        if self.mass_lo.value() > self.mass_hi.value() or self.fr_lo.value() > self.fr_hi.value():
            raise ValueError("Randomisation: min must not exceed max")

    def apply(self, new_seed: bool) -> None:
        self.validate()
        d = self.terrain_dict()
        if d["type"] == "random":
            d["type"] = random.choice(TERRAIN_TYPES)
        d["new_seed"] = False
        if new_seed:
            d["seed"] = random.randint(0, 2_000_000)
            self.seed.setValue(d["seed"])
        self.send("terrain", d)

    def generate_random(self) -> None:
        self.apply(new_seed=True)

    def read_config(self, cfg: Dict[str, Any]) -> None:
        self.validate()
        d = self.terrain_dict()
        cfg["terrain"].update({k: v for k, v in d.items() if k != "type" or v != "random"})
        cfg["terrain"]["random_type"] = d["type"] == "random"
        if d["type"] == "random":
            cfg["terrain"]["type"] = "flat"
        cfg["randomization"].update(self.randomization_dict())
