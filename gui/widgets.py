"""Small widget factories shared by the panels."""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QSlider, QSpinBox, QWidget)


def dspin(lo: float, hi: float, val: float, step: float = 0.01, decimals: int = 3, tip: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi); w.setDecimals(decimals); w.setSingleStep(step); w.setValue(val)
    w.setKeyboardTracking(False)
    if tip:
        w.setToolTip(tip)
    return w


def ispin(lo: int, hi: int, val: int, step: int = 1, tip: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi); w.setSingleStep(step); w.setValue(val)
    w.setKeyboardTracking(False)
    if tip:
        w.setToolTip(tip)
    return w


def sci_spin(lo: float, hi: float, val: float, decimals: int = 6, tip: str = "") -> QDoubleSpinBox:
    return dspin(lo, hi, val, step=10 ** -(decimals - 2), decimals=decimals, tip=tip)


def group(title: str, rows: Iterable[Tuple[str, QWidget]]) -> QGroupBox:
    g = QGroupBox(title)
    f = QFormLayout(g)
    f.setLabelAlignment(Qt.AlignLeft)
    f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for label, w in rows:
        f.addRow(label, w)
    return g


class LabeledSlider(QWidget):
    """Float slider with a numeric read-out; ``changed(value)`` is called while dragging."""

    def __init__(self, lo: float, hi: float, val: float, decimals: int = 2, on_change: Optional[Callable[[float], None]] = None):
        super().__init__()
        self.scale = 10 ** decimals
        self.decimals = decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(lo * self.scale), int(hi * self.scale))
        self.label = QLabel()
        self.label.setMinimumWidth(48)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.on_change = on_change
        self.slider.valueChanged.connect(self._moved)
        self.setValue(val)

    def _moved(self, _: int) -> None:
        self.label.setText(f"{self.value():.{self.decimals}f}")
        if self.on_change:
            self.on_change(self.value())

    def value(self) -> float:
        return self.slider.value() / self.scale

    def setValue(self, v: float) -> None:
        self.slider.setValue(int(round(v * self.scale)))
        self.label.setText(f"{self.value():.{self.decimals}f}")


def csv_ints(text: str) -> list:
    """'256, 128, 64' -> [256, 128, 64] (raises ValueError on bad input)."""
    vals = [int(t) for t in text.replace(";", ",").split(",") if t.strip()]
    if not vals:
        raise ValueError("empty list")
    return vals
