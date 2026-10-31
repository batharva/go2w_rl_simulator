"""Shows evaluation / comparison results."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QDialog, QLabel, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout

COLS = [("scenario", "Scenario"), ("avg_episode_reward", "Reward"), ("distance_travelled", "Dist [m]"),
        ("avg_velocity", "Speed [m/s]"), ("vel_error", "Vel err [m/s]"), ("fall_rate", "Fall rate"),
        ("completion_rate", "Completion"), ("terrain_completion_rate", "Terrain compl."),
        ("orientation_deviation_rad", "Tilt [rad]"), ("energy_w", "Power [W]"), ("foot_slip_mps", "Slip [m/s]"),
        ("passed", "Pass")]


class ResultsDialog(QDialog):
    def __init__(self, parent, result: Dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("Evaluation results")
        self.resize(1050, 520)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)
        for model, rep in result["reports"].items():
            t = QTableWidget(len(rep["scenarios"]) + 1, len(COLS))
            t.setHorizontalHeaderLabels([c[1] for c in COLS])
            rows = rep["scenarios"] + [{**rep["overall"], "scenario": "OVERALL (mean)", "passed": rep["overall"]["passed_all_flat_scenarios"]}]
            for r, row in enumerate(rows):
                for c, (k, _) in enumerate(COLS):
                    v = row.get(k, "")
                    txt = ("yes" if v else "no") if k == "passed" else (f"{v:.3f}" if isinstance(v, float) else str(v))
                    it = QTableWidgetItem(txt)
                    if k == "passed":
                        it.setForeground(QColor("#3ecf8e" if v else "#ef5b5b"))
                    t.setItem(r, c, it)
            t.resizeColumnsToContents()
            holder = QVBoxLayout()
            tabs.addTab(t, Path(model).name)
            verdict = rep["overall"]["verdict"]
            lay.addWidget(QLabel(f"<b>{Path(model).name}</b> ({rep.get('model_timesteps', '?'):,} steps): {verdict}"))
        lay.addWidget(QLabel(f"Reports: {result.get('csv', '')}  |  {result.get('json', '')}"))
