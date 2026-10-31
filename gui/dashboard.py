"""Main dashboard: status cards + the primary action buttons."""
from __future__ import annotations

from typing import Dict

from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from gui.styles import BAD, GOOD, MUTED, WARN

CARDS = [("robot", "Robot status"), ("training", "Training status"), ("episode", "Episode"),
         ("steps", "Total training steps"), ("ep_reward", "Episode reward"), ("avg_reward", "Average reward"),
         ("ep_len", "Episode length"), ("speed", "Walking speed"), ("height", "Base height"),
         ("rollpitch", "Roll / pitch"), ("fps", "Training FPS"), ("terrain", "Terrain"), ("stage", "Curriculum")]


class Card(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(0)
        t = QLabel(title.upper())
        t.setObjectName("card_title")
        self.value = QLabel("-")
        self.value.setObjectName("card_value")
        lay.addWidget(t)
        lay.addWidget(self.value)

    def set(self, text: str, color: str = "") -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color:{color};" if color else "")


class Dashboard(QWidget):
    def __init__(self) -> None:
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.cards: Dict[str, Card] = {}
        for i, (key, title) in enumerate(CARDS):
            c = Card(title)
            self.cards[key] = c
            grid.addWidget(c, i // 7, i % 7)
        outer.addLayout(grid)
        self.progress = QProgressBar()
        self.progress.setFormat("training progress  %p%")
        outer.addWidget(self.progress)

        def btn(text: str, name: str = "") -> QPushButton:
            b = QPushButton(text)
            if name:
                b.setObjectName(name)
            return b
        self.start_training = btn("▶ Start Training", "primary")
        self.pause_training = btn("⏸ Pause")
        self.resume_training = btn("⏵ Resume")
        self.stop_training = btn("⏹ Stop Training", "danger")
        self.reset_env = btn("↺ Reset Environment")
        self.start_sim = btn("Start Simulation", "primary")
        self.stop_sim = btn("Stop Simulation")
        self.load_model = btn("Load Model…")
        self.save_model = btn("Save Model…")
        self.eval_model = btn("Evaluate Model")
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        for b in (self.start_training, self.pause_training, self.resume_training, self.stop_training):
            row1.addWidget(b)
        row1.addSpacing(18)
        for b in (self.start_sim, self.stop_sim, self.reset_env):
            row1.addWidget(b)
        row1.addStretch(1)
        for b in (self.load_model, self.save_model, self.eval_model):
            row2.addWidget(b)
        row2.addStretch(1)
        outer.addLayout(row1)
        outer.addLayout(row2)
        self.set_training_state("idle")
        self.set_sim_state(False)

    def set_training_state(self, state: str) -> None:
        colors = {"idle": MUTED, "starting": WARN, "running": GOOD, "paused": WARN, "stopping": WARN,
                  "finished": MUTED, "error": BAD}
        self.cards["training"].set(state.capitalize(), colors.get(state, ""))
        running = state in ("starting", "running", "paused", "stopping")
        self.start_training.setEnabled(not running)
        self.pause_training.setEnabled(state == "running")
        self.resume_training.setEnabled(state == "paused")
        self.stop_training.setEnabled(running and state != "stopping")

    def set_sim_state(self, running: bool, text: str = "") -> None:
        self.cards["robot"].set(text or ("Simulating" if running else "Stopped"), GOOD if running else MUTED)
        self.start_sim.setEnabled(not running)
        self.stop_sim.setEnabled(running)
        self.reset_env.setEnabled(running)
