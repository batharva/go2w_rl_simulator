"""PPO configuration + curriculum stage selection (validated before training starts)."""
from __future__ import annotations

from typing import Any, Dict, List

from PySide6.QtWidgets import (QCheckBox, QComboBox, QLabel, QLineEdit, QScrollArea, QVBoxLayout, QWidget)

from gui.widgets import csv_ints, dspin, group, ispin, sci_spin


class TrainingPanel(QWidget):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        p = cfg["ppo"]
        self.lr = sci_spin(1e-6, 1e-1, p["learning_rate"], 6, "Adam learning rate")
        self.gamma = dspin(0.5, 1.0, p["gamma"], 0.001, 4, "Discount factor")
        self.lam = dspin(0.5, 1.0, p["gae_lambda"], 0.001, 4, "GAE lambda")
        self.n_steps = ispin(16, 100_000, p["n_steps"], 64, "Rollout steps per environment per update")
        self.batch = ispin(2, 1_000_000, p["batch_size"], 64, "Mini-batch size (<= n_steps * n_envs)")
        self.epochs = ispin(1, 100, p["n_epochs"], 1, "Optimisation epochs per update")
        self.clip = dspin(0.01, 0.9, p["clip_range"], 0.01, 3, "PPO clip range")
        self.ent = sci_spin(0.0, 0.5, p["ent_coef"], 5, "Entropy coefficient")
        self.vf = dspin(0.0, 10.0, p["vf_coef"], 0.1, 3, "Value-function coefficient")
        self.grad = dspin(0.01, 100.0, p["max_grad_norm"], 0.1, 3, "Gradient clipping norm")
        self.total = ispin(1_000, 2_000_000_000, p["total_timesteps"], 100_000, "Target total timesteps")
        self.n_envs = ispin(1, 64, p["n_envs"], 1, "Parallel environments (one process each)")
        self.seed = ispin(0, 2_000_000_000, p["seed"], 1, "Global random seed (reproducibility)")
        self.arch_pi = QLineEdit(", ".join(map(str, p["net_arch_pi"])))
        self.arch_vf = QLineEdit(", ".join(map(str, p["net_arch_vf"])))
        self.act = QComboBox(); self.act.addItems(["elu", "tanh", "relu"]); self.act.setCurrentText(p["activation"])
        self.log_std = dspin(-3.0, 1.0, p["log_std_init"], 0.1, 2, "Initial log std of the Gaussian policy")
        self.device = QComboBox(); self.device.addItems(["cpu", "cuda", "auto"]); self.device.setCurrentText(p["device"])
        self.ckpt = ispin(1_000, 2_000_000_000, p["checkpoint_freq"], 10_000, "Checkpoint every N timesteps")

        c = cfg["curriculum"]
        self.curr_enabled = QCheckBox("Enable curriculum (performance-gated)")
        self.curr_enabled.setChecked(c["enabled"])
        self.eval_freq = ispin(1_000, 2_000_000_000, c["eval_freq"], 10_000, "Timesteps between curriculum evaluations")
        self.eval_eps = ispin(1, 100, c["eval_episodes"], 1)
        self.min_steps = ispin(0, 2_000_000_000, c["min_steps_per_stage"], 10_000, "Minimum steps before a stage may end")
        self.stage_boxes: List[QCheckBox] = []
        stage_layout_rows = []
        for s in c["stages"]:
            cb = QCheckBox(s["name"].replace("_", " "))
            cb.setChecked(s.get("enabled", True))
            self.stage_boxes.append(cb)
            stage_layout_rows.append(("", cb))

        lay = QVBoxLayout(self)
        lay.addWidget(group("PPO hyper-parameters", [
            ("Learning rate", self.lr), ("Gamma", self.gamma), ("GAE lambda", self.lam),
            ("Rollout steps / env", self.n_steps), ("Batch size", self.batch), ("Epochs", self.epochs),
            ("Clip range", self.clip), ("Entropy coef", self.ent), ("Value coef", self.vf),
            ("Max grad norm", self.grad), ("Total timesteps", self.total), ("Parallel envs", self.n_envs),
            ("Seed", self.seed)]))
        lay.addWidget(group("Network", [("Actor layers", self.arch_pi), ("Critic layers", self.arch_vf),
                                        ("Activation", self.act), ("Initial log std", self.log_std),
                                        ("Torch device", self.device), ("Checkpoint every", self.ckpt)]))
        g = group("Curriculum", [("", self.curr_enabled), ("Evaluate every", self.eval_freq),
                                 ("Eval episodes", self.eval_eps), ("Min steps / stage", self.min_steps)] + stage_layout_rows)
        lay.addWidget(g)
        self.msg = QLabel("")
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        lay.addStretch(1)

    def read_config(self, cfg: Dict[str, Any]) -> None:
        """Write the widget values into ``cfg``. Raises ValueError with a readable message on bad input."""
        p = cfg["ppo"]
        try:
            p["net_arch_pi"], p["net_arch_vf"] = csv_ints(self.arch_pi.text()), csv_ints(self.arch_vf.text())
        except ValueError as e:
            raise ValueError(f"Network layer sizes must be comma-separated integers, e.g. 256, 128, 64 ({e})")
        p.update(learning_rate=self.lr.value(), gamma=self.gamma.value(), gae_lambda=self.lam.value(),
                 n_steps=self.n_steps.value(), batch_size=self.batch.value(), n_epochs=self.epochs.value(),
                 clip_range=self.clip.value(), ent_coef=self.ent.value(), vf_coef=self.vf.value(),
                 max_grad_norm=self.grad.value(), total_timesteps=self.total.value(), n_envs=self.n_envs.value(),
                 seed=self.seed.value(), activation=self.act.currentText(), log_std_init=self.log_std.value(),
                 device=self.device.currentText(), checkpoint_freq=self.ckpt.value())
        c = cfg["curriculum"]
        c.update(enabled=self.curr_enabled.isChecked(), eval_freq=self.eval_freq.value(),
                 eval_episodes=self.eval_eps.value(), min_steps_per_stage=self.min_steps.value())
        for s, cb in zip(c["stages"], self.stage_boxes):
            s["enabled"] = cb.isChecked()
