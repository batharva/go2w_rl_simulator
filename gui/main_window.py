"""Main application window: wires panels, worker-process controllers and the dashboard together."""
from __future__ import annotations

import copy
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QFileDialog, QInputDialog, QMainWindow, QMessageBox, QProgressDialog, QScrollArea,
                               QSplitter, QTabWidget, QVBoxLayout, QWidget)

from gui.analytics_panel import AnalyticsPanel
from gui.control_panel import ControlPanel
from gui.dashboard import Dashboard
from gui.model_panel import ModelPanel
from gui.process_manager import EvalController, SimulationController, TrainingController
from gui.results_dialog import ResultsDialog
from gui.reward_panel import RewardPanel
from gui.simulation_view import SimulationView
from gui.terrain_panel import TerrainPanel
from gui.training_panel import TrainingPanel
from rl.ppo_trainer import validate_ppo_config
from utils.config import resolve_path
from utils.logger import get_logger

log = get_logger(__name__)


def _scroll(w: QWidget) -> QScrollArea:
    s = QScrollArea()
    s.setWidgetResizable(True)
    s.setWidget(w)
    return s


class MainWindow(QMainWindow):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.setWindowTitle("Unitree Go2 - PyBullet PPO Locomotion Lab")
        self.resize(1560, 980)
        self.cfg = cfg
        self.sim = SimulationController()
        self.train = TrainingController()
        self.evalc = EvalController()
        self.train_state = "idle"
        self.pending_model: Optional[str] = None
        self.loaded_model_path: Optional[str] = None
        self.resume_path: Optional[str] = None
        self.last_steps = 0
        self.progress_dlg: Optional[QProgressDialog] = None
        self.confirm_quit = True            # tests / scripted use can disable the modal confirmation

        # ---- panels -------------------------------------------------------------------
        self.dashboard = Dashboard()
        self.view = SimulationView(self.sim.send)
        self.control = ControlPanel(cfg, self.sim.send)
        self.training = TrainingPanel(cfg)
        self.terrain = TerrainPanel(cfg, self.sim.send)
        self.reward = RewardPanel(cfg, self.sim.send)
        self.models = ModelPanel(cfg)
        self.analytics = AnalyticsPanel()

        tabs = QTabWidget()
        for w, name in ((self.control, "Robot control"), (self.training, "PPO / curriculum"), (self.terrain, "Terrain"),
                        (self.reward, "Rewards"), (self.models, "Models")):
            tabs.addTab(_scroll(w) if w is not self.models else w, name)
        tabs.setMinimumWidth(430)
        mid = QSplitter(Qt.Horizontal)
        mid.addWidget(self.view)
        mid.addWidget(tabs)
        mid.setStretchFactor(0, 3)
        mid.setStretchFactor(1, 1)
        vert = QSplitter(Qt.Vertical)
        top = QWidget(); tl = QVBoxLayout(top); tl.setContentsMargins(8, 8, 8, 0)
        tl.addWidget(self.dashboard)
        vert.addWidget(top)
        vert.addWidget(mid)
        vert.addWidget(self.analytics)
        vert.setStretchFactor(1, 4)
        vert.setStretchFactor(2, 3)
        root = QWidget(); root.setObjectName("root")
        QVBoxLayout(root).addWidget(vert)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready. Start the simulation to see the robot, or start training.")

        self._build_menu()
        self._connect()
        self.snapshot_timer = QTimer(self)
        self.snapshot_timer.timeout.connect(self._load_snapshot)

    # ================================================================== wiring
    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&Simulation")
        a = QAction("Load latest training snapshot into simulation", self); a.triggered.connect(self._load_snapshot)
        m.addAction(a)
        self.auto_follow = QAction("Auto-follow training snapshot (every 30 s)", self, checkable=True)
        self.auto_follow.toggled.connect(lambda on: self.snapshot_timer.start(30_000) if on else self.snapshot_timer.stop())
        m.addAction(self.auto_follow)
        v = self.menuBar().addMenu("&Tools")
        r = QAction("Validate robot (URDF report)", self); r.triggered.connect(self._validate_robot)
        v.addAction(r)

    def _connect(self) -> None:
        d = self.dashboard
        d.start_sim.clicked.connect(self.start_simulation)
        d.stop_sim.clicked.connect(self.stop_simulation)
        d.reset_env.clicked.connect(lambda: self.sim.send("reset"))
        d.start_training.clicked.connect(lambda: self.start_training(self.resume_path if self._ask_resume() else None))
        d.pause_training.clicked.connect(self._pause)
        d.resume_training.clicked.connect(self._resume)
        d.stop_training.clicked.connect(self.stop_training)
        d.load_model.clicked.connect(self.load_model_dialog)
        d.save_model.clicked.connect(self.save_model)
        d.eval_model.clicked.connect(self.evaluate_dialog)
        self.sim.message.connect(self.on_sim_message)
        self.sim.stopped.connect(self._sim_stopped)
        self.train.message.connect(self.on_train_message)
        self.train.stopped.connect(self._train_stopped)
        self.evalc.message.connect(self.on_eval_message)
        self.models.load_requested.connect(self.load_model_into_sim)
        self.models.resume_requested.connect(lambda p: self.start_training(p))
        self.models.evaluate_requested.connect(self.run_evaluation)

    def _ask_resume(self) -> bool:
        """Start Training normally starts fresh; 'Resume training from…' (Models tab) passes a checkpoint."""
        self.resume_path = None
        return False

    # ================================================================== config
    def collect_config(self) -> Dict[str, Any]:
        cfg = copy.deepcopy(self.cfg)
        self.training.read_config(cfg)
        self.terrain.read_config(cfg)
        self.reward.read_config(cfg)
        cfg["control"]["kp"], cfg["control"]["kd"] = self.control.kp.value(), self.control.kd.value()
        cfg["gui"]["realtime_factor"] = self.control.rtf.value()
        return cfg

    # ================================================================== simulation
    def start_simulation(self) -> None:
        try:
            cfg = self.collect_config()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid settings", str(e)); return
        self.sim.start(cfg)
        self.dashboard.set_sim_state(True, "Starting…")
        self.statusBar().showMessage("Starting PyBullet simulation process…")

    def stop_simulation(self) -> None:
        self.sim.request_stop()
        self.dashboard.set_sim_state(False, "Stopping…")

    def _sim_stopped(self) -> None:
        self.dashboard.set_sim_state(False)
        self.view.set_stopped()
        self.statusBar().showMessage("Simulation stopped")

    def on_sim_message(self, m: Dict[str, Any]) -> None:
        t = m["type"]
        if t == "frame":
            self.view.update_frame(m)
            tel = m.get("telemetry") or {}
            if tel:
                c = self.dashboard.cards
                c["speed"].set(f"{tel['speed']:.2f} m/s"); c["height"].set(f"{tel['height']:.3f} m")
                c["rollpitch"].set(f"{tel['roll'] * 57.3:+.1f}° / {tel['pitch'] * 57.3:+.1f}°")
                c["terrain"].set(tel["terrain"]); c["ep_reward"].set(f"{tel['ep_reward']:.1f}")
                if self.train_state == "idle":
                    c["episode"].set(str(tel["episode"]))
        elif t == "ready":
            self.dashboard.set_sim_state(True)
            self.statusBar().showMessage(f"Simulation running (observation dim {m['obs_dim']})")
            if self.pending_model:
                self.sim.send("load_model", self.pending_model); self.pending_model = None
        elif t == "model":
            if m["loaded"]:
                self.control.set_mode("policy")
                self.statusBar().showMessage(f"Policy '{m['name']}' loaded ({m['timesteps']:,} training steps)")
            else:
                QMessageBox.warning(self, "Model could not be loaded", m["error"])
        elif t == "terrain":
            p = m["params"]
            self.statusBar().showMessage(f"Terrain: {p['type']}  difficulty {p['difficulty']:.2f}  seed {p['seed']}")
        elif t == "error":
            self.statusBar().showMessage(m["message"])
            QMessageBox.critical(self, "Simulation error", m["message"] + ("\n\n" + m["traceback"][-800:] if m.get("traceback") else ""))

    # ================================================================== training
    def start_training(self, resume: Optional[str] = None) -> None:
        if self.train.running:
            QMessageBox.information(self, "Training", "Training is already running."); return
        try:
            cfg = self.collect_config()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid settings", str(e)); return
        errs = validate_ppo_config(cfg)
        if errs:
            QMessageBox.warning(self, "Invalid PPO configuration", "\n".join("• " + e for e in errs)); return
        if resume is None:
            self.analytics.clear()
        self.analytics.target_steps = cfg["ppo"]["total_timesteps"]
        self.train.start(cfg, resume)
        self._set_train_state("starting")
        self.statusBar().showMessage("Training worker started" + (f" (resuming {Path(resume).name})" if resume else ""))

    def _set_train_state(self, s: str) -> None:
        self.train_state = s
        self.dashboard.set_training_state(s)

    def _pause(self) -> None:
        self.train.pause()

    def _resume(self) -> None:
        self.train.resume()

    def stop_training(self) -> None:
        self.train.request_stop()
        self._set_train_state("stopping")
        self.statusBar().showMessage("Stopping training - the current model will be saved as models/last_model.zip …")

    def _train_stopped(self) -> None:
        if self.train_state != "error":
            self._set_train_state("idle")
        self.models.refresh()
        self.statusBar().showMessage("Training worker has exited")

    def on_train_message(self, m: Dict[str, Any]) -> None:
        t, c = m["type"], self.dashboard.cards
        if t == "status":
            if m["state"] in ("starting", "running", "paused") and self.train_state != "stopping":
                self._set_train_state(m["state"])
        elif t == "info":
            self.statusBar().showMessage(f"PPO ready: {m['parameters']:,} parameters, obs dim {m['obs_dim']}, "
                                         f"{m['remaining']:,} steps to go")
        elif t == "episode":
            self.analytics.add_episode(m)
            c["steps"].set(f"{m['timesteps']:,}"); c["episode"].set(str(m["episode"]))
            c["ep_reward"].set(f"{m['return']:.1f}"); c["avg_reward"].set(f"{m['mean_return']:.1f}")
            c["ep_len"].set(str(m["length"]))
            tgt = max(self.analytics.target_steps, 1)
            self.dashboard.progress.setValue(int(min(100, 100 * m["timesteps"] / tgt)))
        elif t == "update":
            self.analytics.add_update(m)
            c["fps"].set(f"{m['fps']:.0f}")
        elif t == "curriculum":
            self.analytics.add_curriculum(m)
            c["stage"].set(f"{m['stage_index'] + 1}: {m['stage'][2:]}  d={m['difficulty']:.2f}")
            self.statusBar().showMessage(f"Curriculum eval: survival {m['metrics']['survival_rate']:.2f}, "
                                         f"vel err {m['metrics']['vel_error']:.2f} → {m['decision']} {m['reason']}")
        elif t == "checkpoint":
            self.models.refresh()
            if m.get("final"):
                self.statusBar().showMessage(f"Model saved: {m['path']}")
        elif t == "error":
            self._set_train_state("error")
            QMessageBox.critical(self, "Training error", m["message"] + "\n\n" + m.get("traceback", "")[-1200:])

    # ================================================================== models
    def load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load PPO model", str(resolve_path(self.cfg, "models")),
                                              "SB3 models (*.zip)")
        if path:
            self.load_model_into_sim(path)

    def load_model_into_sim(self, path: str) -> None:
        self.loaded_model_path = path
        if self.sim.running:
            self.sim.send("load_model", path)
        else:
            self.pending_model = path
            self.start_simulation()

    def _load_snapshot(self) -> None:
        snap = resolve_path(self.cfg, "models") / "latest_policy.zip"
        if snap.exists() and self.sim.running:
            self.sim.send("load_model", str(snap))
        elif not snap.exists():
            self.statusBar().showMessage("No training snapshot yet (written every snapshot_freq steps)")

    def save_model(self) -> None:
        if self.train.running:
            self.train.save_now()
            self.statusBar().showMessage("Saving current training model to models/manual_<steps>.zip …")
            return
        if not self.loaded_model_path:
            QMessageBox.information(self, "Save model", "No model is loaded and no training is running."); return
        dst, _ = QFileDialog.getSaveFileName(self, "Save a copy of the loaded model", str(resolve_path(self.cfg, "models") / "saved_model.zip"), "SB3 models (*.zip)")
        if dst:
            from rl.model_manager import ModelManager
            ModelManager(self.cfg).copy_to(self.loaded_model_path, dst)
            self.models.refresh()

    # ================================================================== evaluation
    def evaluate_dialog(self) -> None:
        sel = self.models.selected_paths() or ([self.loaded_model_path] if self.loaded_model_path else [])
        if not sel:
            path, _ = QFileDialog.getOpenFileName(self, "Model to evaluate", str(resolve_path(self.cfg, "models")), "SB3 models (*.zip)")
            if not path:
                return
            sel = [path]
        self.run_evaluation(sel)

    def run_evaluation(self, paths: List[str]) -> None:
        if self.evalc.running:
            QMessageBox.information(self, "Evaluation", "An evaluation is already running."); return
        eps, ok = QInputDialog.getInt(self, "Evaluation", "Episodes per scenario:", int(self.cfg["evaluation"]["episodes_per_scenario"]), 1, 50)
        if not ok:
            return
        try:
            cfg = self.collect_config()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid settings", str(e)); return
        self.evalc.start(cfg, paths, eps, robust=False)
        self.progress_dlg = QProgressDialog("Evaluating…", "Cancel", 0, 100, self)
        self.progress_dlg.setWindowModality(Qt.WindowModal)
        self.progress_dlg.canceled.connect(self.evalc.request_stop)
        self.progress_dlg.show()

    def on_eval_message(self, m: Dict[str, Any]) -> None:
        t = m["type"]
        if t == "progress" and self.progress_dlg:
            self.progress_dlg.setLabelText(f"Evaluating: {m['name']}")
            self.progress_dlg.setValue(int(100 * m["done"] / max(m["total"], 1)))
        elif t == "result":
            if self.progress_dlg: self.progress_dlg.close()
            ResultsDialog(self, m).exec()
        elif t == "error":
            if self.progress_dlg: self.progress_dlg.close()
            QMessageBox.critical(self, "Evaluation error", m["message"])
        elif t == "finished" and self.progress_dlg:
            self.progress_dlg.close()

    def _validate_robot(self) -> None:
        from utils.robot_validation import validate_robot
        rep = validate_robot(self.collect_config(), verbose=False)
        QMessageBox.information(self, "Robot validation", f"{'OK' if rep['ok'] else 'FAILED'}\n\nlinks: {rep['num_links']}  joints: "
                                f"{rep['num_joints']}  actuated: {rep['num_actuated_joints']}\nmissing meshes: {len(rep['missing_meshes'])}\n"
                                + "\n".join(rep["warnings"] + rep["errors"]) + "\n\nFull report: results/robot_validation.json")

    # ================================================================== shutdown
    def closeEvent(self, e) -> None:  # noqa: N802
        if self.train.running:
            if self.confirm_quit and QMessageBox.question(self, "Quit", "Training is running. Stop it (model will be saved) and quit?") != QMessageBox.Yes:
                e.ignore(); return
            self.train.request_stop()
        for c in (self.sim, self.evalc, self.train):
            c.shutdown()
        e.accept()
