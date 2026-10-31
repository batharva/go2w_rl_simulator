"""Tests 13-15: GUI launches, stays responsive during training, stopping training terminates the worker."""
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.styles import QSS  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QApplication.instance() or QApplication([])
    a.setStyleSheet(QSS)
    return a


def spin(app, seconds, until=None):
    """Pump the Qt event loop; returns the longest gap between iterations (responsiveness probe)."""
    t0, last, worst = time.time(), time.time(), 0.0
    while time.time() - t0 < seconds:
        app.processEvents()
        now = time.time()
        worst = max(worst, now - last)
        last = now
        time.sleep(0.005)
        if until and until():
            break
    return worst


@pytest.fixture()
def window(app, cfg):
    cfg["gui"].update(render_width=320, render_height=200, render_fps=8)
    w = MainWindow(cfg)
    w.confirm_quit = False
    w.show()
    yield w
    w.close()


def test13_gui_launches_with_all_panels_and_buttons(window):
    d = window.dashboard
    for b in ("start_training", "pause_training", "resume_training", "stop_training", "reset_env", "start_sim",
              "stop_sim", "load_model", "save_model", "eval_model"):
        assert getattr(d, b).text()
    assert window.view and window.control and window.training and window.terrain and window.reward and window.analytics and window.models
    assert len(window.reward.spins) == 20 and len(window.control.manual_sliders) == 12
    assert not d.stop_training.isEnabled() and d.start_training.isEnabled()


def test_simulation_process_streams_frames_and_accepts_commands(app, window):
    frames = []
    orig = window.view.update_frame
    window.view.update_frame = lambda m: (frames.append(m), orig(m))
    window.start_simulation()
    spin(app, 40, until=lambda: len(frames) >= 3)
    assert len(frames) >= 3 and frames[-1]["telemetry"]["terrain"] == "flat"
    window.terrain.type.setCurrentText("stairs")
    window.terrain.generate_random()
    spin(app, 15, until=lambda: window.statusBar().currentMessage().startswith("Terrain: stairs"))
    assert window.statusBar().currentMessage().startswith("Terrain: stairs")
    window.control.vx.setValue(0.3)
    window.control.set_mode("manual")
    window.control.manual_sliders[1].setValue(0.5)
    n0 = len(frames)
    spin(app, 10, until=lambda: len(frames) > n0 + 2)
    assert frames[-1]["telemetry"]["mode"] == "manual"
    assert abs(frames[-1]["telemetry"]["command"][0] - 0.3) < 1e-6
    window.stop_simulation()
    spin(app, 20, until=lambda: not window.sim.running)
    assert not window.sim.running


def test14_15_gui_responsive_during_training_and_stop_is_clean(app, window, cfg):
    window.training.n_envs.setValue(1)
    window.training.n_steps.setValue(128); window.training.batch.setValue(64)
    window.training.total.setValue(2_000_000)          # long: must be ended by the Stop button
    window.training.ckpt.setValue(1000)
    window.training.eval_freq.setValue(1000)
    window.training.min_steps.setValue(0)
    window.start_training()
    assert window.train.running
    worst = spin(app, 60, until=lambda: window.analytics.ep["t"] and window.train_state == "running"
                 and len(window.analytics.upd["t"]) >= 2)
    assert window.train_state == "running"
    assert window.analytics.ep["t"] and len(window.analytics.upd["t"]) >= 2   # real metrics arrived from the worker
    worst = max(worst, spin(app, 5))
    assert worst < 0.5, f"GUI froze for {worst:.2f}s during training"
    # pause / resume
    window._pause(); spin(app, 8, until=lambda: window.train_state == "paused")
    assert window.train_state == "paused"
    window._resume(); spin(app, 8, until=lambda: window.train_state == "running")
    assert window.train_state == "running"
    # stop
    proc = window.train.process
    window.stop_training()
    spin(app, 90, until=lambda: not window.train.running)
    assert not window.train.running and not proc.is_alive()
    assert (Path(cfg["paths"]["models"]) / "last_model.zip").exists()
    spin(app, 10, until=lambda: window.train_state == "idle")      # the GUI notices the exit on its next poll
    assert window.train_state == "idle"


def test_load_policy_into_live_simulation(app, window, cfg):
    """A trained .zip can be loaded from the GUI; the sim worker switches to policy mode and obeys commands."""
    from rl.ppo_trainer import PPOTrainer
    PPOTrainer(cfg).train()
    model = str(Path(cfg["paths"]["models"]) / "last_model.zip")
    frames = []
    orig = window.view.update_frame
    window.view.update_frame = lambda m: (frames.append(m), orig(m))
    window.load_model_into_sim(model)                      # starts the simulation first (pending model)
    spin(app, 60, until=lambda: frames and frames[-1]["telemetry"]["mode"] == "policy")
    assert frames[-1]["telemetry"]["mode"] == "policy"
    assert frames[-1]["telemetry"]["model"] == "last_model.zip"
    window.control._direction((1, 0, 0))                   # "Forward" button
    spin(app, 5)
    assert frames[-1]["telemetry"]["command"][0] == pytest.approx(0.5)
    window.models.refresh()
    assert any(e["name"] == "last_model.zip" for e in window.models.entries)
    window.stop_simulation()
    spin(app, 20, until=lambda: not window.sim.running)


def test_invalid_ppo_settings_are_rejected_before_training(app, window, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: shown.append(a[2]))
    window.training.batch.setValue(1_000_000)               # larger than n_steps * n_envs
    window.start_training()
    assert shown and "batch_size" in shown[0] and not window.train.running
    window.training.arch_pi.setText("abc")
    window.start_training()
    assert "comma-separated" in shown[-1]
