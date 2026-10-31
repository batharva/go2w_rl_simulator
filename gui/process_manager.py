"""Qt-side controllers for the worker processes (training, live simulation, evaluation).

Each controller starts a ``spawn`` process, drains its message queue from a ``QTimer`` (so the GUI thread never blocks),
and shuts the process down cleanly (cooperative stop event first, terminate only after a timeout).
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from utils.logger import get_logger

log = get_logger(__name__)
CTX = mp.get_context("spawn")


class _Controller(QObject):
    message = Signal(dict)
    stopped = Signal()

    def __init__(self, poll_ms: int = 40, max_per_poll: int = 200):
        super().__init__()
        self.process: Optional[mp.process.BaseProcess] = None
        self.out_q = None
        self.stop_event = None
        self._max = max_per_poll
        self._stop_deadline: Optional[float] = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(poll_ms)

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.is_alive()

    def _poll(self) -> None:
        if self.out_q is not None:
            for _ in range(self._max):
                try:
                    self.message.emit(self.out_q.get_nowait())
                except queue.Empty:
                    break
        if self.process is not None and not self.process.is_alive():
            self._drain_and_cleanup()
        elif self._stop_deadline is not None and time.time() > self._stop_deadline and self.process is not None:
            log.warning("%s did not stop in time - terminating", type(self).__name__)
            self.process.terminate()

    def _drain_and_cleanup(self) -> None:
        if self.out_q is not None:                      # final messages (e.g. 'finished')
            while True:
                try:
                    self.message.emit(self.out_q.get_nowait())
                except (queue.Empty, OSError, ValueError):
                    break
        if self.process is not None:
            self.process.join(timeout=1.0)
        self.process, self._stop_deadline = None, None
        self.stopped.emit()

    def request_stop(self, timeout_s: float = 90.0) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        self._stop_deadline = time.time() + timeout_s

    def shutdown(self) -> None:
        """Blocking stop used when the window closes."""
        if self.process is not None and self.process.is_alive():
            if self.stop_event is not None:
                self.stop_event.set()
            self.process.join(timeout=15)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=3)
        self.process = None


class TrainingController(_Controller):
    def start(self, cfg: Dict[str, Any], resume: Optional[str] = None) -> None:
        from rl.ppo_trainer import training_worker_main
        if self.running:
            raise RuntimeError("Training is already running")
        self.stop_event, self.pause_event, self.save_event = CTX.Event(), CTX.Event(), CTX.Event()
        self.out_q = CTX.Queue(maxsize=5000)
        self._stop_deadline = None
        self.process = CTX.Process(target=training_worker_main, name="go2-training", daemon=False,
                                   args=(cfg, self.stop_event, self.pause_event, self.out_q, resume, self.save_event))
        self.process.start()

    def pause(self) -> None:
        if self.running:
            self.pause_event.set()

    def resume(self) -> None:
        if self.running:
            self.pause_event.clear()

    def save_now(self) -> None:
        if self.running:
            self.save_event.set()

    def request_stop(self, timeout_s: float = 120.0) -> None:
        if self.running:
            self.pause_event.clear()                     # a paused worker must wake up to see the stop request
        super().request_stop(timeout_s)


class SimulationController(_Controller):
    def __init__(self) -> None:
        super().__init__(poll_ms=25, max_per_poll=20)
        self.cmd_q = None

    def start(self, cfg: Dict[str, Any]) -> None:
        from simulation.sim_worker import simulation_worker_main
        if self.running:
            return
        self.stop_event = CTX.Event()
        self.cmd_q = CTX.Queue(maxsize=1000)
        self.out_q = CTX.Queue(maxsize=6)
        self._stop_deadline = None
        self.process = CTX.Process(target=simulation_worker_main, name="go2-sim", daemon=True,
                                   args=(cfg, self.cmd_q, self.out_q, self.stop_event))
        self.process.start()

    def send(self, *cmd: Any) -> None:
        if self.running and self.cmd_q is not None:
            try:
                self.cmd_q.put_nowait(cmd)
            except queue.Full:
                log.warning("simulation command queue full - dropped %s", cmd[0])

    def request_stop(self, timeout_s: float = 20.0) -> None:
        self.send("quit")
        super().request_stop(timeout_s)


class EvalController(_Controller):
    def start(self, cfg: Dict[str, Any], model_paths: List[str], episodes: int, robust: bool) -> None:
        from rl.evaluation import evaluation_worker_main
        if self.running:
            raise RuntimeError("An evaluation is already running")
        self.stop_event = CTX.Event()
        self.out_q = CTX.Queue(maxsize=200)
        self._stop_deadline = None
        self.process = CTX.Process(target=evaluation_worker_main, name="go2-eval", daemon=True,
                                   args=(cfg, model_paths, episodes, robust, self.out_q, self.stop_event))
        self.process.start()
