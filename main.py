#!/usr/bin/env python
"""Entry point.

  python main.py                                   GUI (default)
  python main.py --validate                        URDF / robot validation report
  python main.py --headless --train [--resume P]   PPO training without GUI
  python main.py --evaluate --model M [--compare M2 M3]   evaluation report (CSV + JSON)
  python main.py --infer --model M                 real-time inference in PyBullet's native GUI window
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.config import load_config, resolve_path  # noqa: E402
from utils.logger import get_logger, setup_logging  # noqa: E402

log = get_logger("main")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Unitree Go2 PyBullet PPO simulator", formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: config.yaml)")
    ap.add_argument("--headless", action="store_true", help="no GUI (use with --train)")
    ap.add_argument("--train", action="store_true", help="run PPO training")
    ap.add_argument("--resume", default=None, help="checkpoint .zip to resume training from")
    ap.add_argument("--timesteps", type=int, default=None, help="total training timesteps (target)")
    ap.add_argument("--n-envs", type=int, default=None, help="parallel environments")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-curriculum", action="store_true", help="disable the curriculum")
    ap.add_argument("--evaluate", action="store_true", help="evaluate --model on the standard scenarios")
    ap.add_argument("--compare", nargs="*", default=None, help="additional models to compare with --model")
    ap.add_argument("--model", default=None, help="model .zip for --evaluate / --infer")
    ap.add_argument("--episodes", type=int, default=None, help="episodes per evaluation scenario")
    ap.add_argument("--robust", action="store_true", help="evaluate with domain randomisation + pushes")
    ap.add_argument("--infer", action="store_true", help="real-time policy inference in the native PyBullet GUI window")
    ap.add_argument("--terrain", default=None, help="terrain type for --infer")
    ap.add_argument("--command", nargs=3, type=float, default=[0.5, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    ap.add_argument("--validate", action="store_true", help="validate the URDF / robot and exit")
    return ap.parse_args(argv)


def build_cfg(a: argparse.Namespace):
    ov = {}
    if a.timesteps: ov.setdefault("ppo", {})["total_timesteps"] = a.timesteps
    if a.n_envs: ov.setdefault("ppo", {})["n_envs"] = a.n_envs
    if a.seed is not None: ov.setdefault("ppo", {})["seed"] = a.seed
    if a.no_curriculum: ov["curriculum"] = {"enabled": False}
    return load_config(a.config, ov)


def run_validate(cfg) -> int:
    from utils.robot_validation import validate_robot
    return 0 if validate_robot(cfg)["ok"] else 1


def run_train(cfg, resume) -> int:
    from rl.ppo_trainer import PPOTrainer, validate_ppo_config
    errs = validate_ppo_config(cfg)
    if errs:
        print("Invalid configuration:\n  - " + "\n  - ".join(errs)); return 2
    def sink(m):
        if m["type"] == "update":
            print(f"[{m['timesteps']:>10,}] fps={m['fps']:.0f} mean_return={m['mean_return']} vel_err={m['mean_vel_error']} "
                  f"pi_loss={m['policy_loss']} v_loss={m['value_loss']}", flush=True)
        elif m["type"] == "curriculum":
            print(f"   curriculum eval @ {m['timesteps']:,}: stage={m['stage']} d={m['difficulty']:.2f} survival="
                  f"{m['metrics']['survival_rate']:.2f} vel_err={m['metrics']['vel_error']:.2f} -> {m['decision']} "
                  f"({m['reason']}){' [best saved]' if m['best_saved'] else ''}", flush=True)
        elif m["type"] == "checkpoint":
            print("   checkpoint:", m["path"], flush=True)
    import signal
    trainer = PPOTrainer(cfg, sink)
    signal.signal(signal.SIGINT, lambda *_: (print("\nCtrl-C: stopping after the current step and saving…"), trainer.stop_event.set()))
    trainer.train(resume)
    return 0


def run_evaluate(cfg, a) -> int:
    from rl.evaluation import compare_models, evaluate_model_file
    if not a.model:
        print("--evaluate needs --model"); return 2
    if a.compare:
        res = compare_models([a.model] + a.compare, cfg, a.episodes, a.robust, lambda i, n, name: print(f"[{i + 1}/{n}] {name}"))
        print("comparison written:", res["csv"], res["json"])
        return 0
    rep = evaluate_model_file(a.model, cfg, a.episodes, a.robust, progress=lambda i, n, name: print(f"[{i}/{n}] {name}", flush=True))
    o = rep["overall"]
    print(f"\n{'scenario':<15}{'fall':>6}{'complete':>9}{'vel_err':>9}{'speed':>7}{'dist':>7}{'tilt':>7}{'power':>8}{'slip':>7}  pass")
    for r in rep["scenarios"]:
        print(f"{r['scenario']:<15}{r['fall_rate']:>6.2f}{r['completion_rate']:>9.2f}{r['vel_error']:>9.3f}{r['avg_velocity']:>7.2f}"
              f"{r['distance_travelled']:>7.2f}{r['orientation_deviation_rad']:>7.3f}{r['energy_w']:>8.1f}{r['foot_slip_mps']:>7.3f}  {'yes' if r['passed'] else 'no'}")
    print("\nVERDICT:", o["verdict"])
    print("reports:", rep["report_csv"], rep["report_json"])
    return 0


def run_infer(cfg, a) -> int:
    import numpy as np
    from environment.go2_env import Go2QuadrupedEnv
    from rl.model_manager import ModelManager
    if not a.model:
        print("--infer needs --model"); return 2
    cfg["terrain"]["regen_every_episodes"] = 0
    if a.terrain: cfg["terrain"]["type"] = a.terrain
    env = Go2QuadrupedEnv(cfg, gui=True, seed=0)
    model = ModelManager.load_ppo(a.model)
    env.set_command(*a.command)
    obs, _ = env.reset()
    import pybullet as p
    try:
        while True:
            t0 = time.time()
            obs, _, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0])
            pos = env.sim.base_state().pos
            p.resetDebugVisualizerCamera(1.5, 45, -25, pos.tolist(), physicsClientId=env.sim.cid)
            if term or trunc: obs, _ = env.reset()
            time.sleep(max(0.0, env.dt - (time.time() - t0)))
    except (KeyboardInterrupt, p.error):
        pass
    finally:
        env.close()
    return 0


def run_gui(cfg) -> int:
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow
    from gui.styles import QSS
    app = QApplication(sys.argv[:1])
    app.setStyleSheet(QSS)
    w = MainWindow(cfg)
    w.show()
    return app.exec()


def main() -> int:
    a = parse_args()
    setup_logging()
    cfg = build_cfg(a)
    setup_logging(log_file=resolve_path(cfg, "logs") / "app.log")
    if a.validate: return run_validate(cfg)
    if a.train: return run_train(cfg, a.resume)
    if a.evaluate: return run_evaluate(cfg, a)
    if a.infer: return run_infer(cfg, a)
    if a.headless:
        print("--headless needs --train, --evaluate or --validate"); return 2
    return run_gui(cfg)


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
