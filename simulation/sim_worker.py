"""Live-simulation worker process for the GUI.

It owns its *own* PyBullet client (DIRECT mode, software renderer), steps the Gym environment in (scaled) real time,
renders frames and streams frames + telemetry to the GUI through a queue.  The GUI never touches PyBullet, and the
training workers have their own clients, so physics clients are never shared.

Modes:  ``hold``   - zero action (robot holds the nominal stance);
        ``policy`` - the loaded PPO policy follows the velocity command;
        ``manual`` - 12 joint-target sliders (normalised actions) for actuator / joint testing.
"""
from __future__ import annotations

import json
import queue
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from utils.config import deep_update
from utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def _telemetry(env, mode: str, model_name: str, sim_fps: float, ep_reward: float) -> Dict[str, Any]:
    t = env.telemetry()
    if not t:
        return {}
    return {
        "speed": t["speed"], "vx": t["vx"], "vy": t["vy"], "wz": t["wz"], "height": t["height"],
        "height_above_feet": t["height_above_feet"], "roll": t["roll"], "pitch": t["pitch"], "yaw": t["yaw"],
        "terrain": t["terrain"], "command": [float(c) for c in t["command"]], "episode": t["episode"],
        "step": t["step"], "time": t["t"], "reward": t["reward"], "ep_reward": ep_reward,
        "parts": {k: float(v) for k, v in t["parts"].items()}, "contacts": [bool(c) for c in t["contacts"]],
        "mode": mode, "model": model_name, "sim_fps": sim_fps, "term_reason": t.get("term_reason", ""),
        "pos": [float(x) for x in t["pos"]], "torque_max": float(np.abs(t["tau"]).max()),
        "difficulty": env.sim.terrain.params.difficulty,
    }


def simulation_worker_main(cfg: Dict[str, Any], cmd_q, out_q, stop_event) -> None:
    setup_logging()
    from environment.go2_env import Go2QuadrupedEnv
    from rl.model_manager import ModelManager

    def emit(msg: Dict[str, Any], block: bool = False) -> None:
        try:
            out_q.put(msg, block=block, timeout=1.0) if block else out_q.put_nowait(msg)
        except queue.Full:
            pass

    try:
        cfg = json.loads(json.dumps(cfg))
        deep_update(cfg, {"terrain": {"regen_every_episodes": 0}})   # terrain only changes on user request
        g = cfg["gui"]
        w, h = int(g["render_width"]), int(g["render_height"])
        env = Go2QuadrupedEnv(cfg, render_mode="rgb_array", seed=int(cfg["ppo"]["seed"]))
        obs, _ = env.reset()
        emit({"type": "ready", "obs_dim": int(obs.shape[0])}, block=True)
    except Exception as e:  # noqa: BLE001
        emit({"type": "error", "message": f"Simulation failed to start: {e}", "traceback": traceback.format_exc()},
             block=True)
        return

    mode, model, model_name = "hold", None, ""
    paused = False
    manual = np.zeros(12, dtype=np.float32)
    cam = dict(distance=1.6, yaw=45.0, pitch=-25.0, follow=True, target=[0.0, 0.0, 0.2])
    rtf = float(g["realtime_factor"])
    frame_dt = 1.0 / max(float(g["render_fps"]), 1.0)
    last_frame = 0.0
    ep_reward, fps_t, fps_n, sim_fps = 0.0, time.time(), 0, 0.0
    pending_reload = False

    def load_model(path: str) -> None:
        nonlocal model, model_name, mode
        try:
            m = ModelManager.load_ppo(path, device="cpu")
            if m.observation_space.shape != env.observation_space.shape:
                raise ValueError(f"model expects obs {m.observation_space.shape}, env gives {env.observation_space.shape}")
            model, model_name = m, Path(path).name
            mode = "policy"
            emit({"type": "model", "loaded": True, "name": model_name, "timesteps": int(m.num_timesteps)}, True)
        except Exception as e:  # noqa: BLE001
            emit({"type": "model", "loaded": False, "error": f"{type(e).__name__}: {e}"}, True)

    def handle(cmd) -> bool:
        nonlocal mode, paused, manual, rtf, ep_reward, obs
        k = cmd[0]
        if k == "quit":
            return False
        if k == "command":
            env.set_command(*cmd[1:4])
        elif k == "clear_command":
            env.clear_command()
        elif k == "mode":
            mode = cmd[1]
        elif k == "manual_action":
            manual = np.asarray(cmd[1], dtype=np.float32)
        elif k == "load_model":
            load_model(cmd[1])
        elif k == "unload_model":
            nonlocal_set_model(None)
        elif k == "camera":
            cam.update(cmd[1])
        elif k == "pause":
            paused = bool(cmd[1])
        elif k == "reset":
            obs, _ = env.reset()
            ep_reward = 0.0
        elif k == "terrain":
            params = dict(cmd[1])
            new_seed = params.pop("new_seed", False)
            env.cfg["terrain"].update(params)
            try:
                info = env.regenerate_terrain(seed=None if new_seed else int(env.cfg["terrain"]["seed"]))
                obs, _ = env.reset()
                ep_reward = 0.0
                emit({"type": "terrain", "params": info}, True)
            except Exception as e:  # noqa: BLE001
                emit({"type": "error", "message": f"Terrain error: {e}", "traceback": ""}, True)
        elif k == "reward_weights":
            env.set_reward_weights(cmd[1])
        elif k == "randomization":
            env.set_randomization(**cmd[1])
        elif k == "gains":
            env.set_gains(cmd[1], cmd[2])
        elif k == "push":
            env.apply_push(cmd[1], cmd[2])
        elif k == "rtf":
            rtf = float(cmd[1])
        return True

    def nonlocal_set_model(m) -> None:
        nonlocal model, model_name, mode
        model, model_name = m, ""
        if m is None and mode == "policy":
            mode = "hold"

    try:
        running = True
        while running and not stop_event.is_set():
            t_loop = time.time()
            try:
                while True:
                    running = handle(cmd_q.get_nowait()) and running
            except queue.Empty:
                pass
            if paused:
                time.sleep(0.03)
            else:
                if mode == "policy" and model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                elif mode == "manual":
                    action = manual
                else:
                    action = np.zeros(12, dtype=np.float32)
                obs, r, term, trunc, info = env.step(action)
                ep_reward += r
                fps_n += 1
                if term or trunc:
                    emit({"type": "episode_end", "stats": info["episode_stats"]})
                    obs, _ = env.reset()
                    ep_reward = 0.0
            now = time.time()
            if now - fps_t > 1.0:
                sim_fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
            if now - last_frame >= frame_dt:
                last_frame = now
                target = env.sim.base_state().pos if cam["follow"] else np.asarray(cam["target"])
                img = env.sim.render(w, h, tuple(target), cam["distance"], cam["yaw"], cam["pitch"])
                emit({"type": "frame", "w": w, "h": h, "data": img.tobytes(),
                      "telemetry": _telemetry(env, mode, model_name, sim_fps, ep_reward)})
            if not paused:
                sleep = env.dt / max(rtf, 1e-3) - (time.time() - t_loop)
                if sleep > 0:
                    time.sleep(sleep)
    except Exception as e:  # noqa: BLE001
        log.exception("Simulation worker crashed")
        emit({"type": "error", "message": f"Simulation crashed: {type(e).__name__}: {e}",
              "traceback": traceback.format_exc()}, True)
    finally:
        env.close()
        emit({"type": "closed"}, True)
