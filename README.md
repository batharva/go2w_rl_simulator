# Unitree Go2 – PyBullet + PPO locomotion lab

A complete, modular Python project that loads **your** `go2_description.urdf` into PyBullet, simulates the robot with
a torque-controlled PD actuator model, trains a walking policy with **PPO (Stable-Baselines3)** through a performance-gated
**curriculum** over **procedurally randomised terrain**, and provides a dark **PySide6 GUI** with live analytics, manual
control, terrain editing, model management and an evaluation/comparison tool.

> **Honest status.** The code was executed and tested (`pytest tests`: 72 passed, see *Testing*). The *learning result*
> depends on training time; see section 8 for what a real run on one CPU core achieved, and use
> `python main.py --evaluate …` to measure your own checkpoints. The evaluator reports **PASS / NOT PASSED** from
> measured numbers – nothing in the project claims the robot walks unless the evaluation says so.

## 1. System requirements

* Linux / macOS / Windows, 4 GB RAM; a multi-core CPU is strongly recommended for training (one PyBullet env per core).
* Python 3.10 – 3.12 (developed on 3.11), no GPU required (PPO with small MLPs is faster on CPU).
* Display only for the GUI (`QT_QPA_PLATFORM=offscreen` is used by the tests).

## 2. Installation

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
`pybullet` ships wheels for most platform/Python combinations; otherwise pip compiles it (needs a C++ compiler, ~10 min).
`conda install -c conda-forge pybullet pytorch-cpu pyside6 stable-baselines3` is a good alternative.

### URDF and mesh assets

* The URDF is at `assets/go2_description.urdf` (copied from your upload, never modified).
* Its `package://go2_description/dae/*.dae` references are resolved by `AssetResolver` (searches `paths.mesh_dirs`,
  `assets/dae/`, the URDF folder, then the whole project for the file name).
  **Copy `base.dae, hip.dae, thigh.dae, calf.dae, foot.dae, thigh_mirror.dae, calf_mirror.dae` into `assets/dae/`** to get the
  textured Go2 in the GUI.
* **If the meshes are missing** (they were not part of the upload) a warning is logged and each affected link is drawn with
  its own *collision geometry* (grey). Physics is unaffected – collision and inertia come from the URDF. No other robot
  is ever substituted.
* The loader writes a *derived* URDF into `.cache/` (absolute mesh paths + the fix below); your file is untouched.
  **Inertia fix:** PyBullet silently gives **1 kg** to every link without `<inertial>` and treats `mass=0` links as static.
  The Go2 URDF has 11 such links (`imu`, `radar`, `front_camera`, `*_calflower`, `*_calflower1`) → +9 kg of ghost mass. They are
  given 1 g instead (reported by the validator; total mass 16.098 kg vs 16.087 kg in the URDF).

## 3. Running

```bash
python main.py                                  # GUI
python main.py --validate                       # robot validation report (also saved to results/robot_validation.json)
python main.py --headless --train               # PPO training, no GUI (Ctrl-C = save and stop)
python main.py --headless --train --timesteps 3000000 --n-envs 8 --seed 1
python main.py --headless --train --resume checkpoints/ckpt_0000500000.zip --timesteps 3000000
python main.py --evaluate --model models/best_model.zip                    # CSV + JSON in results/
python main.py --evaluate --model models/best_model.zip --compare checkpoints/ckpt_0000500000.zip checkpoints/ckpt_0001000000.zip
python main.py --infer --model models/best_model.zip --command 0.5 0 0     # real-time inference in PyBullet's native window
python main.py --config my_config.yaml --headless --train                  # any YAML overrides config.yaml
tensorboard --logdir logs/tensorboard
```
Other options: `--no-curriculum`, `--episodes N`, `--robust` (evaluate with domain randomisation + pushes), `--terrain T` (for `--infer`).

### GUI workflow

1. **Start Simulation** – a separate process owns a PyBullet client, steps the robot in real time and streams rendered frames
   (software renderer) + telemetry. Mouse: left-drag orbit, wheel zoom, right-drag pan; presets Front/Rear/Left/Right/Top/Iso, *Follow robot*.
2. **Robot control tab** – modes *Hold stance*, *PPO policy* (velocity command sliders + Forward/Back/Left/Right/Stop/Turn buttons)
   and *Manual joints* (12 target sliders for actuator tests); PD gains, real-time factor, push button.
3. **Terrain tab** – type, difficulty, min/max height, roughness, obstacle density/size, slope angle, friction, restitution, seed,
   **Generate Random Terrain** (applied immediately). Domain-randomisation ranges are edited here too.
4. **PPO / curriculum tab** – all hyper-parameters (validated before start), network layers, stages on/off.
5. **Rewards tab** – every weight, *Restore defaults*, *Apply to live simulation*.
6. **Start Training** – a *training process* (SB3 PPO with one process per environment) reports episodes/losses to the plots.
   **Pause / Resume / Stop** are cooperative (stop saves `models/last_model.zip`). **Save Model** while training writes
   `models/manual_<steps>.zip`. *Simulation → Auto-follow training snapshot* shows the latest policy in the live view.
7. **Models tab** – checkpoint history with stage/score, load into simulation, *Resume training from…*, evaluate/compare
   (multi-select), show the training config stored next to each model.
8. **Evaluate Model** – runs the evaluation scenarios in its own process and shows the table + verdict.

The GUI never touches PyBullet: simulation, training and evaluation are separate `spawn` processes with their own physics
clients; the Qt thread only drains queues from a timer and blits finished images (plots are rendered in a worker thread).

## 4. Design

### Observation (dim 52, fixed-scale normalisation, clipped to ±10)
`base lin vel (body) ×2 | base ang vel (body) ×0.25 | projected gravity | command (vx,vy,wz) ×(2,2,0.25) | q − q_default (12) | qdot ×0.05 (12) | previous action (12) | foot contact flags (4)`.
Noise (domain randomisation) is added to the IMU / encoder-type signals only. No terrain information is given (blind locomotion).

### Action → torque
12 actions in [−1, 1] → joint position target `q_default + scale·a·(distance to the URDF limit on that side)`, `scale=0.25`, always inside the URDF limits
(a margin of 0.02 rad). Every physics step: `τ = Kp(q_des−q) + Kd(0−q̇) − damping·q̇ − Coulomb·tanh(q̇/0.05)`, clipped to the URDF **effort**
(× per-joint motor strength) and derated linearly to 0 at the URDF **velocity** limit. Default `Kp=40, Kd=1`.
Physics 1/240 s, 5 sub-steps per action ⇒ 48 Hz control (50 Hz requested; the exact period is `substeps/240`).

### Default stance
Computed, not assumed: 2-link IK from the segment lengths measured via forward kinematics of the URDF frames (thigh ≈ 0.79 rad, calf ≈ −1.58 rad),
base height from FK so the feet just touch the ground (0.312 m); validated in tests by holding the stance for 3 s.
Collision pairs that overlap *by design* in that stance (`*_calflower1` ↔ `*_foot`) are detected and disabled; all other self-collisions stay on.

### Reward (`environment/reward.py`, weights in `config.yaml`)
Velocity tracking `exp(−err²/σ)` for vx, vy, wz; alive bonus; foot-contact; swing-phase (air time) reward; penalties for orientation, base height
(relative to the *feet*), vertical/roll-pitch velocity, joint velocity/acceleration, action rate, energy `Σ|τ·q̇|`, foot slip, deviation from default
pose, **stall** (commanded to move but not moving), body contact, self-collision and a one-off fall penalty. All 20 terms are logged per episode
and plotted (*Reward breakdown* tab). Reward-hacking guards are documented in the module docstring.

### Terrain (`terrain/`)
Heightfield **collision shapes** (box for flat): `flat, uneven, stepping, stairs, slopes, rough, obstacles, mixed`, difficulty-scaled, with a flat spawn pad,
friction/restitution on the body, seeded and reproducible. Verified by ray-casting the physics body against the height maps (tests).

### Domain randomisation
Mass+inertia scale, joint damping, ground friction, motor strength, Kp scale, sensor noise, random pushes (velocity impulses), initial xy/yaw/joint offsets.
All ranges in `randomization:`; seeded.

### Curriculum (`rl/curriculum.py`)
Stage 1 standing → 2 forward/backward walking → 3 multi-directional + turning (+DR) → 4 terrain adaptation (uneven, slopes, obstacles, rough) → 5 robust
(mixed, stairs, stepping, pushes). Every `eval_freq` steps the current policy is evaluated deterministically in the stage's conditions; a stage ends only when
`survival ≥ min_survival` **and** `velocity error ≤ max_vel_error`, after at least `min_steps_per_stage`; inside a stage the terrain difficulty rises in steps.
Stages can be disabled individually (GUI or `curriculum.stages[i].enabled`). State is stored with every checkpoint and restored on resume. "Best model" =
furthest curriculum progress, then evaluation score.

### Evaluation (`rl/evaluation.py`)
Scenarios: flat stand / forward / backward / lateral / turn, uneven, slopes, stairs, obstacles, mixed (unseen seeds, deterministic policy). Metrics: reward, distance,
mean speed, velocity-tracking error, fall rate, completion rate, tilt, power, foot slip, terrain-completion rate → `results/eval_*.csv|json`. Verdict `PASS` only if all
flat-ground scenarios satisfy `evaluation.pass_*`; otherwise `NOT PASSED`.

## 5. Project layout
```
main.py  config.yaml  requirements.txt  README.md
assets/        go2_description.urdf (+ dae/ for meshes)
simulation/    physics_config, robot_loader (assets, URDF fixes, IK), robot_controller (PD/torque), pybullet_sim, sim_worker
environment/   go2_env, observation, reward, termination, domain_randomization
terrain/       terrain_generator + flat, uneven, stairs, slopes, obstacles, mixed
rl/            ppo_trainer, callbacks, curriculum, evaluation, model_manager
gui/           main_window, dashboard, simulation_view, training_panel, control_panel, terrain_panel, reward_panel, analytics_panel, model_panel, process_manager, results_dialog
utils/         config, logger, robot_validation
tests/         pytest suite (robot, env, terrain, ppo/curriculum, evaluation, gui)
models/ checkpoints/ logs/ results/
```

## 6. Testing
```bash
pytest tests -q          # ≈ 1 min; uses QT_QPA_PLATFORM=offscreen
```
Covers: URDF load, 12 joints found by name (non-sequential indices), no ghost mass, URDF limits, reset, standing for 3 s, torque/limit enforcement, 100+ env steps,
`gymnasium.check_env`, reproducibility, reward/termination/stall behaviour, DR, all terrain types + collision ray-casts, PPO init, short training, checkpoint
save/reload/resume, curriculum gating, evaluation reports and comparison, GUI launch, frame streaming, **GUI responsiveness during training**, pause/resume,
**clean worker shutdown**.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| `pip install pybullet` fails to build | install a C++ compiler (`build-essential` / Xcode CLT / MSVC Build Tools) or use conda-forge's `pybullet` |
| `qt.qpa.plugin: Could not load the Qt platform plugin "xcb"` | `sudo apt install libxcb-cursor0 libxkbcommon-x11-0 libegl1` ; on servers use `QT_QPA_PLATFORM=offscreen` (tests) |
| GUI robot is grey blocks | `.dae` meshes missing → copy them to `assets/dae/` (physics unaffected) |
| `RuntimeError: This probably means that you are not using fork to start your child processes…` | run through `python main.py` / keep the `if __name__ == "__main__":` guard in your own scripts (workers use `spawn`) |
| Training is slow | use `--n-envs <cores>`; each env is one process. 1 core ≈ 590 steps/s here |
| `--infer` shows nothing / `Not connected` | it needs a display (opens PyBullet's native OpenGL window); use the GUI otherwise |
| numpy 2.x errors with pybullet | `pip install "numpy<2"` |
| Robot jitters / unstable | lower `control.kp/kd`, or raise `simulation.solver_iterations`; check `python main.py --validate` |

## 8. Training results so far (honest status) and what to expect

One real run was made on a **single CPU core** (~560 steps/s, 1 environment, default PPO settings, 700 k steps ≈ 21 min):

* Stage 1 (standing balance) was learned within ~60 k steps and the curriculum advanced after a passing evaluation.
* In stage 2 (walking) the training return rose (≈ 17 → ≈ 700) but the **greedy policy still just stands**. Evaluating the 700 k checkpoint with
  `python main.py --evaluate …` gives `VERDICT: NOT PASSED`: speed ≈ 0.01–0.02 m/s with 0.4–0.5 m/s commanded; only `flat_stand` passes.
* The evaluator and the curriculum gate were hardened after this run so that standing still can never count as walking/turning
  (regression tests: `test_stationary_policy_never_passes_moving_scenarios`, `test_curriculum_does_not_pass_a_robot_that_just_stands_still`).

**So the pipeline is verified, but a walking policy is *not* delivered.** Legged locomotion by PPO typically needs tens of millions of
samples. Use all your cores:

```bash
python main.py --headless --train --n-envs 8 --timesteps 30000000        # ≈ 8× faster than one core; checkpoints every 100k steps
tensorboard --logdir logs/tensorboard                                      # watch reward/*, eval/*, curriculum/*
python main.py --evaluate --model models/best_model.zip                    # measured verdict, results/eval_*.csv|json
python main.py --headless --train --resume checkpoints/ckpt_0005000000.zip --timesteps 60000000   # continue later
```
The GUI shows the same run (Start Training), including the live reward breakdown, and *Simulation → Auto-follow training snapshot*
lets you watch the current policy while it trains.

If after a few million steps the evaluation still shows `speed ≈ 0` while commanded to walk (the "stand still" local optimum), try, in `config.yaml`:
`ppo.log_std_init: -0.3` (more exploration), `reward.weights.stall: -2.0`, `reward.air_time_threshold: 0.25`, `reward.sigma_velocity: 0.25`,
or temporarily disable stage-1/3 and run stage 2 only (`curriculum.stages[i].enabled: false`). These are suggestions – I have not tuned or verified them.
Known risk: with uniformly random actions the robot drifts at ≈ 0.6 m/s in simulation ("vibration locomotion"); the `foot_slip`, `action_rate`,
`joint_vel` and `stall` terms penalise it, but watch for it in the *Reward breakdown* plots and in the GUI.
