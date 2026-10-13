# Running the RL cube policy through the safety layer

[English](SAFETY.md) | [中文](SAFETY.zh-CN.md) | [한국어](SAFETY.ko.md)

Everything needed to put the trained in-hand cube-reorientation policy (v1 right hand, MuJoCo) behind
`Orca_safety` on your own machine: the code, six trained models, and the commands. No training is needed.

## What it does

Each 10 ms control step the policy outputs 17 joint targets (`q_nominal`). With the safety layer on, they go
through `orca_safety_v2.filter.CBFSafetyFilter` together with the measured joint angles, and the returned
`q_safe` is what the servos get. `q_safe` is also fed back as the policy's previous target. The hand's
self-clearance is measured after every step in both modes, with the same 108 link pairs the filter uses.

The filter is imported from your installed `orca_safety_v2`. If you install `Orca_safety` editable (below),
**any change you make under `Orca_safety/src/` is what gets tested** — nothing is copied.

## Setup (about 5 minutes)

Python 3.11. From the directory that contains `Orcahand-Safe-Assembly/`:

```bash
# 1. v1 URDF and collision meshes (skip if you already have a checkout: pass --urdf instead)
git clone https://github.com/orcahand/orcahand_description.git

# 2. a separate environment, so your teleop environment is not touched
uv venv safety_venv --python 3.11 --seed          # or: python3.11 -m venv safety_venv
safety_venv/bin/pip install "mujoco==3.12.0" \
    -r Orcahand-Safe-Assembly/Orca_rl/requirements.txt \
    -e "./Orcahand-Safe-Assembly/Orca_safety[geometry,safety]"

# 3. check
cd Orcahand-Safe-Assembly/Orca_rl
../../safety_venv/bin/python -m orca_rl.evaluate_safety --model models/run23/final_model.zip --episodes 4 --workers 4
```

- Install with the environment's own `pip`, not `uv pip`: `requirements.txt` installs orca_sim editable from
  git, which `uv pip` refuses.
- **`mujoco==3.12.0` is what our numbers were measured with** (except the two `run22_ft_*` rows, evaluated on
  another machine with 3.13.0). With 3.12.0 the check above prints exactly `57%` (off) and `11%` (on, 25.94%
  fallback) for those 4 episodes. Another MuJoCo version still runs, but contact dynamics differ slightly and
  individual episodes come out differently: 3.15.0 gave 38% and 20% on the same 4 episodes. We have not
  measured how far the 100-episode numbers move between versions, so compare off and on within one version.
- These steps were run from scratch in an empty directory on macOS (Apple silicon), Python 3.11.

The URDF is looked for in `Orcahand-Safe-Assembly/orcahand_description/` and next to `Orcahand-Safe-Assembly/`;
anywhere else, pass `--urdf /path/to/orcahand_description/v1/models/urdf/orcahand_right.urdf`.

## Commands

Run from `Orcahand-Safe-Assembly/Orca_rl/` with the environment's Python (`PY=../../safety_venv/bin/python`).

```bash
# Off vs on, same policy, same 100 reset seeds, 45-degree goals. ~1.5 min off, ~8 min on (8 processes, M4 Mac).
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --goal-angle 45 --episodes 100 \
    --out my_results/run23.json

# Your own safety parameters: copy Orca_safety's v1_right.yaml, edit it, point at it
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --modes on \
    --safety-config /path/to/my_v1_right.yaml --out my_results/run23_mycfg.json

# One JSON line per control step (q_current, q_nominal, q_safe, clearances, active pairs,
# fallback reason, per-stage timings) -> my_logs/run23.off.jsonl and my_logs/run23.on.jsonl
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --episodes 8 --log my_logs/run23

# Side-by-side video, off (left) and on (right); the bar is the smallest self-clearance
$PY -m orca_rl.record_safety --model models/run23/final_model.zip --seeds 0 1 --out off_on.mp4
```

Useful flags: `--modes off` / `--modes on` to run one side, `--workers N`, `--seed N` (first reset seed),
`--goal-angle` (harder goals are further rotations; 30 / 45 / 60 are the ones we report).

## Models (`models/`)

| Folder | What it is | Goal success at 45°, safety off / on |
|---|---|---|
| `run23` | v1 policy, seed 1, 30M steps, trained without the safety layer | 43% / 34% |
| `run22` | same, seed 0 | 39% / 36% |
| `run23_ft_ctrl` | run23 + 5M more steps, still without the safety layer | 50% / 31% |
| `run23_ft_safety` | run23 + 5M more steps with the filter in the training loop | 40% / 42% |
| `run22_ft_ctrl` | run22 + 5M more steps, without | 45% / 35% |
| `run22_ft_safety` | run22 + 5M more steps, with | 39% / 36% |

Each folder holds `final_model.zip`, `vecnormalize.pkl` (observation statistics, loaded automatically) and
`env_kwargs.json` (hand version, action mode and scale, read automatically).

## What we measured, and what may matter for the safety layer

Details: [`results/2026-10-12-safety-layer.md`](results/2026-10-12-safety-layer.md) and
[`results/2026-10-13-safety-finetune.md`](results/2026-10-13-safety-finetune.md) (Chinese).

- Without the filter the policy has links of different fingers touching for 57–74% of steps (MuJoCo's own
  contact detection agrees: 59% vs 56%). With it: 0–0.12%.
- The filter changes the target on ~97% of steps and falls back on 18–29% of them. In the 4-episode check
  all 415 fallbacks were `nonlinear_command_margin_violation`, none a QP failure (the summary prints the
  reasons). On the teleop recordings in `Orca_safety`'s benchmark fallback was 0.2–0.7%, all from the QP
  residual check, so this policy is a much harsher input.
- The reset pose has 6.2 mm between `right_middle_mp` and `right_ring_mp`, the same pair and distance as in
  the safety benchmark's `open` scenario, so with a 15 mm activation distance some constraints are always active.
- The filter does not know about the cube. The cube pushes fingers together, so clearance is still under
  5 mm on 19–30% of steps with the filter on.
- Filter time per step on this workload: 65–80 ms mean (M4 Mac), about half of it finite-difference
  gradients. `orca_rl/safety_fast.py` computes the same gradients from the closest points
  (23 ms → 3 ms, max difference 7e-7 m/rad over 1857 active rows). It is only used by `train.py --safety`;
  the evaluation above runs the filter as it is in `Orca_safety/src/`.

## Training with the filter in the loop (optional, slow)

```bash
$PY -m orca_rl.train --name my_ft_safety --version v1 --obs-rotvec --obs-fingertips --seed 1 \
    --n-envs 10 --n-steps 409 --timesteps 5000000 --curriculum-start-deg 45 \
    --resume models/run23/final_model.zip --safety
```

About 175 steps/s on an M4 Mac (5M steps ≈ 8 h). Output goes to `runs/my_ft_safety/` (not tracked by git).
