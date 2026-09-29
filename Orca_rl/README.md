# Orca_rl: In-Hand Cube Reorientation with the ORCA Hand (PPO)

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

The ORCA hand lies palm-up with a cube resting in the palm. The policy must turn the cube's red face to a commanded direction and hold it within 15° for 10 consecutive steps.
After each success a new goal is drawn and the episode continues. Training uses PPO from Stable-Baselines3; simulation uses the upstream [orca_sim](https://github.com/orcahand/orca_sim), unmodified.

## Current results

Goal success rate (goals solved / goal attempts). Fixed difficulty, 100 episodes per angle, two training seeds.

| Hand | Configuration | 30° | 45° | 60° | Curriculum angle reached in training |
|---|---|---|---|---|---|
| v2 | Baseline (run10 / run12) | 49% / 53% | 39% / 37% | 28% / 25% | 42° / 43° |
| v2 | + goal rotation vector + fingertips & contacts (run20 / run21) | 74% / 77% | 65% / 62% | 54% / 52% | 82° / 79° |
| **v1** | Baseline (run24, seed 0 only) | 29% | 22% | 15% | 27° |
| **v1** | + goal rotation vector + fingertips & contacts (run22 / run23) | 51% / 55% | 39% / 43% | 29% / 31% | 47° / 50° |

- On v2, adding the goal rotation vector ("which way to turn, and how far") and fingertip observations raised success by about 25–30 percentage points at every angle.
- **The physical hand is v1.** The same configuration does clearly worse on v1. The most likely reason is that the v1 simulation scene places the cube too far from the fingers (see `results/2026-09-27-run22-24-v1.md`); the scene needs to be aligned with the real setup.
- Full per-experiment records are in `results/`; a chronological log of problems found and fixed is in `orca_rl/README.md`. Both are written in Chinese.

- PPO on the teammates' generated tasks (pick up, gesture, pick-and-place): see `taskgen_rl/` and `results/2026-09-29-taskgen.md`.

- Demo videos (no text overlay): `demo_videos/`.

## Layout

```
Orca_rl/
├── orca_rl/
│   ├── task.py           # environment: reward, success test, curriculum, observation, action mapping
│   ├── train.py          # PPO training
│   ├── evaluate.py       # fixed-difficulty evaluation (use this, not the training curves)
│   ├── diagnose.py       # splits failures into "never reaches the goal" vs "cannot hold it"
│   ├── checks.py         # checks the reward cannot be gamed; run it after any reward change
│   ├── record.py         # video of one policy
│   ├── compare_video.py  # several policies side by side on the same seeds
│   └── README.md         # post-mortems from run1 to now
├── results/              # one record per experiment
└── requirements.txt
```

## Setup

Python 3.11. From the `Orca_rl/` directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m orca_rl.checks          # should print "all checks passed"
```

## Common commands

Run everything from the `Orca_rl/` directory. Outputs go to `runs/<name>/` (not tracked by git).

```bash
# Train: v1 hand, current best configuration
python -m orca_rl.train --name v1_s0 --version v1 --seed 0 --obs-rotvec --obs-fingertips \
    --timesteps 30_000_000 --n-envs 14 --n-steps 292

# Evaluate: fixed difficulty, 100 episodes
python -m orca_rl.evaluate --model runs/v1_s0/final_model.zip --goal-angle 45 --episodes 100

# Diagnose: does it fail to reach the goal, or fail to hold it?
python -m orca_rl.diagnose --model runs/v1_s0/final_model.zip --goal-angle 45 --episodes 60

# Videos
python -m orca_rl.record --model runs/v1_s0/final_model.zip --goal-angle 45 --out v1_s0.mp4
python -m orca_rl.compare_video --goal-angle 60 --out videos/compare.mp4 \
    --run runs/a "config A" --run runs/b "config B"
```

- About 6,000 steps/s on an M4 Mac (30M steps ≈ 1.5 h) and about 2,500 steps/s on an i9-11900F under WSL (≈ 3.5 h). Use `--device cpu`: the network is small and a GPU is slower.
- The environment settings used for training are written to `runs/<name>/env_kwargs.json`; evaluation, diagnosis and video scripts read them automatically, so the hand version and action scale cannot be mixed up.
- Stopping training with Ctrl-C or `kill` saves the model first. To resume from a checkpoint use `--resume runs/<name>/checkpoints/ppo_<N>_steps.zip`; `--timesteps` is then the number of additional steps.

## Action and control conventions

- Control period **10 ms**: MuJoCo timestep 2 ms, each action applied for 5 steps (orca_sim defaults).
- 17-dimensional action in [−1, 1], **relative** mode: each step adds `0.15 × a × halfspan` to the previous servo target (halfspan = half of that joint's range), then clips to the joint range.
  On v1 the servo target moves at most 4.7–9.9° per 10 ms. Action scales of 0.06 and 0.3 were both worse than 0.15.
- **v1 and v2 order their actuators differently** (v1: wrist, thumb ×4, index, middle, ring, pinky; v2: wrist, pinky, ring, middle, index, thumb), so a policy trained on one cannot run on the other.

## Still open before running on the physical hand

- The observation uses information only the simulator has: cube pose and angular velocity, red-face direction, and per-finger contact flags. On hardware these need a camera or motion capture, plus touch sensing or an estimate.
- No domain randomization yet (`--randomize-physics` is implemented but has not been trained).
- The v1 simulation scene (cube size, mass, placement, hand pose) needs to match the real setup.
