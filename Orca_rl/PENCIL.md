# Orca_rl: Pencil Spinning with the ORCA v1 Hand (PPO)

[English](PENCIL.md) | [中文](PENCIL.zh-CN.md) | [한국어](PENCIL.ko.md)

The ORCA v1 right hand (the physical hand's version) lies palm-up with a pencil across its palm and fingers.
The task is to keep turning the pencil about the vertical axis — counter-clockwise seen from above —
without dropping it. There is no goal and no success bonus: the reward is the yaw the pencil actually gains,
so an episode's return is roughly the number of radians it was turned.

This document covers the scene, **the reward function in detail**, the observation and action, every
environment and training parameter, a reference for every function, and the results so far.

**Status (2026-10-15): the task trains, but the policy does not spin continuously yet.** The trained model
`pencil1` turns the pencil about 90° in the first 2.5 s of an episode, then closes the hand into a fist
around it and holds still. A fix (`--pose-penalty`) is implemented and checked but not trained yet.

Demo: [`demo_videos/7_pencil_v1.mp4`](demo_videos/7_pencil_v1.mp4) (pencil1, reset seeds 0–2, not selected, half speed).

## Contents

1. [Results](#1-results)
2. [Quickstart](#2-quickstart)
3. [Files](#3-files)
4. [Scene and episode](#4-scene-and-episode)
5. [Reward function](#5-reward-function)
6. [Observation](#6-observation)
7. [Action and control](#7-action-and-control)
8. [Environment parameters](#8-environment-parameters)
9. [Training: PPO and logged metrics](#9-training-ppo-and-logged-metrics)
10. [Function reference](#10-function-reference)
11. [Known issues and next steps](#11-known-issues-and-next-steps)

## 1. Results

Deterministic policy, 10 s episodes, evaluation seed 1234, MuJoCo 3.12.0.

| Policy | Revolutions / episode | Mean spin rate | Drop rate |
|---|---|---|---|
| Do nothing (30 episodes) | 0.00 | 0.1°/s | 10% |
| `pencil_smoke`, 3M steps (50 episodes) | 0.13 | 4.7°/s | 4% |
| `pencil1` checkpoint 5M (50 episodes) | 0.10 | 3.9°/s | 8% |
| `pencil1` checkpoint 10M | 0.16 | 6.0°/s | 12% |
| `pencil1` checkpoint 15M | 0.25 | 9.9°/s | 28% |
| `pencil1` checkpoint 20M | 0.30 | 11.5°/s | 8% |
| `pencil1` checkpoint 25M | 0.33 | 13.3°/s | 28% |
| **`pencil1` final, 30M (100 episodes)** | **0.29** (max 0.55) | 11.9°/s | 14% |
| Reference: scripted oracle at the 1 rad/s cap | 1.59 | 57°/s | 0% |

"Revolutions" is the net yaw of the pencil's axis in the commanded direction, counted only while the pencil
is held (Section 5.2). The mean spin rate divides by all episode time, dropped episodes included.

**Only the start of each episode turns.** pencil1's episodes split into four 2.5 s windows
(episodes without a drop only):

| Evaluation seed | 0–2.5 s | 2.5–5 s | 5–7.5 s | 7.5–10 s | Still turning (>10°) in the last 5 s |
|---|---|---|---|---|---|
| 1234 (48 of 60 episodes) | 82° | 19° | 4° | 1° | 10% |
| 0 (54 of 60 episodes) | 93° | 18° | 3° | 0° | 9% |

**What the hand is doing when it stops** (30 episodes, mean):

| Time | Mean finger flexion (4 fingers) | Wrist | Servo targets at a joint limit | Fingers touching the pencil | Pencil speed |
|---|---|---|---|---|---|
| 1.0 s | 0.00 rad | −0.10 | 28% | 1.6 | 128 mm/s |
| 2.5 s | 0.39 rad | −0.40 | 41% | 1.8 | 55 mm/s |
| 10 s | 1.04 rad | −0.60 | 75% | 2.6 | 12 mm/s |

The hand starts flat (flexion 0, wrist −0.30). After one push it closes into a fist around the pencil.
To keep turning it would have to let go and regrasp, which risks a drop (−2 and the rest of the episode);
a clamped pencil earns nothing but cannot fall. The cube task hit the same failure ("clamp and freeze",
`orca_rl/README.md`, post-mortem 1, error 2). Full records: `results/2026-10-14-pencil1.md` and
`results/2026-10-14-pencil-spin-smoke.md` (in Chinese).

Training pencil1 took 1 h 41 min on an M4 Mac (10 envs, about 5,000 steps/s).

## 2. Quickstart

Python 3.11, from the `Orca_rl/` directory, in the **training** environment
(`env/requirements-train.lock.txt`, see `env/README.md`). No change to orca_sim is needed: the pencil scene
ships with `orca_rl` (Section 10.1).

```bash
# 1. Check the reward (about 1 minute). Must end with "all checks passed".
python -m orca_rl.pencil_checks

# 2. Evaluate the trained model (100 episodes reproduce 0.29 rev / 14% drop / return 1.27 +- 1.23)
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 100

# 3. Baselines
python -m orca_rl.pencil_eval --policy zero --episodes 30
python -m orca_rl.pencil_eval --policy random --episodes 10

# 4. Video: with a text overlay (640x480), or demo footage without text (1280x720)
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --video pencil1.mp4
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --seed 0 \
    --video 7_pencil_v1.mp4 --clean

# 5. Train (pencil1's settings; about 1.5-2.5 h on an M4 Mac)
python -m orca_rl.pencil_train --name pencil1 --timesteps 30_000_000 --n-envs 10

# 6. Next experiment: the same, plus the pose penalty (Section 5.6)
python -m orca_rl.pencil_train --name pencil2 --pose-penalty 5e-4 --timesteps 30_000_000 --n-envs 10

tensorboard --logdir runs/
```

- Outputs go to `runs/<name>/`: `final_model.zip`, `vecnormalize.pkl`, `env_kwargs.json`, `checkpoints/`
  (every 1M steps), `tb/`. The evaluation script reads `env_kwargs.json`, so the reward and action settings
  of a model cannot be mixed up.
- Stopping with Ctrl-C or `kill <pid>` saves the model first. `--resume runs/<name>/checkpoints/ppo_<N>_steps.zip`
  continues a run; `--timesteps` is then the number of additional steps.
- Use the training environment for training. In the safety-layer environment (`safety_venv`, numpy 2.3.5),
  2 of 5 multi-process starts died while a worker process was importing numpy
  (`RuntimeError: Could not parse python long as longdouble`), before any task code ran. The training
  environment (numpy 2.4.6) never did. Evaluation and the checks run fine in both.

## 3. Files

```
Orca_rl/
├── PENCIL.md / PENCIL.zh-CN.md / PENCIL.ko.md   # this document
├── orca_rl/
│   ├── pencil.py          # the task: PencilSpin (scene loading, reward, observation, reset)
│   ├── pencil_checks.py   # scripted probes that prove the reward cannot be gamed
│   ├── pencil_train.py    # PPO training (reuses train.py's callbacks and vectorization)
│   ├── pencil_eval.py     # evaluation and video
│   ├── scenes/
│   │   └── scene_right_pencil_spin_v1.xml   # v1 hand + pencil (template, see 10.1)
│   └── train.py           # build_vec_env / make_env_fn gained an env_cls argument
├── models/pencil1/        # final_model.zip, vecnormalize.pkl, env_kwargs.json
├── results/2026-10-14-pencil-spin-smoke.md, 2026-10-14-pencil1.md
└── demo_videos/7_pencil_v1.mp4
```

The task reuses the cube task's hand, action mapping, fingertip detection and physics randomization from
`orca_rl/task.py`, and the cube task's tuned PPO settings.

## 4. Scene and episode

### 4.1 The pencil

| Property | Value | Note |
|---|---|---|
| Length | 170 mm | Real pencils are 175–190 mm; slightly shorter so the ends clear the thumb |
| Diameter | 8 mm | Real: 7 mm; slightly thicker so contacts with the hand's meshes stay well-conditioned |
| Mass | 8 g | |
| Inertia (diagonal) | 6.4e-8, 1.93e-5, 1.93e-5 kg·m² | Solid rod: `m r²/2` about the long axis, `m(3r² + L²)/12` across |
| Collision shape | One capsule along the body's local x axis | +x is the tip, −x the eraser; tip, ferrule and eraser are visual only |
| Friction | 1.0 sliding, 0.005 torsional, 0.0001 rolling | Same as the hand's geoms |
| Spawn | (0.17, 0.045, 0.17) m | 8 mm above where it comes to rest (z ≈ 0.162), over the finger roots |

Centred further back (x = 0.15) the eraser end landed on the tower behind the palm and the pencil rested
tilted 20°.

### 4.2 Hand pose (`PENCIL_HAND_POSE`)

The cube task's palm-up pose with the fingers brought together and the wrist bent back. Joints not listed
keep the cube task's reference values (thumb_mcp 0, thumb_abd −0.733, thumb_pip −0.585, thumb_dip −0.505,
all finger mcp/pip 0).

| Joint | Value (rad) | Cube-task value |
|---|---|---|
| `right_wrist` | −0.30 | −0.15 |
| `right_index_abd` | −0.15 | −0.40 |
| `right_middle_abd` | 0.00 | 0.00 |
| `right_ring_abd` | −0.05 | 0.17 |
| `right_pinky_abd` | 0.08 | 0.52 |

- In the cube pose the fingertips splay about 3 cm apart. Of 6 initial yaws tried (pencil centred at
  x = 0.15), the pencil fell through
  the gaps at 60°, 90° and 120°. With this pose it stayed on the hand at all 12 yaws tried (3 s settle each),
  resting 3–4° off level.
- The wrist servo sags under the hand's weight, −0.30 → −0.21, which leaves the palm about level.
- Pressing the fingers tighter makes them collide with each other, which is exactly what the team's safety
  layer exists to prevent.

### 4.3 Reset

1. Physics randomization, if enabled (Section 8).
2. Hand to `PENCIL_HAND_POSE`; pencil to the spawn point with a uniform ±5 mm xy offset and a uniform random
   yaw in [0°, 360°).
3. **Settle for 100 control steps (1 s)** with the servo targets held at the pose. After 100 steps the yaw
   drifts 0.6° on average over the next 2 s (max 2.5°); after only 60 steps a pencil could still be rolling
   about its axis at 26 rad/s.
4. **Rejection:** if the settled pencil is not held, or tilted more than 15°, draw again (up to 10 attempts;
   only the first is seeded). Over 400 random resets 5.8% needed a redraw and none needed more than two.
   The case inspected was a pencil at 45° shifted 5 mm toward the thumb: it spawned inside the thumb and was
   thrown to the floor while settling.
5. The first action continues from the servo targets held while settling (re-deriving them from the sagged
   joint angles would let every joint go slack on step one), and `q_rest` — the settled joint angles — is
   stored for the optional pose penalty.

### 4.4 Episode end

| Condition | Result |
|---|---|
| Pencil centre below 0.12 m (`drop_height`) | `terminated`, reward −2 on that step |
| 1000 steps (10 s) | `truncated` |
| MuJoCo state became NaN/Inf | `truncated`, reward 0, `info["sim_unstable"] = True` (Section 11) |

## 5. Reward function

### 5.1 Formula

Per control step (10 ms):

```
r_t =   spin_coef           * paid_t                                   # yaw gained, rate-capped (5.3-5.5)
      - action_rate_penalty * sum_j ((target_t,j - target_t-1,j) / halfspan_j)^2
      - pose_penalty        * sum_j (q_t,j - q_rest,j)^2                # optional, off by default
      - drop_penalty        * [pencil dropped on this step]
```

| Term | Default | Largest value per step | Per 10 s episode |
|---|---|---|---|
| `spin_coef * paid` | 1.0 | +0.01 (1 rad/s × 10 ms) | at most +10 |
| Action rate | 0.002 | −0.00077 (all 17 joints moving the full 0.15 halfspan) | — |
| Pose | 0 (try 5e-4) | pencil1's final fist (≈9 rad²) at 5e-4: −0.0045 | — |
| Drop | 2.0 | −2, once; the episode ends | — |

`j` runs over the 17 actuated joints; `halfspan_j` is half of joint `j`'s control range; `q` are measured
joint angles. The return of a perfect 10 s episode is therefore about 10 (the oracle in Section 5.8 scores
9.98).

### 5.2 When yaw counts

Yaw is only counted on a step where both hold:

- **The pencil is held**: its centre is above `in_hand_height` = 0.13 m **and** it touches the palm or any
  link of a finger (`_contacts`). The tower behind the palm is part of the hand model, but a pencil resting
  on it is not held.
- **The pencil is near level**: its axis is within `max_tilt_deg` = 45° of horizontal.

On other steps no yaw is paid and nothing is billed yet: the reference heading is frozen (Section 5.5).

### 5.3 Measuring yaw: the heading of the pencil's axis

Let `a` be the pencil's long axis (its local x axis, eraser → tip) in world coordinates. The heading is the
direction of its horizontal projection:

```
psi = atan2(a_y, a_x)
spin_t = sigma * wrap(psi_t - psi_ref)        # sigma = +1 for "ccw", -1 for "cw"; wrap() into [-pi, pi)
```

`psi_ref` is the heading at the last counted step. Why this and not the angular velocity `ω_z`:

- **Rolling a round pencil about its own axis** — the cheapest motion there is on a palm — has angular
  velocity along `a`, which leaves `a` unchanged. It contributes nothing to `psi`. If the pencil were tilted,
  it would contribute to `ω_z`.
- **A pencil stood on end and twirled** has a near-zero horizontal projection; it is excluded by the 45° tilt
  limit anyway.
- Differencing the heading over a step measures the rotation that actually happened, rather than an
  instantaneous velocity sample.

### 5.4 The rate cap and the carry

Paying `spin_t` directly would make a flick — throwing the pencil into a free spin — worth more than turning
it steadily. So forward yaw is capped at `max_spin_rate` = 1 rad/s. The cap is applied **only to forward
yaw**, and forward yaw over the cap is **carried** (up to `spin_carry_rad` = 0.2 rad) to later steps:

```
on a counted step:
    k       = steps since the last counted step          # 1 normally; more after a freeze
    gained  = spin_t + carry
    paid_t  = min(gained, max_spin_rate * dt * k)        # dt = 0.01 s
    carry   = min(gained - paid_t, spin_carry_rad)       # never negative
    psi_ref = psi_t
on any other step:
    paid_t  = 0                                          # psi_ref and carry are kept
```

**Invariant.** On every counted step `carry_old + spin_t = paid_t + carry_new + discarded`, with
`carry ≥ 0` and `discarded ≥ 0`. Summed over an episode:

```
sum(paid) = sum(spin) - carry_final - sum(discarded)  <=  sum(spin) = net counted rotation
```

**The return can never exceed the net rotation**, whatever the policy does. And since each step pays at most
`max_spin_rate * dt`, an episode pays at most 1 rad/s × 10 s = 10.

Why each piece is there:

| Design | What goes wrong without it | Measured (`pencil_checks.py`) |
|---|---|---|
| Cap forward yaw | Flicking the pencil into a fast free spin pays more than turning it steadily | Turning at 5× the cap pays 10.00, the same as at the cap (9.98) |
| Cap **only** forward yaw | With a cap on both sides, creep forward slowly (paid in full), snap back fast (billed only up to the cap): zero net rotation, positive reward | Ratchet: 0.05 here; a two-sided cap would pay **3.41** |
| Carry the excess | A pencil pressed by servoing fingers shivers. Each shiver's forward half is cut at the cap, its backward half billed in full, so shivering becomes a hidden cost | Turning 0.5 rad/s with a ±2° shiver: 4.90 here (5 rad turned); without the carry **−10.63** |
| Limit the carry to 0.2 rad | An unlimited carry would bank a flick and pay it out later | — |

The carry was added after the first smoke run. Without it, under random actions the net yaw was −0.15 rad
but the reward paid −3.54, and in training `task/paid_frac` (reward-paid yaw / net yaw) was only 0.13–0.29 at
300k–400k steps. With the carry it was 0.66–0.83 at the same stage and 0.93–0.97 by the end.

### 5.5 Freezing while the pencil is not counted

While the pencil is not held or tilted past 45°, `psi_ref` stays where it was. When the pencil is counted
again, `spin` is the whole change since the freeze, and the cap allows `k` steps' worth. Without this a
policy could be paid for turning the pencil forward in the hand, toss it, turn it back in the air
(not billed), catch it, and repeat. Measured: forward 0.5 rad in hand, back 1.0 rad in the air, per cycle,
scores **−4.85** per episode; without the freeze it would score positive.

### 5.6 Penalties

- **Action rate** (`action_rate_penalty` = 0.002): `Σ_j (Δtarget_j / halfspan_j)²`, the same smoothness term
  as the cube task. Each step's change is at most 0.15 halfspan per joint, so this is at most 0.00077 per
  step: it shapes smoothness without competing with the task.
- **Drop** (`drop_penalty` = 2.0): charged once when the pencil centre falls below 0.12 m, and the episode
  ends, so a drop also loses the rest of the episode. Kept well below the 10 a full episode can earn: in the
  cube task a drop penalty larger than the success bonus taught the policy to clamp the object and freeze.
- **Pose** (`pose_penalty`, default 0): `Σ_j (q_j − q_rest_j)²` over the 17 joints, against the angles the
  hand settled in at reset (not the commanded pose, so gravity's 0.09 rad wrist sag costs nothing and an idle
  hand pays exactly 0). It exists because of pencil1's fist (Section 1). At 5e-4:

  | Hand state | Σ(q − q_rest)² | Cost per step | Share of the maximum spin reward per step |
  |---|---|---|---|
  | Idle | 0 | 0 | 0% |
  | All 8 finger mcp/pip joints at 0.3–0.4 rad | 0.72–1.28 rad² | 0.0004–0.0006 | 4–6% |
  | pencil1's final fist (estimated from its mean flexion) | ≈ 9 rad² | ≈ 0.0045 | ≈ 45% |
  | Scripted fist in `pencil_checks`, finger joints driven to their limits | — | ≈ 0.0105 | ≈ 105% (−10.53 per episode) |

### 5.7 What the reward deliberately does not have

| Not included | Why |
|---|---|
| Success bonus | There is no discrete goal; any bonus for "a revolution" would be a step function of the same yaw the reward already pays continuously |
| Alive / holding bonus | Pays the policy for doing nothing — the stock cube task's stalling exploit (`orca_rl/README.md`) |
| Constant per-step cost | Makes ending the episode early (dropping on purpose) attractive |
| Reward on `ω_z` | Pays for rolling a tilted pencil about its own axis (Section 5.3) |
| Pencil position term | No sign yet that the pencil wanders; a drop already ends the episode |

### 5.8 Scripted checks (`python -m orca_rl.pencil_checks`)

Most probes move the pencil directly; they are not policies a hand could execute, they are tests of what
the reward pays for. 10 episodes of 10 s each:

```
policy (10 episodes of 10 s)             return    revs   drop
--------------------------------------------------------------
do nothing                                -0.20    0.00    10%
random                                    -2.12   -0.00   100%
oracle: ccw at the cap, 1 rad/s            9.98    1.59     0%
oracle: ccw at 5x the cap                 10.00    7.96     0%
oracle: wrong way (cw), 1 rad/s          -10.01   -1.59     0%
ratchet: creep fwd, snap back              0.05    0.01     0%
0.5 rad/s with a +-2 deg shiver            4.90    0.78     0%
roll about its own axis, 20 rad/s          0.02    0.00     0%
stand it 60 deg up and turn                0.00    0.00     0%
toss: fwd in hand, back in air            -4.85   -0.77     0%

with --pose-penalty 0.0005:
  do nothing                              -0.20    0.00    10%
  oracle: ccw at the cap, 1 rad/s          9.98    1.59     0%
  close a fist and hold                  -10.73    0.03    10%
```

Assertions: doing nothing earns ≈ 0; turning at the cap for 10 s earns ≈ 10; 5× faster earns no more; the
wrong direction is charged in full; ratcheting earns nothing; shivering costs nothing; rolling in place and
turning a standing pencil earn nothing; yaw lost in the air is billed on landing; with the pose penalty,
idling and the oracle are unchanged and a fist costs more than 2 per episode. The random row varies a
little between runs (its actions are not seeded). "Do nothing" drops 10% of the time because of the slow
slip described in Section 11.

## 6. Observation

90 values, float64, normalized during training by `VecNormalize` (running mean/std, clipped to ±10).

| Index | Size | Content | Frame |
|---|---|---|---|
| 0–16 | 17 | Hand joint angles (actuator order, Section 7) | — |
| 17–19 | 3 | Pencil centre position | world |
| 20–23 | 4 | Pencil orientation quaternion (w, x, y, z) | world |
| 24–40 | 17 | Hand joint velocities | — |
| 41–43 | 3 | Pencil linear velocity | world |
| 44–46 | 3 | Pencil angular velocity | pencil body |
| 47–49 | 3 | Pencil axis `a` (eraser → tip), unit vector | world |
| 50–52 | 3 | Pencil angular velocity | world |
| 53–69 | 17 | Current servo targets, mapped to [−1, 1] | — |
| 70–84 | 15 | Fingertip positions minus the pencil centre: thumb, index, middle, ring, pinky (x, y, z each) | world |
| 85–89 | 5 | Whether any link of each finger touches the pencil: thumb, index, middle, ring, pinky (0/1) | — |

- The world frame is also the hand's frame: the hand does not move.
- The servo targets (53–69) make the task Markov in the action: in relative mode the action is a change to
  them, and `kp · (target − q)` is the grip force. In the cube task, before they were observed, the policy
  drove its targets bang-bang into the joint limits 77% of the time (post-mortem 4).
- Fingertip offsets and contact flags halved the cube task's drop rate (post-mortem 11), so they are always on.
- With `--randomize-physics`, Gaussian noise (σ = 0.005 rad) is added to the 17 hand joint angles.
- Indices 17–23, 41–52, 70–89 are simulator-only information (Section 11).

## 7. Action and control

- **Timing:** MuJoCo timestep 2 ms, each action held for 5 steps → 10 ms control period (100 Hz).
- **Actuators:** 17 position servos, `kp` = 2.0 N·m/rad, force limit ±1 N·m (orca_sim's v1 model).
- **Action:** 17 values in [−1, 1]. Default `relative` mode:
  `target_t = clip(target_t−1 + 0.15 · a · halfspan, low, high)` — each step moves a servo target by at most
  15% of half its range. `absolute` mode maps `a` straight onto the range.
- **Actuator order (v1):** wrist; thumb mcp, abd, pip, dip; index abd, mcp, pip; middle abd, mcp, pip;
  ring abd, mcp, pip; pinky abd, mcp, pip. v2 orders them differently, so policies are not interchangeable.

| Actuator | Range (rad) | | Actuator | Range (rad) |
|---|---|---|---|---|
| wrist | −0.674 … 0.897 | | middle_abd | −0.646 … 0.646 |
| thumb_mcp | −0.873 … 0.873 | | middle_mcp | −0.349 … 1.588 |
| thumb_abd | −1.082 … 0.000 | | middle_pip | −0.349 … 1.868 |
| thumb_pip | −0.794 … 1.230 | | ring_abd | −0.476 … 0.806 |
| thumb_dip | −0.854 … 1.450 | | ring_mcp | −0.349 … 1.588 |
| index_abd | −1.046 … 0.246 | | ring_pip | −0.349 … 1.868 |
| index_mcp | −0.349 … 1.658 | | pinky_abd | −0.122 … 1.169 |
| index_pip | −0.349 … 1.885 | | pinky_mcp | −0.349 … 1.710 |
| | | | pinky_pip | −0.349 … 1.885 |

## 8. Environment parameters

`PencilSpin(render_mode=None, version="v1", **kwargs)`. Every value used for training is written to
`runs/<name>/env_kwargs.json`.

| Argument | Default | Meaning |
|---|---|---|
| `version` | `"v1"` | Only v1 has a pencil scene; anything else raises |
| `max_episode_steps` | 1000 | Episode length (10 s) |
| `spin_direction` | `"ccw"` | `"ccw"` (+z, counter-clockwise from above) or `"cw"` |
| `max_spin_rate` | 1.0 | rad/s; forward yaw above this rate is not paid (5.4) |
| `spin_carry_rad` | 0.2 | rad of over-cap forward yaw carried to later steps (5.4) |
| `spin_coef` | 1.0 | Weight of the spin term |
| `drop_penalty` | 2.0 | Charged once on a drop (5.6) |
| `pose_penalty` | 0.0 | Per rad² from the settled pose (5.6); 5e-4 is the proposed value |
| `action_rate_penalty` | 0.002 | Smoothness term (5.6) |
| `max_tilt_deg` | 45.0 | Yaw counts only within this of level (5.2) |
| `in_hand_height` | 0.13 | m; held requires the pencil centre above this (5.2) |
| `drop_height` | 0.12 | m; below this the episode ends as a drop |
| `action_mode` | `"relative"` | `"relative"` or `"absolute"` (7) |
| `action_scale` | 0.15 | Fraction of the halfspan a relative action can move per step |
| `randomize_physics` | False | Per reset: pencil mass ±30%, pencil friction ±30%, joint damping ±30%, actuator gain ±20%, joint-angle noise σ 0.005 rad |
| `randomization` | None | Dict overriding those ranges (keys `cube_mass`, `cube_friction`, `joint_damping`, `actuator_gain`, `obs_noise_rad`; names shared with the cube task) |
| `reset_settle_steps` | 100 | Control steps the pencil settles before the episode starts (4.3) |
| `initial_yaw` | `"random"` | `"random"` (uniform 0–360°) or `"nominal"` (0°) |
| `pencil_pos_xy_jitter` | 0.005 | m, uniform xy offset of the spawn point |

Fixed in code: up to 10 reset attempts, 15° maximum settled tilt (4.3); `PENCIL_HAND_POSE` (4.2).

`reset(seed=None, options=None)` accepts `options={"cube_quat": q, "cube_pos": p}` to pin the pencil's spawn
(no rejection sampling then), plus the parent class's options (`hand_pose_by_joint`, `hand_qpos`, …).

`info` returned by `reset` and `step`:

| Key | Meaning |
|---|---|
| `yaw_deg`, `tilt_deg` | Heading of the pencil axis, and its angle from horizontal |
| `pencil_pos`, `pencil_height` | Pencil centre (m) |
| `in_hand`, `dropped` | Held (5.2); below `drop_height` |
| `elapsed_steps` | Steps so far in the episode |
| `turned_rad`, `turned_revs` | Net counted yaw so far in the commanded direction |
| `paid_rad` | The part of it the reward has paid |
| `in_hand_frac` | Fraction of steps held so far |
| `spin_this_step` | `spin_t` on this step (0 if not counted); `step` only |
| `sim_unstable` | Present and True when the step was cut off by a MuJoCo blow-up |

## 9. Training: PPO and logged metrics

Stable-Baselines3 2.9.0 PPO. The values are the cube task's tuned settings; `orca_rl/README.md` explains why
each differs from SB3's defaults.

| Flag | Default | Note |
|---|---|---|
| `--n-envs` | 10 | Parallel environments (`SubprocVecEnv`) |
| `--n-steps` | 512 | Steps per environment per rollout → 5,120 transitions per update |
| `--batch-size` | 1024 | 5 minibatches per epoch |
| `--n-epochs` | 10 | Upper bound; `target_kl` usually stops earlier (pencil1 ended at 2 per update) |
| `--lr` | 1.5e-4 | Constant |
| `--gamma` | 0.995 | ≈ 2 s horizon at 100 Hz |
| `--gae-lambda` | 0.95 | |
| `--clip-range` | 0.2 | |
| `--ent-coef` | 0.0 | An entropy bonus drove the action std to 11 in the cube task |
| `--vf-coef` | 0.5 | |
| `--max-grad-norm` | 0.5 | |
| `--target-kl` | 0.02 | Ends the epoch loop early |
| `--log-std-init` | −0.5 | Initial action std 0.61 (std 1 saturates [−1, 1]) |
| `--max-log-std` | 0.0 | Hard cap: std ≤ 1, applied after every rollout |
| `--device` | `cpu` | The network is small; a GPU is slower |
| `--checkpoint-every` | 1,000,000 | Model and normalization statistics |
| `--seed` | 0 | Environment `i` is reset with `seed + i` |

Task flags: `--max-episode-steps`, `--spin-direction`, `--max-spin-rate`, `--spin-carry`, `--drop-penalty`,
`--pose-penalty`, `--max-tilt-deg`, `--action-mode`, `--action-scale`, `--randomize-physics`
(same meaning as Section 8). `--resume MODEL.zip` continues training; hyperparameters are re-read from the
command line.

**Network:** `MlpPolicy`, separate actor and critic, each 90 → 256 → 256 with tanh; a linear head to 17 action
means (actor) or 1 value (critic); a state-independent learned log-std per action. 182,819 parameters.
**Normalization:** `VecNormalize` of observations and rewards (reward normalization uses the same γ).

**TensorBoard** (`task/` is written every 40 finished episodes; it is training data with exploration noise —
use `pencil_eval.py` for numbers):

| Metric | Meaning |
|---|---|
| `task/revs_per_episode` | **Progress metric.** Mean counted revolutions per episode (oracle: 1.59) |
| `task/spin_rate_deg_s` | Counted yaw per second of episode, drops included |
| `task/paid_frac` | Reward-paid yaw / counted yaw. Near 1: steady turning. Well below 1: over the cap or shivering |
| `task/drop_rate` | Fraction of episodes ending in a drop |
| `task/episode_length` | Mean steps per episode |
| `task/in_hand_frac` | Fraction of steps the pencil is held |
| `task/unstable_rate` | Fraction of episodes cut off by a MuJoCo blow-up; should stay near 0 |
| `train/epochs_used` | PPO epochs actually run before `target_kl` stopped them |
| `train/std` | Action std; pencil1 went 0.61 → 0.30 |

## 10. Function reference

### 10.1 `orca_rl/pencil.py`

| Name | What it does |
|---|---|
| `PENCIL_HAND_POSE` | Joint-angle overrides for the reset pose (4.2) |
| `SCENE_TEMPLATE` | Path of `scenes/scene_right_pencil_spin_v1.xml` |
| `pencil_scene_path()` | Fills the template's `{orca_sim}` placeholders with the installed orca_sim package's path and writes the result to `<tmp>/orca_rl/scene_right_pencil_spin_v1_<hash>.xml` (atomic rename, so parallel workers never read a partial file). This keeps orca_sim the unmodified upstream package: MuJoCo resolves includes relative to the including file, so the template names orca_sim's floor scene and v1 hand model by absolute path. The compiled model was checked identical, field by field, to the earlier in-package scene |
| `_wrap(angle)` | Wraps an angle into [−π, π) |
| `saved_pencil_kwargs(model_path)` | Reads the `env_kwargs.json` next to a model (or one level up, for checkpoints) and drops the `task` tag |

`class PencilSpin(OrcaHandRightCubeOrientation)` — the pencil plays the role of the parent's "cube" (`_cube_body_id`, `_cube_qpos_adr`, …).

| Method | What it does |
|---|---|
| `__init__(...)` | Validates arguments, loads the scene via `pencil_scene_path()`, builds the [−1, 1] action space, collects the pencil's collision geoms, finds the fingers (`_setup_fingers`), stores nominal physics for randomization, builds the observation space |
| `_axis()` | Pencil's local x axis in world coordinates (eraser → tip) |
| `_yaw()` | `atan2(a_y, a_x)` — heading of the axis's horizontal projection |
| `_tilt_rad()` | `asin(|a_z|)` — angle between the axis and horizontal |
| `_angvel_world()` | Rotates the free joint's body-frame angular velocity into the world frame |
| `_contacts()` | One pass over MuJoCo contacts: (pencil touches palm or a finger, 5 per-finger flags) |
| `_in_hand()` | Pencil centre above `in_hand_height` and touching palm or finger |
| `_resolve_initial_cube_quat(options)` | Spawn orientation: `options["cube_quat"]`, else a random (or zero) yaw about world z |
| `_get_obs()` | The 90-value observation (Section 6) |
| `_get_info()` | The `info` dict (Section 8) |
| `_reset_tracking()` | Zeroes the episode's yaw bookkeeping: `psi_ref`, `k`, carry, counted and paid yaw, held steps |
| `reset(seed, options)` | Randomize → place → settle → reject and redraw if needed → continue from the held servo targets, store `q_rest` (4.3) |
| `step(action)` | Map action → step MuJoCo 5× → detect a blow-up → held? dropped? → counted yaw with cap and carry → reward (5.1) → terminated / truncated → obs, reward, flags, info |

Borrowed unchanged from `CubeReorientContinuous` (`task.py`): `_setup_fingers` (finds each finger's links and
its fingertip collision geom from the kinematic tree), `_apply_randomization` (per-reset physics
randomization), `_settle_cube` (holds the servo targets for `reset_settle_steps`), `_target_from_action`
(relative/absolute action mapping and clipping). Inherited from orca_sim: `_cube_pos`, `_cube_qvel`,
`_cube_dropped`, `_compose_ctrl_from_qpos`, `_quat_from_axis_angle`, `render`, `close`.

### 10.2 `orca_rl/pencil_checks.py`

| Name | What it does |
|---|---|
| `run(policy, seeds, steps, **env_kwargs)` | Runs a scripted policy for 10 seeds; returns mean return, revolutions, drop rate, and what a two-sided cap and a no-carry cap would have paid |
| `place(env, pos, quat, angvel_body)` | Poses the pencil directly (then `mj_forward`), returns a zero action |
| `yaw_quat(yaw, tilt)` | Quaternion of a pencil at that heading, raised `tilt` from level |
| `zero`, `random` | Baselines |
| `turning(rate)` | Oracle: turns the pencil flat on the palm at `rate` rad/s |
| `ratchet` | Creeps forward 0.2 rad at 0.5 rad/s, snaps back 0.2 rad at 4 rad/s |
| `shiver` | Turns at 0.5 rad/s with a ±2° shiver every 2 steps |
| `roll` | Spins the pencil about its own axis at 20 rad/s |
| `standing` | Turns the pencil while it stands 60° up |
| `toss` | Forward 0.5 rad in hand, back 1.0 rad 5 cm up in the air, set down |
| `fist` | Curls the four fingers to their limits and holds |
| `main()` | Runs every probe, prints the table, asserts each property, exits 1 on any failure |

### 10.3 `orca_rl/pencil_train.py`

| Name | What it does |
|---|---|
| `PencilMetricsCallback(window=40)` | Collects the final `info` of each finished episode and logs the `task/` metrics every 40 episodes |
| `parse_args()` | Command-line flags (Section 9) |
| `main()` | Writes `env_kwargs.json` (with `"task": "pencil_spin"`), builds the vectorized environments, creates or resumes PPO, trains with the callbacks, saves the model and normalization statistics (also on Ctrl-C / SIGTERM) |

Reused from `train.py`: `build_vec_env(..., env_cls=PencilSpin)`, `EpochsUsedCallback`, `ClampLogStdCallback`,
`resolve_tensorboard_dir`. The only change to `train.py` is the new `env_cls` argument of `make_env_fn` and
`build_vec_env` (default `CubeReorientContinuous`, so the cube task is unaffected).

### 10.4 `orca_rl/pencil_eval.py`

| Name | What it does |
|---|---|
| `build_actor(args)` | `zero`, `random`, or a PPO model with its `VecNormalize` statistics and its saved environment settings |
| `make_camera(env)` | Fixed camera on the palm (azimuth 200°, elevation −55°, distance 0.34 m) |
| `main()` | Runs the episodes and prints revolutions, spin rate, drop rate, episode length, held fraction, return; optionally writes a video |

Flags: `--model`, `--policy {model,random,zero}`, `--episodes` (50), `--max-episode-steps` (1000), `--seed`
(1234; episode `i` uses `seed + i`), `--stochastic`, `--randomize-physics`, `--video OUT.mp4`, `--fps` (50 =
half speed), `--clean` (1280×720, no text).

## 11. Known issues and next steps

- **The policy clamps instead of spinning** (Section 1). Next: `pencil2` with `--pose-penalty 5e-4`, the only
  change from pencil1. The number to watch is the share of episodes still turning in the last 5 s (pencil1:
  about 10%). Any conclusion then needs a second seed, as with the cube task.
- **An idle pencil lying along the fingers slips between them.** At headings near 15° and 165° with the hand
  still, the pencil slides into a gap in 4–8 s and falls through on end; this is why "do nothing" drops 10% of
  episodes. It is real v1 geometry; the policy has to handle it (close or curl the fingers, or not stop there).
- **MuJoCo occasionally blows up.** Once in the 3M-step smoke run and twice in pencil1's 30M steps
  (`MUJOCO_LOG.TXT`: NaN/Inf in QACC on a pencil rotational degree of freedom). MuJoCo then resets the whole
  state to the XML pose; `PencilSpin.step` detects this from the simulation clock going backwards and
  truncates the episode, unscored. The first event in both runs was identical (simulation time 1.4260,
  DOF 21), so it reproduces with seed 0. If `task/unstable_rate` grows, raise the pencil's long-axis inertia
  (6.4e-8 kg·m²) or shorten the timestep.
- **Simulator-only observations.** The pencil's pose, velocities and per-finger contacts are not available on
  the hardware; before running on the real hand this needs an asymmetric actor-critic or state estimation,
  as for the cube task.
- **Round pencil.** The collision capsule rolls more easily than a real hexagonal pencil. Rolling earns
  nothing (5.3) but makes it easier to roll off the palm.
- **Not trained yet:** physics randomization; a second seed; the safety layer (`--safety` exists only for the
  cube task).
- **v1 only.**
