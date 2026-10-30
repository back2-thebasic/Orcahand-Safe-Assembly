# Orca_rl: Pencil Spinning with the ORCA v1 Hand (PPO)

[English](PENCIL.md) | [中文](PENCIL.zh-CN.md) | [한국어](PENCIL.ko.md)

The ORCA v1 right hand (the physical hand's version) lies palm-up with a pencil across its palm and fingers.
The task is to keep turning the pencil about the vertical axis — counter-clockwise seen from above —
without dropping it. There is no goal and no success bonus: the reward is the yaw the pencil actually gains,
so an episode's return is roughly the number of radians it was turned.

This document covers the scene, **the reward function in detail**, the observation and action, every
environment and training parameter, a reference for every function, and the results so far.

**Status (2026-10-30): one model spins continuously, now at twice the speed.** `pencil5_s1_fast` — pencil5_s1
trained 30M steps more with the reward's rate cap doubled to 2 rad/s — turns the pencil 2.98 revolutions per
10 s episode (113°/s; the cap allows 3.18), dropping it in 2% of 100 episodes and standing it up in 5%, and in
20 s episodes turns as much in the second 10 s as in the first. Every model before pencil5_s1 turns the pencil once and
then parks it; each fix closed one parking spot and the next run found another (Section 1.2). Trained with
four seeds each, all four pencil5 runs turn more than all four pencil4 runs, but only seed 1 learned to spin
continuously; the other three stand the pencil up or drop it more often than any pencil4 run (Section 1.3).
While spinning, the fingers touch each other 69% of the time (Section 1.4).

Demos, half speed:
[`demo_videos/7_pencil_v1.mp4`](demo_videos/7_pencil_v1.mp4) (pencil1, reset seeds 0–2, not selected);
[`demo_videos/8_pencil_compare.mp4`](demo_videos/8_pencil_compare.mp4) (no policy, pencil1, pencil2 and pencil3
side by side from the same two starts);
[`demo_videos/9_pencil3_30m_vs_60m.mp4`](demo_videos/9_pencil3_30m_vs_60m.mp4) (pencil3 at 30M and 60M steps,
20 s episodes);
[`demo_videos/10_pencil4_vs_pencil5.mp4`](demo_videos/10_pencil4_vs_pencil5.mp4) (pencil4 and pencil5, 20 s episodes);
[`demo_videos/11_pencil5_seed0_vs_seed1.mp4`](demo_videos/11_pencil5_seed0_vs_seed1.mp4) (pencil5 trained with
seed 0 and with seed 1, 20 s episodes, reset seeds not selected);
[`demo_videos/12_pencil5_s1_20s.mp4`](demo_videos/12_pencil5_s1_20s.mp4) (pencil5_s1, one 20 s episode, no text);
[`demo_videos/13_pencil5_s1_vs_fast.mp4`](demo_videos/13_pencil5_s1_vs_fast.mp4) (pencil5_s1 and pencil5_s1_fast,
20 s episodes, reset seeds not selected);
[`demo_videos/14_pencil4_vs_pencil5_seeds.mp4`](demo_videos/14_pencil4_vs_pencil5_seeds.mp4) (pencil4 and pencil5,
four training seeds each, from one start).

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

### 1.1 The models

Deterministic policy, 10 s episodes, evaluation seed 1234, always from an ordinary reset, MuJoCo 3.12.0.
All runs: 10 environments on an M4 Mac, seed 0 unless noted.

| Model | What changed | Revolutions / episode | Mean spin rate | Drop | Ended standing up | Return |
|---|---|---|---|---|---|---|
| Do nothing (30 episodes) | — | 0.00 | 0.1°/s | 10% | — | |
| `pencil_smoke`, 3M steps (50 episodes) | defaults, a short run | 0.13 | 4.7°/s | 4% | — | |
| `pencil1`, 30M (100 episodes) | defaults | 0.29 (max 0.55) | 11.9°/s | 14% | — | 1.27 ± 1.23 |
| `pencil2`, 30M (100 episodes) | + `--pose-penalty 5e-4` | 0.27 (max 0.49) | 9.7°/s | 2% | — | 0.55 ± 0.99 |
| `pencil3`, 30M (100 episodes) | + `--max-stand-deg 60 --stall-start-prob 0.5` | 0.22 (max 0.45) | 8.8°/s | 2% | 11% | −0.17 ± 0.95 |
| `pencil3_60m`, 60M (100 episodes) | pencil3 trained 30M steps more | 0.29 (max 0.58) | 11.2°/s | 1% | 11% | 0.26 ± 1.04 |
| `pencil4`, 30M (100 episodes) | + `--uncounted-penalty 0.5 --goal-step-deg 45` | 0.34 (max 0.58) | 12.6°/s | 2% | 5% | 1.94 ± 2.61 |
| `pencil5`, 30M (100 episodes) | + `--stall-penalty 0.5` | 0.39 (max 0.71) | 16.8°/s | 2% | 22% | 0.17 ± 2.40 |
| **`pencil5_s1`, 30M (100 episodes)** | pencil5 with `--seed 1` | **1.50** (max 1.95) | 55.5°/s | 0% | 5% | 15.48 ± 3.98 |
| `pencil4_s1`, 30M (100 episodes) | pencil4 with `--seed 1` | 0.18 (max 0.44) | 7.0°/s | 2% | 10% | 0.61 ± 1.81 |
| `pencil4_s2`, 30M (100 episodes) | pencil4 with `--seed 2` | 0.25 (max 0.59) | 10.4°/s | 2% | 18% | 1.06 ± 1.86 |
| `pencil4_s3`, 30M (100 episodes) | pencil4 with `--seed 3` | 0.28 (max 0.55) | 10.6°/s | 3% | 7% | 1.55 ± 1.68 |
| `pencil5_s2`, 30M (100 episodes) | pencil5 with `--seed 2` | 0.44 (max 0.76) | 19.1°/s | 5% | 33% | 1.22 ± 2.68 |
| `pencil5_s3`, 30M (100 episodes) | pencil5 with `--seed 3` | 0.52 (max 0.94) | 31.1°/s | 28% | 62% | 2.84 ± 2.25 |
| `pencil5_60m`, 60M (100 episodes) | pencil5 (seed 0) trained 30M steps more | 0.40 (max 0.75) | 16.4°/s | 4% | 16% | −0.25 ± 2.53 |
| **`pencil5_s1_fast`, 60M (100 episodes)** | pencil5_s1 trained 30M steps more with `--max-spin-rate 2` | **2.98** (max 3.46) | 112.8°/s | 2% | 5% | 36.75 ± 8.73 |
| Reference: scripted oracle at the 1 rad/s cap | | 1.59 | 57°/s | 0% | — | 9.98 (21.75 with pencil4's goals) |
| Reference: scripted oracle at the 2 rad/s cap | | 3.18 | 115°/s | 0% | — | |

"Revolutions" is the net yaw of the pencil's axis in the commanded direction, counted only while the pencil
is held and near level (Section 5.2). The mean spin rate divides by all episode time, dropped episodes
included. Returns are not comparable between rows: the runs charge different penalties. Checkpoint-by-checkpoint
numbers are in each run's results file.

Training time: pencil1 1 h 41 min (about 5,000 steps/s), pencil2 1 h 01 min (8,200), pencil3 1 h 08 min
(7,300), the 30M steps from pencil3 to pencil3_60m 54 min, pencil4 58 min (8,500), pencil5 59 min (8,500),
pencil5_s1 1 h 07 min (7,500); pencil4_s1–s3 and pencil5_s2–s3 56–62 min each (8,100–9,000), the 30M steps from
pencil5 to pencil5_60m 56 min, from pencil5_s1 to pencil5_s1_fast 53 min (9,400).

### 1.2 pencil1–pencil5 turn once, then park

Four 2.5 s windows, mean degrees turned, over 60 episodes (only those that neither dropped nor stood up):

| Model | Evaluation seed | 0–2.5 s | 2.5–5 s | 5–7.5 s | 7.5–10 s | Still turning (>10°) in the last 5 s |
|---|---|---|---|---|---|---|
| pencil1 | 1234 | 82° | 19° | 4° | 1° | 10% |
| pencil1 | 0 | 93° | 18° | 3° | 0° | 9% |
| pencil2 | 1234 | 80° | 19° | 0° | 1° | 5% |
| pencil2 | 0 | 74° | 16° | 2° | 2° | 5% |
| pencil3 | 1234 | 56° | 21° | 5° | 3° | 14% |
| pencil3 | 0 | 62° | 20° | 3° | 1° | 14% |
| pencil3_60m | 1234 | 60° | 36° | 16° | 2° | 48% |
| pencil3_60m | 0 | 56° | 38° | 13° | 2° | 46% |
| pencil4 | 1234 | 91° | 27° | 7° | 3° | 15% |
| pencil4 | 0 | 86° | 29° | 5° | 1° | 15% |
| pencil5 | 1234 | 98° | 47° | 10° | 3° | 36% |
| pencil5 | 0 | 102° | 45° | 10° | 2° | 22% |
| **pencil5_s1** | 1234 | 129° | 141° | 151° | 144° | 100% |
| **pencil5_s1** | 0 | 122° | 146° | 141° | 142° | 97% |

pencil3_60m's and pencil5's higher "still turning" shares are a longer, slower first push (pencil3_60m stops
at 4.5 s instead of 3.4 s at the median, pencil5 at 4.1 s); their last 2.5 s are as still as the others'.
pencil5_s1 turns at about the same rate in every window (Section 1.3).

**How each model parks** (at 10 s, evaluation seed 1234):

| Model | Parks by | Finger flexion | Servo targets at a limit | Pencil tilt (median) | Stop heading | Why it is free |
|---|---|---|---|---|---|---|
| pencil1 | closing a fist around the pencil | 1.11 rad | 75% | 13° | 79°, R 0.63 | A clamped pencil earns nothing but cannot fall |
| pencil2 | standing the pencil up in a gap between the fingers | −0.26 rad | 47% | 67° | 18°, R 0.83 | Past 45° of tilt yaw is not counted (5.2) |
| pencil3 | leaving it flat and still on the open hand | −0.06 rad | 13% | 3° | 52°, R 0.95 | Holding still costs nothing |
| pencil3_60m | sinking it into a finger gap, below the 0.13 m held line | −0.03 rad | 20% | 33° | 68°, R 0.93 | Not held, so not counted (5.2) |
| pencil4 | leaving it flat and still across the palm | 0.06 rad | 40% | 2° | 86°, R 0.80 | Holding still costs nothing |
| pencil5 | leaving it flat and still, diagonally on the palm | −0.02 rad | 34% | 5° | 135°, R 0.94 | It is not free (0.5/s, 5.6): the policy pays |
| pencil5_s1 | does not park | 0.00 rad | 24% | 8° | spread out, R 0.13 | — |

Finger flexion is the mean of the 8 finger mcp/pip joints; the reset pose is 0. The stop heading is where the
pencil's last 10° of turning ended, mod 180° (0° lies along the fingers, 90° across the palm), with its
concentration R (1 = always the same heading, 0 = uniform). Starting headings are uniform; for pencil1–pencil5
the stop headings are not: each learned "push the pencil from wherever it starts to one heading, then stop",
and how far it turns depends on how far the start was from that heading (correlation +0.27 to +0.49). The
stop headings differ, and each model's push carries the pencil through the others' stop headings, so no
heading is a geometric dead end. What none of them does is **push a second time**: lift or slide the fingers
back, re-contact, push again. Up to pencil4 that risked a drop while parking cost nothing; pencil5 pays
0.5/s to park and still parks. pencil5_s1's stop headings are just where the pencil happened to be when the
episode ended.

**What happens after 10 s** (40 episodes of 20 s each; turning columns only over episodes that lasted 20 s):

| Model | 0–10 s | 10–20 s | Turning >10° in 10–20 s | Ended early within 20 s |
|---|---|---|---|---|
| pencil1 | 115° | −4° | 0% | 13 of 40 |
| pencil2 | 100° | 2° | 3% | 3 of 40 |
| pencil3 | 86° | 7° | 17% | 5 of 40 (stood up, median 4.0 s) |
| pencil3_60m | 149° | −2° | 0% | 25 of 40 (24 stood up, median 15.0 s) |
| pencil4 | 127° | +2° | 6% | 6 of 40 (4 stood up, 2 dropped) |
| pencil5 | 152° | +0° | 16% | 15 of 40 (13 stood up, median 4.8 s; 2 dropped) |
| **pencil5_s1** | **562°** | **+575°** | **100%** | 2 of 40 (1 stood up at 10.8 s, 1 dropped at 19.7 s) |

pencil3_60m's sunk pencil slowly tips upright and passes 60° mostly after the 10 s an episode lasts. With
γ = 0.995 a −2 five seconds away is worth about 8% of −2 today, so the policy hardly sees that cost.

**Why restarting from stalls (`--stall-start-prob`) did little.** The observation has no clock, so a restart
from a saved state is physically the same as letting the original episode run on (`pencil_checks` verifies
the trajectories are bit-identical). pencil1 and pencil2 already spent about 75% of every episode parked: the
policy never lacked samples of parked states. Restarts raised that share to about 87% and removed some of the
easy "first push" experience. What is missing is exploration that finds a second push from those states.

Full records (in Chinese): `results/2026-10-14-pencil-spin-smoke.md`, `results/2026-10-14-pencil1.md`,
`results/2026-10-26-pencil2.md`, `results/2026-10-27-pencil3.md` (pencil3 and pencil3_60m),
`results/2026-10-28-pencil4.md`, `results/2026-10-29-pencil5.md` (pencil5 and pencil5_s1),
`results/2026-10-30-overnight.md` (four seeds each, pencil5_60m, pencil5_s1_fast).

### 1.3 pencil5_s1: continuous spinning, and how sure we are

pencil5_s1 is pencil5 — the same flags, the same code — trained with `--seed 1` instead of 0.

| 100 episodes of 10 s, evaluation seed 1234 | pencil4 | pencil5 (seed 0) | pencil5_s1 (seed 1) |
|---|---|---|---|
| Revolutions / episode | 0.34 | 0.39 | **1.50** |
| Held but stalled (under 10° of new ground in 1 s) | 58% | 42% | 2% |
| Episodes that never stall | 3% | 10% | 70% |
| Ended early (dropped or stood up) | 7% | 24% | 5% |

With evaluation seed 0 pencil5_s1 gives the same: 1.51 revolutions, no drop, 3% stood up.

**Training.** For 13M steps pencil5_s1 was only a little ahead of seed 0 (0.39 against 0.30 revolutions at
10M; training data, with exploration noise). From about 14M it accelerated — 0.91 at 20M, 1.21 at 25M, 1.40 at
30M — while seed 0 flattened at about 0.4. By the end, episodes resumed from a stall turned about as much as
fresh ones (1.23 against 1.40): no fixed parking spot was left. The stand-up rate was as high as seed 0's
early in training (20–40%) and fell to 8%.

**It is real turning.** Over 40 episodes the yaw was recomputed from the raw simulation state, outside the
reward code. The reward's counted yaw, the axis heading unwrapped step by step, and the heading of the pencil
as an unoriented line (which a flip end over end would leave unchanged) all give 565° per episode. The largest
change in a single step is 5.9°. Tilt: median 9°, 95th percentile 24°, never past 45° — the pencil is turned
lying down, not stood up.

**The motion.** The pencil lies diagonally across the four fingers, which curl and straighten in turn and walk
it round; the thumb stays mostly out of the way (in contact 12% of the time; ring 50%, index 31%, middle 29%,
pinky 23%). The finger mcp joints move at about 0.3 Hz, one cycle per half turn of the pencil
(180° at 56°/s ≈ 3.2 s): the pencil is the same end to end, so the motion repeats every half turn. Now and then
the pencil leaves the hand for a step or two (held 95% of the time).

**The fingers touch each other 49% of the time** (pencil5 seed 0: 93%; an idle hand: 0%). The reward does
not charge for it, but the team's safety layer exists to prevent exactly this, so it has to be dealt with
before the real hand.

**Stall penalty or seed: both.** pencil4 and pencil5 were then trained with seeds 1–3 as well (100 episodes of
10 s each, evaluation seed 1234):

| Training seed | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| pencil4: revolutions | 0.34 | 0.18 | 0.25 | 0.28 |
| pencil5: revolutions | 0.39 | **1.50** | 0.44 | 0.52 |
| pencil4: held but stalled | 58% | 65% | 56% | 65% |
| pencil5: held but stalled | 42% | 2% | 33% | 14% |
| pencil4: ended early (dropped + stood up) | 7% | 12% | 20% | 10% |
| pencil5: ended early (dropped + stood up) | 24% | 5% | 38% | 90% |

- The stall penalty helps, every time: the worst pencil5 run (0.39) turns more than the best pencil4 run
  (0.34), and every pencil5 run is stalled less than every pencil4 run. If the penalty did nothing, the chance
  that the four pencil5 runs would all rank above the four pencil4 runs is 1/70 ≈ 0.014 (exact one-sided
  Mann–Whitney). Four seeds per setting is still few, and 0.39 against 0.34 is close.
- It is not enough: only seed 1 learned to spin continuously. Seed 2 turns about 150° in the first 5 s and then
  stalls or stands the pencil up; seed 3 really spins (24°/s at the median), but 35 of 40 episodes end early.
- Apart from seed 1, pencil5 stands the pencil up more often than any pencil4 run (22%, 33%, 62% against at
  most 18%). A likely reason, not tested yet: standing up ends the episode with the drop penalty of 2. Under
  pencil4, parking for good is worth 0, so failing costs 2 more than parking; under pencil5, parking for good
  costs 0.5 · dt / (1 − γ) = 1.00, so failing costs only 1 more. Seeds that did not learn to spin are more
  willing to gamble on a move that may stand the pencil up. Training pencil5 with `--drop-penalty 3` would
  test this.
- Seed 0 does not take off with more steps: trained to 60M (`pencil5_60m`) it turns 0.40 revolutions and is
  stalled 42% of the time, as at 30M. Seed 1 took off at 14M; seed 0 is stuck elsewhere.

Every training curve, and how seeds 2 and 3 move: `results/2026-10-30-overnight.md`.

### 1.4 pencil5_s1_fast: twice the rate cap

pencil5_s1 trained 30M steps more with `--max-spin-rate 2` (the rate the reward pays up to, in rad/s);
everything else unchanged.

| 100 episodes of 10 s, evaluation seed 1234 | pencil5_s1 (cap 1 rad/s) | pencil5_s1_fast (cap 2 rad/s) |
|---|---|---|
| Revolutions / episode | 1.50 (oracle at the cap: 1.59) | **2.98** (oracle at the cap: 3.18) |
| Mean spin rate | 55.5°/s | 112.8°/s |
| Dropped / stood up | 0% / 5% | 2% / 5% |
| Held | 95% | 91% |
| Held but stalled | 2% | 0% |
| Two fingers touching each other | 50% | **69%** |

The rate doubled, and the training curve was still rising at the end (2.63 → 2.80 revolutions over the last
10M steps, with exploration noise): the speed is set by the reward's cap, not by the hand.

Checked from the raw simulation state as in Section 1.3, over 40 episodes of 20 s: 35 ran the full 20 s,
1 dropped, 4 stood up. The counted yaw (2038° per episode), the axis heading (2040°) and the unoriented line
(2040°) agree, so this is not flips end over end. In the episodes that ran the full 20 s the pencil turned 1121°
in the first 10 s and 1167° in the second, and every one was still turning at the end. Tilt: median 8.5°,
95th percentile 22°, never past 45°. The motion is pencil5_s1's, faster: the four fingers walk the pencil round
(in contact: ring 43%, middle 33%, index 21%, pinky 18%, thumb 6%).

**Finger-finger contact grows with speed: 69% of the time**, against 50% for pencil5_s1 measured the same way
(Section 1.3's 49% was over other episodes). It is now the main problem before the real hand (Section 11).

## 2. Quickstart

Python 3.11, from the `Orca_rl/` directory, in the **training** environment
(`env/requirements-train.lock.txt`, see `env/README.md`). No change to orca_sim is needed: the pencil scene
ships with `orca_rl` (Section 10.1).

```bash
# 1. Check the reward (about 1 minute). Must end with "all checks passed".
python -m orca_rl.pencil_checks

# 2. Evaluate a trained model. 100 episodes reproduce Section 1.1, e.g.
#    pencil1: 0.29 rev / 14% drop / return 1.27 +- 1.23
#    pencil5_s1: 1.50 rev / 0% drop / 5% stood up / return 15.48 +- 3.98
#    pencil5_s1_fast: 2.98 rev / 2% drop / 5% stood up / return 36.75 +- 8.73
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 100
python -m orca_rl.pencil_eval --model models/pencil5_s1/final_model.zip --episodes 100
python -m orca_rl.pencil_eval --model models/pencil5_s1_fast/final_model.zip --episodes 100

# 3. Baselines
python -m orca_rl.pencil_eval --policy zero --episodes 30
python -m orca_rl.pencil_eval --policy random --episodes 10

# 4. Video: with a text overlay (640x480), or demo footage without text (1280x720)
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --video pencil1.mp4
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --seed 0 \
    --video 7_pencil_v1.mp4 --clean
python -m orca_rl.pencil_eval --model models/pencil5_s1/final_model.zip --episodes 1 \
    --max-episode-steps 2000 --video 12_pencil5_s1_20s.mp4 --clean

# 5. Train, about 1-2 h each on an M4 Mac. The runs in Section 1.1:
python -m orca_rl.pencil_train --name pencil1 --timesteps 30_000_000 --n-envs 10
python -m orca_rl.pencil_train --name pencil2 --timesteps 30_000_000 --n-envs 10 --pose-penalty 5e-4
python -m orca_rl.pencil_train --name pencil3 --timesteps 30_000_000 --n-envs 10 --pose-penalty 5e-4 \
    --max-stand-deg 60 --stall-start-prob 0.5
python -m orca_rl.pencil_train --name pencil3_60m --resume runs/pencil3/final_model.zip \
    --timesteps 30_000_000 --n-envs 10 --pose-penalty 5e-4 --max-stand-deg 60 --stall-start-prob 0.5
python -m orca_rl.pencil_train --name pencil4 --timesteps 30_000_000 --n-envs 10 --pose-penalty 5e-4 \
    --max-stand-deg 60 --stall-start-prob 0.5 --uncounted-penalty 0.5 --goal-step-deg 45 --goal-bonus 1
python -m orca_rl.pencil_train --name pencil5 --timesteps 30_000_000 --n-envs 10 --pose-penalty 5e-4 \
    --max-stand-deg 60 --stall-start-prob 0.5 --uncounted-penalty 0.5 --goal-step-deg 45 --goal-bonus 1 \
    --stall-penalty 0.5
# pencil5_s1: the pencil5 command with --name pencil5_s1 --seed 1
# pencil4_s1–s3, pencil5_s2–s3: the pencil4 / pencil5 command with --name <name> --seed 1, 2 or 3
# pencil5_60m: the pencil5 command with --name pencil5_60m --resume runs/pencil5/final_model.zip
python -m orca_rl.pencil_train --name pencil5_s1_fast --resume runs/pencil5_s1/final_model.zip --seed 1 \
    --timesteps 30_000_000 --n-envs 10 --max-spin-rate 2.0 --pose-penalty 5e-4 --max-stand-deg 60 \
    --stall-start-prob 0.5 --uncounted-penalty 0.5 --goal-step-deg 45 --goal-bonus 1 --stall-penalty 0.5

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
├── models/                # each: final_model.zip, vecnormalize.pkl, env_kwargs.json
│   ├── pencil1/  pencil2/  pencil3/  pencil3_60m/  pencil4/  pencil5/  pencil5_s1/
│   ├── pencil4_s1/  pencil4_s2/  pencil4_s3/  pencil5_s2/  pencil5_s3/  pencil5_60m/  pencil5_s1_fast/
├── results/               # 2026-10-14-pencil-spin-smoke.md, 2026-10-14-pencil1.md,
│                          # 2026-10-26-pencil2.md, 2026-10-27-pencil3.md,
│                          # 2026-10-28-pencil4.md, 2026-10-29-pencil5.md, 2026-10-30-overnight.md
└── demo_videos/           # 7_pencil_v1.mp4, 8_pencil_compare.mp4, 9_pencil3_30m_vs_60m.mp4,
                           # 10_pencil4_vs_pencil5.mp4, 11_pencil5_seed0_vs_seed1.mp4, 12_pencil5_s1_20s.mp4,
                           # 13_pencil5_s1_vs_fast.mp4, 14_pencil4_vs_pencil5_seeds.mp4
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

**Restarting from a stall** (`stall_start_prob`, default 0; pencil3: 0.5). With this probability, and only
once there is something saved, a reset skips steps 2–4 and resumes a state where an earlier episode stalled:

- *Stalled* means the pencil is held but its counted yaw grew less than `stall_turn_deg` (10°) over the last
  `stall_window_steps` (100 steps, 1 s). Only the **first** stall of an episode that itself started fresh is
  saved, so the buffer samples where the policy gets stuck from an ordinary start, not where it drifts after
  many restarts.
- Each environment keeps its own ring buffer of the last `stall_buffer_size` (64) stalls. A state is the full
  MuJoCo integration state (`mj_getState` with `mjSTATE_INTEGRATION`: time, qpos, qvel, actuator state,
  controls, warm start), so a resumed episode steps bit-identically to the original.
- The resumed episode counts yaw from zero and runs a full 1000 steps. The pose penalty keeps the original
  episode's `q_rest`; otherwise a stalled pose would become free.
- The draw is skipped while the buffer is empty, so with the option off a seeded reset is exactly what it
  always was. The buffer is not saved with the model. `pencil_eval.py` always resets fresh.

### 4.4 Episode end

| Condition | Result |
|---|---|
| Pencil centre below 0.12 m (`drop_height`) | `terminated`, reward −2 on that step |
| Pencil tilted more than `max_stand_deg` from level (off by default; pencil3: 60°) | `terminated`, reward −2 on that step, `info["stood_up"] = True` |
| 1000 steps (10 s) | `truncated` |
| MuJoCo state became NaN/Inf | `truncated`, reward 0, `info["sim_unstable"] = True` (Section 11) |

## 5. Reward function

### 5.1 Formula

Per control step (10 ms):

```
r_t =   spin_coef           * paid_t                                   # yaw gained, rate-capped (5.3-5.5)
      + goal_bonus          * [a goal heading reached on this step]    # optional, off by default (5.6)
      - action_rate_penalty * sum_j ((target_t,j - target_t-1,j) / halfspan_j)^2
      - pose_penalty        * sum_j (q_t,j - q_rest,j)^2                # optional, off by default
      - uncounted_penalty   * dt * [pencil not counted on this step]    # optional, off by default (5.6)
      - stall_penalty       * dt * [counted, but stalled]               # optional, off by default (5.6)
      - drop_penalty        * [pencil dropped, or tilted past max_stand_deg, on this step]
```

| Term | Default | Largest value per step | Per 10 s episode |
|---|---|---|---|
| `spin_coef * paid` | 1.0 | +0.01 (1 rad/s × 10 ms) | at most +10 |
| Action rate | 0.002 | −0.00077 (all 17 joints moving the full 0.15 halfspan) | — |
| Pose | 0 (pencil2, pencil3: 5e-4) | pencil1's final fist (≈9 rad²) at 5e-4: −0.0045 | — |
| Drop or stand-up | 2.0 | −2, once; the episode ends | — |
| Goal bonus | off (pencil4, pencil5: 1 every 45°) | +1, on the step a goal is reached | +12 at the cap |
| Not counted | 0 (pencil4, pencil5: 0.5) | −0.005 | −5 if never counted |
| Stalled | 0 (pencil5: 0.5) | −0.005 | about −4.5 for an idle hand (the first second is never a stall) |

`j` runs over the 17 actuated joints; `halfspan_j` is half of joint `j`'s control range; `q` are measured
joint angles. The return of a perfect 10 s episode is therefore about 10 (the oracle in Section 5.8 scores
9.98), or 21.75 with goal headings every 45°.

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

### 5.6 Penalties and the goal bonus

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

  In pencil2 it did its job: the fingers ended open (flexion −0.26 rad instead of 1.11) and drops fell from
  14% to 2%. The policy parked another way instead.
- **Standing up** (`max_stand_deg`, default off; pencil3: 60°): a pencil tilted further than this from level
  ends the episode and pays `drop_penalty`, like a drop. pencil2 stood the pencil up in a finger gap (median
  tilt 67°), where yaw is not counted and nothing is charged. Between `max_tilt_deg` (45°) and
  `max_stand_deg` the pencil is still neither paid nor charged and the episode goes on. In pencil3 no episode
  ended its 10 s tilted past 45°.
- **Not counted** (`uncounted_penalty`, default 0; pencil4, pencil5: 0.5): charged per second while the pencil
  is not counted (not held, or tilted past 45°). pencil2 parked the pencil standing in a finger gap and
  pencil3_60m sank it below the held line, both unscored and free; this closes every such spot at once. With
  γ = 0.995 a charge of 0.5/s that never ends is worth 0.5 · 0.01 / 0.005 = 1.0 discounted, half the drop
  penalty, so dropping on purpose does not pay.
- **Stalled** (`stall_penalty`, default 0; pencil5: 0.5): the same charge while the pencil is counted but has
  gained less than `stall_turn_deg` (10°) of **new ground** over the last `stall_window_steps` (1 s); never on
  top of the not-counted charge. New ground is measured from the furthest the pencil has been turned so far,
  so working it back and forth over one arc is a stall (the 0.2 rad ratchet is charged 90% of the time);
  measuring the net turn over the window instead would leave a forward-10°, back-10° cycle half free. The
  first second of an episode is never a stall. pencil4 parked flat and still on the palm, the one free state
  it had left; with both charges every state but turning forward costs the same 0.5/s, and parking still
  costs at most 1.0 discounted, under the drop penalty. The price: an idle hand now scores about −4.5 instead
  of 0, so that check changed (5.8).
- **Goal headings** (`goal_step_deg`, default off; pencil4, pencil5: 45° with `goal_bonus` 1): a goal heading
  `goal_step_deg` ahead of the pencil is added to the observation (3 values, Section 6); reaching it pays
  `goal_bonus` and sets the next one as far again. Progress toward a goal is accumulated from *paid* yaw, so
  goals inherit the rate cap and the carry: flicking reaches them no sooner than turning at the cap, and at
  every step the bonuses so far cover no more turn than the pencil has made (asserted in 5.8). Turning back
  moves the goal away again.

### 5.7 What the reward deliberately does not have

| Not included | Why |
|---|---|
| Success bonus (by default) | There is no discrete goal; any bonus for "a revolution" would be a step function of the same yaw the reward already pays continuously. The optional goal headings (5.6) are such a step function, built on paid yaw so they cannot pay for more turn than was made |
| Alive / holding bonus | Pays the policy for doing nothing — the stock cube task's stalling exploit (`orca_rl/README.md`) |
| Unconditional per-step cost | Makes ending the episode early (dropping on purpose) attractive. The optional not-counted and stall charges apply only while the pencil makes no progress, and their discounted total stays under the drop penalty (5.6) |
| Reward on `ω_z` | Pays for rolling a tilted pencil about its own axis (Section 5.3) |
| Pencil position term | No sign yet that the pencil wanders; a drop already ends the episode |

### 5.8 Scripted checks (`python -m orca_rl.pencil_checks`)

Most probes move the pencil directly; they are not policies a hand could execute, they are tests of what
the reward pays for. 10 episodes of 10 s each:

```
policy (10 episodes of 10 s)             return    revs   drop
--------------------------------------------------------------
do nothing                                -0.20    0.00    10%
random                                    -2.31   -0.02   100%
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

with --max-stand-deg 60:
  do nothing                              -0.20    0.00     0%  stood up  10%
  oracle: ccw at the cap, 1 rad/s          9.98    1.59     0%  stood up   0%
  stand it 50 deg up and turn              0.00    0.00     0%  stood up   0%
  stand it 70 deg up and hold             -2.00    0.00     0%  stood up 100%

with --stall-start-prob 0.5 (and --max-stand-deg 60):
  do nothing                              -0.20   -0.00     0%  resumed  43%
  oracle: ccw at the cap, 1 rad/s          9.98    1.59     0%  resumed   0%

with --uncounted-penalty 0.5 --goal-step-deg 45 --goal-bonus 1 (and --max-stand-deg 60):
  do nothing                              -0.22    0.00     0%  goals   0.0
  oracle: ccw at the cap, 1 rad/s         21.75    1.59     0%  goals  12.0
  oracle: ccw at 5x the cap               21.74    7.96     0%  goals  12.0
  oracle: wrong way (cw), 1 rad/s        -10.23   -1.59     0%  goals   0.0
  ratchet: creep fwd, snap back            0.05    0.01     0%  goals   0.0
  0.5 rad/s with a +-2 deg shiver         10.59    0.78     0%  goals   5.9
  stand it 50 deg up and turn             -5.00    0.00     0%  goals   0.0
  toss: fwd in hand, back in air          -6.32   -0.77     0%  goals   0.0
  sink it below the held line             -5.00    0.00     0%  goals   0.0

with --stall-penalty 0.5 (and pencil4's options):
                                         return    revs   drop  goals  stalled   disc.
  do nothing                              -4.48    0.00     0%    0.0      88%   -0.60
  drop it at once                         -2.00    0.00   100%    0.0       0%   -2.00
  oracle: ccw at the cap, 1 rad/s         21.75    1.59     0%   12.0       0%    3.94
  oracle: ccw at 5x the cap               21.74    7.96     0%   12.0       0%    3.98
  oracle: wrong way (cw), 1 rad/s        -14.54   -1.59     0%    0.0      86%   -2.59
  ccw at 0.25 rad/s (14 deg/s)             4.89    0.38     0%    2.8       0%    0.74
  ccw at 0.1 rad/s (6 deg/s)              -2.50    0.16     0%    1.0      90%   -0.38
  push 2 s at the cap, then park           0.11    0.31     0%    1.9      57%    2.03
  ratchet: creep fwd, snap back           -4.43    0.01     0%    0.0      90%   -0.49
  0.5 rad/s with a +-2 deg shiver         10.59    0.78     0%    5.9       0%    1.72
  stand it 50 deg up and turn             -5.00    0.00     0%    0.0       0%   -0.99
  toss: fwd in hand, back in air          -9.33   -0.77     0%    0.0      60%   -1.20
  sink it below the held line             -5.00    0.00     0%    0.0       0%   -0.99
```

Assertions: doing nothing earns ≈ 0; turning at the cap for 10 s earns ≈ 10; 5× faster earns no more; the
wrong direction is charged in full; ratcheting earns nothing; shivering costs nothing; rolling in place and
turning a standing pencil earn nothing; yaw lost in the air is billed on landing; with the pose penalty,
idling and the oracle are unchanged and a fist costs more than 2 per episode. With `--max-stand-deg 60`:
idling costs the same (its slips into a finger gap now end as a stand-up instead of a drop), the oracle still
earns ≈ 10, 50° up earns and costs nothing, 70° up ends the episode on step one at −2. With
`--stall-start-prob 0.5` (30 seeds): an idle hand stalls and about half the resets resume a stall, idling
from a stall still earns ≈ 0, a policy that never stalls is never resumed, and a resumed state replays the
original episode bit for bit with its yaw counted from zero. With `--uncounted-penalty 0.5 --goal-step-deg 45`
(pencil4): idling costs about the same; the oracle earns its spin plus a bonus every 45° (21.75, 12 goals); 5×
faster reaches no more goals; the wrong direction, ratcheting and tossing reach none; shivering still costs
nothing; standing 50° up and sinking the pencil below the held line now cost 0.5/s; at no step do the bonuses
cover more turn than the pencil has made. With `--stall-penalty 0.5` on top (pencil5) "doing nothing earns ≈ 0"
no longer holds, and is replaced by what the policy actually weighs: idling is charged 0.5/s for exactly the
time it sits stalled (−4.48; the first second is never a stall), yet parking still beats dropping when
discounted with γ = 0.995 (−0.60 against −2.00), and parked forever costs 1.00 < 2. Turning at the cap is never
stalled and earns what it did; 14°/s is never stalled and 6°/s nearly always (the line is 10°/s); pushing 2 s
at the cap and parking — pencil4's end state — earns 2.87 less than under pencil4, every second from 1 s after
it stops being billed as stalled or not counted; a ratchet makes no new ground after its first arc and is
charged as a stall; standing up and sinking pay the not-counted charge once, not a stall on top. The random row
varies a little between runs (its actions are not seeded). "Do nothing" drops 10% of the time because of the
slow slip described in Section 11.

## 6. Observation

90 values, or 93 with goal headings (`goal_step_deg`), float64, normalized during training by `VecNormalize`
(running mean/std, clipped to ±10).

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
| 90–92 | 3 | Only with goal headings: the goal heading as cos, sin; the share of the goal step still to go | world |

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
| `pose_penalty` | 0.0 | Per rad² from the settled pose (5.6); pencil2 and pencil3 use 5e-4 |
| `action_rate_penalty` | 0.002 | Smoothness term (5.6) |
| `max_tilt_deg` | 45.0 | Yaw counts only within this of level (5.2) |
| `max_stand_deg` | None | Tilted past this, the episode ends as a drop (4.4, 5.6); must be ≥ `max_tilt_deg`; pencil3: 60 |
| `uncounted_penalty` | 0.0 | Per second the pencil is not counted (5.6); pencil4, pencil5: 0.5 |
| `stall_penalty` | 0.0 | Per second the pencil is counted but stalled (5.6); pencil5: 0.5 |
| `goal_step_deg` | None | Goal headings this far apart, in the observation (5.6, 6); pencil4, pencil5: 45 |
| `goal_bonus` | 1.0 | Paid on reaching a goal heading |
| `in_hand_height` | 0.13 | m; held requires the pencil centre above this (5.2) |
| `drop_height` | 0.12 | m; below this the episode ends as a drop |
| `action_mode` | `"relative"` | `"relative"` or `"absolute"` (7) |
| `action_scale` | 0.15 | Fraction of the halfspan a relative action can move per step |
| `randomize_physics` | False | Per reset: pencil mass ±30%, pencil friction ±30%, joint damping ±30%, actuator gain ±20%, joint-angle noise σ 0.005 rad |
| `randomization` | None | Dict overriding those ranges (keys `cube_mass`, `cube_friction`, `joint_damping`, `actuator_gain`, `obs_noise_rad`; names shared with the cube task) |
| `reset_settle_steps` | 100 | Control steps the pencil settles before the episode starts (4.3) |
| `initial_yaw` | `"random"` | `"random"` (uniform 0–360°) or `"nominal"` (0°) |
| `pencil_pos_xy_jitter` | 0.005 | m, uniform xy offset of the spawn point |
| `stall_start_prob` | 0.0 | Share of resets that resume a saved stall (4.3); pencil3: 0.5 |
| `stall_buffer_size` | 64 | Stalls kept per environment |
| `stall_window_steps` | 100 | Window over which a stall is judged, for restarts and for the stall penalty |
| `stall_turn_deg` | 10.0 | Less counted yaw than this over the window is a stall; for the stall penalty, less new ground |

Fixed in code: up to 10 reset attempts, 15° maximum settled tilt (4.3); `PENCIL_HAND_POSE` (4.2).

`reset(seed=None, options=None)` accepts `options={"cube_quat": q, "cube_pos": p}` to pin the pencil's spawn
(no rejection sampling then), plus the parent class's options (`hand_pose_by_joint`, `hand_qpos`, …).

`info` returned by `reset` and `step`:

| Key | Meaning |
|---|---|
| `yaw_deg`, `tilt_deg` | Heading of the pencil axis, and its angle from horizontal |
| `pencil_pos`, `pencil_height` | Pencil centre (m) |
| `in_hand`, `dropped` | Held (5.2); below `drop_height` |
| `stood_up` | Tilted past `max_stand_deg` (always False when it is off) |
| `stall_start` | This episode resumed a saved stall (4.3) |
| `elapsed_steps` | Steps so far in the episode |
| `turned_rad`, `turned_revs` | Net counted yaw so far in the commanded direction |
| `paid_rad` | The part of it the reward has paid |
| `in_hand_frac` | Fraction of steps held so far |
| `stalled_frac` | Fraction of steps so far counted but stalled (under 10° of new ground in 1 s), whether or not the stall penalty is on |
| `goals_reached` | Goal headings reached so far (0 without goals) |
| `stalled` | Whether this step was a stall; `step` only |
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
`--pose-penalty`, `--max-tilt-deg`, `--max-stand-deg`, `--stall-start-prob`, `--uncounted-penalty`,
`--stall-penalty`, `--goal-step-deg`, `--goal-bonus`, `--action-mode`, `--action-scale`, `--randomize-physics`
(same meaning as Section 8). `--resume MODEL.zip` continues training; hyperparameters
and task flags are re-read from the command line, so pass the same task flags again.

**Network:** `MlpPolicy`, separate actor and critic, each 90 (93 with goals) → 256 → 256 with tanh; a linear
head to 17 action means (actor) or 1 value (critic); a state-independent learned log-std per action.
182,819 parameters (184,355 with goals).
**Normalization:** `VecNormalize` of observations and rewards (reward normalization uses the same γ).

**TensorBoard** (written every 40 finished episodes; it is training data with exploration noise —
use `pencil_eval.py` for numbers). The `task/` metrics count only episodes that started fresh, so they compare
across runs with and without `--stall-start-prob`; episodes resumed from a stall get the same metrics under
`task_resumed/`.

| Metric | Meaning |
|---|---|
| `task/revs_per_episode` | **Progress metric.** Mean counted revolutions per episode (oracle: 1.59) |
| `task/spin_rate_deg_s` | Counted yaw per second of episode, drops included |
| `task/paid_frac` | Reward-paid yaw / counted yaw. Near 1: steady turning. Well below 1: over the cap or shivering |
| `task/drop_rate` | Fraction of episodes ending in a drop (pencil below 0.12 m) |
| `task/stand_rate` | Fraction ending because the pencil stood up past `max_stand_deg` |
| `task/fail_rate` | Drops and stand-ups together |
| `task/episode_length` | Mean steps per episode |
| `task/in_hand_frac` | Fraction of steps the pencil is held |
| `task/stalled_frac` | Fraction of steps the pencil is counted but stalled, whether or not `--stall-penalty` is on |
| `task/goals_per_episode` | Goal headings reached per episode (with `--goal-step-deg`) |
| `task/unstable_rate` | Fraction of episodes cut off by a MuJoCo blow-up; should stay near 0 |
| `task/stall_start_frac` | Fraction of finished episodes that had resumed a stall |
| `task_resumed/*` | The metrics above over resumed episodes only |
| `train/epochs_used` | PPO epochs actually run before `target_kl` stopped them |
| `train/std` | Action std, starting at 0.61; pencil1 ended at 0.30, pencil3 at 0.31, pencil3_60m at 0.27, pencil4 at 0.33, pencil5 at 0.34, pencil5_s1 at 0.32 |

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
| `_stood_up()` | `max_stand_deg` is set and the tilt is past it |
| `_angvel_world()` | Rotates the free joint's body-frame angular velocity into the world frame |
| `_contacts()` | One pass over MuJoCo contacts: (pencil touches palm or a finger, 5 per-finger flags) |
| `_in_hand()` | Pencil centre above `in_hand_height` and touching palm or finger |
| `_resolve_initial_cube_quat(options)` | Spawn orientation: `options["cube_quat"]`, else a random (or zero) yaw about world z |
| `_get_obs()` | The observation: 90 values, 93 with goal headings (Section 6) |
| `_goal_features()` | The goal heading (cos, sin) and the share of the goal step still to go; empty without goals |
| `_get_info()` | The `info` dict (Section 8) |
| `_reset_tracking()` | Zeroes the episode's yaw bookkeeping: `psi_ref`, `k`, carry, counted and paid yaw, held steps, goal progress, the stall windows (net turn for restarts, furthest turn for the penalty) |
| `_save_stall()` | Saves the MuJoCo integration state and this episode's `q_rest` into the ring buffer (4.3) |
| `_resume_stall()` | Restores a random saved stall: `mj_resetData`, `mj_setState`, `mj_forward`, step counter 0, its `q_rest` |
| `reset(seed, options)` | Randomize → with probability `stall_start_prob` (and a non-empty buffer, no `options`) `_resume_stall()`, otherwise `_fresh_reset()` → continue from the held servo targets → `_reset_tracking()` (4.3) |
| `_fresh_reset(seed, options)` | Place → settle → reject and redraw if needed → store `q_rest` (4.3) |
| `step(action)` | Map action → step MuJoCo 5× → detect a blow-up → held? dropped? → counted yaw with cap and carry → stalled? → reward (5.1): spin, not-counted or stall charge, goals, pose, drop → stood up? → terminated / truncated → remember the first stall of a fresh episode → obs, reward, flags, info |

Borrowed unchanged from `CubeReorientContinuous` (`task.py`): `_setup_fingers` (finds each finger's links and
its fingertip collision geom from the kinematic tree), `_apply_randomization` (per-reset physics
randomization), `_settle_cube` (holds the servo targets for `reset_settle_steps`), `_target_from_action`
(relative/absolute action mapping and clipping). Inherited from orca_sim: `_cube_pos`, `_cube_qvel`,
`_cube_dropped`, `_compose_ctrl_from_qpos`, `_quat_from_axis_angle`, `render`, `close`.

### 10.2 `orca_rl/pencil_checks.py`

| Name | What it does |
|---|---|
| `run(policy, seeds, steps, **env_kwargs)` | Runs a scripted policy for 10 seeds (by default); returns mean return, its discounted value (γ = `GAMMA` = 0.995, what the policy weighs), revolutions, drop and stand-up rates, goals, the share and seconds stalled, seconds billed as stalled or not counted, share of resumed episodes, how often the bonuses ran ahead of the turn, and what a two-sided cap and a no-carry cap would have paid |
| `place(env, pos, quat, angvel_body)` | Poses the pencil directly (then `mj_forward`), returns a zero action |
| `yaw_quat(yaw, tilt)` | Quaternion of a pencil at that heading, raised `tilt` from level |
| `zero`, `random` | Baselines |
| `turning(rate)` | Oracle: turns the pencil flat on the palm at `rate` rad/s |
| `ratchet` | Creeps forward 0.2 rad at 0.5 rad/s, snaps back 0.2 rad at 4 rad/s |
| `shiver` | Turns at 0.5 rad/s with a ±2° shiver every 2 steps |
| `roll` | Spins the pencil about its own axis at 20 rad/s |
| `standing` | Turns the pencil while it stands 60° up |
| `stand_at(deg, rate)` | `standing` at another tilt and rate (50° turning, 70° held still) |
| `toss` | Forward 0.5 rad in hand, back 1.0 rad 5 cm up in the air, set down |
| `sink(env, s, depth)` | Holds the pencil level at 0.125 m, below the held line |
| `push_then_park(seconds, rate)` | Turns at `rate` for `seconds` (2 s at 1 rad/s), then holds the pencil still: pencil4's end state |
| `drop_it` | Puts the pencil beside the hand, below the drop line, on step one |
| `env_drop()` | The default drop penalty |
| `fist` | Curls the four fingers to their limits and holds |
| `resume_is_exact()` | Saves a state after 150 random steps, replays 250 more, resumes the saved state and replays the same 250: true if every observation matches exactly and the resumed episode's yaw starts at zero |
| `main()` | Runs every probe, prints the table, asserts each property, exits 1 on any failure |

### 10.3 `orca_rl/pencil_train.py`

| Name | What it does |
|---|---|
| `PencilMetricsCallback(window=40)` | Collects the final `info` of each finished episode; every 40 episodes logs `task/` over the fresh ones and `task_resumed/` over the resumed ones (`_record`) |
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
| `main()` | Runs the episodes (always from a fresh reset: it sets `stall_start_prob` to 0 whatever the model trained with) and prints revolutions, spin rate, drop rate, stand-ups (when `max_stand_deg` is set), episode length, held fraction, the share held but stalled, goals per episode (with goals), return, and the scripted oracle's revolutions at the model's own rate cap; optionally writes a video, marked DROPPED or STOOD UP when an episode ends early |

Flags: `--model`, `--policy {model,random,zero}`, `--episodes` (50), `--max-episode-steps` (1000), `--seed`
(1234; episode `i` uses `seed + i`), `--stochastic`, `--randomize-physics`, `--video OUT.mp4`, `--fps` (50 =
half speed), `--clean` (1280×720, no text).

## 11. Known issues and next steps

- **Continuous spinning comes from one seed** (Sections 1.3, 1.4). The stall penalty helps every seed, but of
  four pencil5 seeds only seed 1 learned to spin continuously; the others traded parking for standing the
  pencil up or dropping it. Building on pencil5_s1 / pencil5_s1_fast is more reliable than training each new
  variant from scratch and hoping for the seed. Next:
  1. Finger-finger contact: pencil5_s1_fast's fingers touch each other 69% of the time (pencil5_s1: 50%).
     Charge for it, or train with the safety layer, before the real hand. Training on from pencil5_s1_fast
     with a charge for it is in progress.
  2. A higher rate cap: pencil5_s1_fast was still getting faster at the end.
  3. Whether the extra stand-ups come from the stall penalty halving the cost of failing (Section 1.3):
     `--drop-penalty 3`.
  4. Physics randomization, then the observations the real hand can provide (below).
- **Episodes are short compared with the slow failures.** pencil3_60m's sunk pencil usually stands up after
  10 s; γ = 0.995 discounts a cost 5 s ahead to about 8%. Longer episodes or a larger γ would make such
  delayed costs visible.
- **An idle pencil lying along the fingers slips between them.** At headings near 15° and 165° with the hand
  still, the pencil slides into a gap in 4–8 s and falls through on end; this is why "do nothing" drops 10% of
  episodes. It is real v1 geometry; the policy has to handle it (close or curl the fingers, or not stop there).
- **MuJoCo occasionally blows up.** Once in the 3M-step smoke run, twice in pencil1's 30M steps, 3 times in
  pencil2's, never in pencil3's 60M or in pencil4's and pencil5's 30M, once in pencil5_s1's, and once in the seven runs since (pencil5_s3) (`MUJOCO_LOG.TXT`: NaN/Inf in QACC on a pencil rotational degree of freedom). MuJoCo then resets the whole
  state to the XML pose; `PencilSpin.step` detects this from the simulation clock going backwards and
  truncates the episode, unscored. The first event in both runs was identical (simulation time 1.4260,
  DOF 21), so it reproduces with seed 0. If `task/unstable_rate` grows, raise the pencil's long-axis inertia
  (6.4e-8 kg·m²) or shorten the timestep.
- **Simulator-only observations.** The pencil's pose, velocities and per-finger contacts are not available on
  the hardware; before running on the real hand this needs an asymmetric actor-critic or state estimation,
  as for the cube task.
- **Round pencil.** The collision capsule rolls more easily than a real hexagonal pencil. Rolling earns
  nothing (5.3) but makes it easier to roll off the palm.
- **Not trained yet:** physics randomization; the safety layer (`--safety` exists only for the cube task).
- **v1 only.**
