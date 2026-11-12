# Residual tracking of demonstration 003 (CPU MuJoCo, SB3 PPO)

A policy that spins the pen on the ORCA hand by following one retargeted
demonstration, trained on a laptop CPU. It follows Wuji's PenSpin task in
wuji-mjlab: every control step (50 Hz) the policy adds a residual to the
reference finger angles, `target = ref_q[k] + EMA(residual)`, and is rewarded
for keeping the pen and the fingers on the reference. The details are in the
docstring of `track_env.py`.

- Reference: `reference/optimized003-final/` (the retargeted and optimized
  demonstration 003, kept here since `results/` was removed upstream), played
  at the speed of the human demonstration: 273 frames, 5.44 s.
- Replaying the reference with no residual drops the pen within the first second.
- `runs/track003_8/` was trained with physics randomization and a sim2real
  model: servo command delay 0–40 ms, encoder bias and noise, and a camera
  model for the pen pose (calibration offset, jitter, 1–2 ticks of latency,
  dropped frames). About 210M steps, PPO with 10 environments.
- The best policy so far, `runs/track003_18/`, continues it with disturbances: pushes
  on the pen and kicks on the fingertips (Wuji's disturbance events, at the
  strengths of the group's config; `track_env.DISTURB`), and joint speeds
  differenced from the reported angles, as on the hand. 400M steps in all, the
  last 60M with the pushes and kicks 1.3 times as strong, and the last 30M of
  those with a longer reward horizon (discount 0.995 instead of 0.99);
  `runs/track003_10/`, `runs/track003_11/`, `runs/track003_12/` and
  `runs/track003_14/` are earlier stages. track003_18 drops the pen less often
  than track003_14 under pushes but tracks slightly less closely and shakes
  more (see Results); track003_14 is the alternative if shaking matters on the
  hand.

## Results

Each test runs the whole clip from frame 0 after holding it for 1 s, the way a
deployment starts. `runs/track003_5` is the policy before sim2real training.

| Test | track003_5 | track003_8 |
|---|---|---|
| Delay and noise, 100 clips, each its own draw | 42% completed, 1.35 cm / 11.7° | **100%**, 0.79 cm / 3.9° |
| … of which servo delay 28–40 ms (42 clips) | 17% | **100%** |
| Delay and noise, 100 random start frames | 67% reach the end | **100%** |
| Delay and noise + randomized physics, 200 clips | 26% | **96%** |
| Joint shaking above 5 Hz, with delay and noise | 0.233 rad/s | 0.166 rad/s |
| No delay or noise, 100 random start frames | 100%, 0.47 cm / 4.1° | 98%, 0.74 cm / 5.2° |
| No delay or noise, randomized physics, 200 clips | 100%, 0.58 cm / 3.8° | 98%, 0.80 cm / 4.6° |

The full outputs are in `runs/*_eval*.txt`. Videos are in `videos/`
(policy on the left, the forced reference pose on the right):
`track003_8_clean.mp4`, `track003_8_delay_noise.mp4` (34 ms servo delay), and
`track003_5_vs_8_*.mp4` comparing the two policies under delay, low friction
and a heavier pen.

### With disturbances

`bash eval_suite.sh <run>` (outputs in `runs/stage_eval/<run>/`, the same seeds
for every run). All tests but the last have the delay and noise on and
difference the joint speeds, as on the hand; "pushes" are the full-strength pen
pushes and fingertip kicks.

| Test | track003_8 | track003_9 | track003_10 | track003_11 | track003_12 | track003_13 | track003_14 | track003_18 (best so far) |
|---|---|---|---|---|---|---|---|---|
| No pushes, 100 clips | **100%**, 0.79 cm / 4.0° | **100%**, 0.79 cm / 5.9° | **100%**, 0.81 cm / 4.9° | 99%, 0.80 cm / 4.7° | **100%**, 0.79 cm / 5.1° | **100%**, 0.81 cm / 5.2° | **100%**, 0.78 cm / 5.6° | **100%**, 0.84 cm / 5.7° |
| No pushes, 100 random start frames | **100%** reach the end | 99% | 98% | 98% | 98% | **100%** | **100%** | **100%** |
| No pushes, randomized physics, 200 clips | 96% | 93% | 93% | 96% | **98%** | 97% | **98%** | **98%** |
| Pushes at half strength, 100 clips | 88% | 95% | 96% | 98% | 97% | 97% | 98% | **100%** |
| … and randomized physics, 100 clips | 85% | 88% | 92% | 92% | **96%** | 92% | 95% | 94% |
| Pushes, 200 clips | 66% | 86% | 92% | 93% | 92% | 94% | 92% | **96%** |
| Pushes and randomized physics, 200 clips | 65% | 80% | 82% | 87% | 86% | 87% | **92%** | 91% |
| No delay, noise or pushes, randomized physics, 200 clips | 98% | 90% | 95% | 92% | 95% | 92% | **99%** | 98% |

track003_15–17 and track003_19, continuations of track003_14 and track003_18
that were not kept, are left out of this table; their outputs are in `runs/stage_eval/` too. A point or two is
within the noise of 100–200 clips, so the later stages were also run on
further draws, the same for each: seeds 200–999 for the clips and
10200–10599 with randomized physics, with the delay and noise on and the joint
speeds differenced (`evaluate.py --sim2real on --joint-vel diff --starts 0
--seed-offset 200 --s2r-clips 800 --random-physics 400`, with `--disturb 1.0`
for the pushes):

| Further draws | track003_11 | track003_12 | track003_13 | track003_14 | track003_15 | track003_16 | track003_17 | track003_18 | track003_19 |
|---|---|---|---|---|---|---|---|---|---|
| Pushes, 800 clips | 94.5% | 95.5% | 94.6% | 95.2% | 96.1% | 94.2% | 94.5% | 97.2% | 96.5% |
| Pushes and randomized physics, 400 clips | 89.0% | 88.2% | 91.0% | 91.8% | 90.5% | 92.2% | 89.5% | 91.5% | 91.8% |
| No pushes, randomized physics, 400 clips | 94.5% | 97.2% | 97.2% | 98.2% | 96.8% | 98.8% | 97.8% | 97.5% | 98.2% |

Under full pushes with nominal physics track003_11 to track003_17 stay within
two points of each other (94.2–96.1%). With randomized physics, track003_12 drops the pen less often than
track003_11 without pushes, and track003_14 less often than track003_12 with
pushes, as in the table above. track003_15 and track003_16, trained on from
track003_14, do not clearly drop the pen less often; their axis error without
pushes is lower (5.3° against 5.6°), and track003_16's position error higher
(0.84 cm against 0.78 cm). track003_17, trained on from track003_14 with the
pushes and kicks 1.5 times as strong, does not drop the pen less often either;
even under pushes 1.5 times as strong (`--disturb 1.5`, the same 800 clips) it
completes fewer clips than track003_14, 87.2% against 90.0% (56 clips only
track003_14 completes, 34 only track003_17; exact McNemar p = 0.03).

track003_18, trained on from track003_14 as before but with a longer reward
horizon (`--gamma 0.995` instead of 0.99, so that dropping the pen costs more
of the return), drops the pen less often under full pushes: 97.2% of the 800
clips against 95.2% (29 clips only track003_18 completes, 13 only
track003_14; p = 0.02), and with the 200 standard clips added, 97.0% of 1000
against 94.7% (42 against 19; p = 0.004, or 0.02 allowing for the four
continuations of track003_14 tried). With randomized physics (91.5% against
91.8% under pushes, 97.5% against 98.2% without; p ≥ 0.5) and under pushes 1.5
times as strong (90.1% against 90.0%) the two are level. It tracks less
closely and shakes more: without pushes, 0.84 cm against 0.78 cm and 0.199
rad/s against 0.179 (0.201 against 0.183 under pushes).

track003_19, trained on from track003_18 with a still longer horizon
(`--gamma 0.998`), does not drop the pen less often under full pushes (96.5%
of the 800 clips against 97.2%; 8 clips only track003_19 completes, 14 only
track003_18; p = 0.29) or with randomized physics (91.8% against 91.5% under
pushes, 98.2% against 97.5% without; p ≥ 0.5). Under pushes 1.5 times as
strong it completes more, 92.4% against 90.1% (37 against 19; p = 0.02, or
about 0.09 allowing for the four comparisons made), but it
tracks clearly less closely: without pushes 0.97 cm / 7.1° against
0.84 cm / 5.7°, and it shakes slightly more (0.205 rad/s against 0.199).

Videos (red arrows push the pen, orange ones kick the fingertips):
`videos/track003_10_pushes.mp4`, track003_10 under the delay, noise and full
pushes (a draw with 38 ms of servo delay), and `videos/track003_8_vs_10_pushes.mp4`,
both policies under the same draw and the same pushes, picked as one where
track003_8 drops the pen and track003_10 does not. `videos/track003_14_pushes.mp4`
is track003_14 completing the clip under the delay, noise and full pushes, on a
draw with 40 ms of servo delay (the top of the trained range) and 2 ticks of
camera lag; track003_14 completes 36 of seeds 0–39 under the same conditions.
`videos/track003_14_vs_18_pushes.mp4` puts track003_14 and track003_18 under
the same draw (34 ms of servo delay) and the same pushes: track003_14 drops
the pen at frame 135 and track003_18 completes the clip. Of seeds 0–39,
track003_18 completes all four that track003_14 drops and drops one (seed 27)
that track003_14 completes. `videos/track003_18_pushes.mp4` is track003_18 on
the same draw as `videos/track003_14_pushes.mp4` (40 ms of servo delay, 2
ticks of camera lag, the same pushes): it completes the clip, with a mean
error of 0.91 cm / 7.1° against track003_14's 0.82 cm / 6.2°.


## Training history

| Run | Steps | Change | Outcome |
|---|---|---|---|
| track003_1 | 20M | first run, clip 3× slower | tracks (0.52 cm / 2.7°) but shakes |
| track003_2 | +15M | action-rate penalty 0.1→0.5, EMA alpha 0.5→0.3, min std 0.2→0.1 | shaking down 85% |
| track003_3 | +15M | clip at the human speed | 0.54 cm / 2.3° |
| track003_4 | +15M | pen angular velocity in the world frame | 0.49 cm / 2.3°, but fails at friction ×0.5 or pen mass ×2 |
| track003_5 | +20M | physics randomization (friction, pen mass, servo kp, joint damping) | robust to physics, not to delay |
| track003_6–8 | +40M, +45M, +40M | sim2real model; track003_8 at lr 5e-5 | as above |
| track003_9 | +40M | pen pushes and fingertip kicks, ramped from 0 to full strength over 5–60% of the run | 66% → 86% of clips completed under pushes; axis error without pushes 4.0° → 5.9° |
| track003_10 | +30M | full strength from the start, joint speeds differenced, lr 5e-5 | 92% under pushes; axis error without pushes down to 4.9° |
| track003_11 | +30M | as track003_10 | 93% under pushes, 87% with randomized physics too; 96% with randomized physics and no pushes |
| track003_12 | +30M | as track003_10 | as track003_11 under pushes (92%, 86%); 98% with randomized physics and no pushes; shaking down 8% |
| track003_13 | +30M on track003_12 | axis term of the reward 8 → 12 (`--w-rot 12`) | axis error unchanged (5.2°), shaking 0.169 → 0.181 rad/s; no clear gain over track003_12, not kept |
| track003_14 | +30M on track003_12 | pushes and kicks 1.3 times as strong (`--disturb 1.3`; tested at 1) | 92% under pushes with randomized physics (track003_12: 86%), 99% with randomized physics and no delay, noise or pushes; axis error without pushes 5.1° → 5.6°, shaking 0.169 → 0.179 rad/s |
| track003_15 | +30M on track003_14 | as track003_14 | 96% under pushes (track003_14: 92%), but 96% with randomized physics and no delay, noise or pushes (99%); level with track003_14 on further draws; axis error without pushes 5.3° (track003_14: 5.6°); not kept |
| track003_16 | +30M on track003_14 | as track003_14 at half the learning rate (2.5e-5) | level with track003_14 (92% under pushes, with and without randomized physics); axis error without pushes 5.3° (track003_14: 5.6°), but position error 0.84 cm (0.78 cm); not kept |
| track003_17 | +30M on track003_14 | pushes and kicks 1.5 times as strong (`--disturb 1.5`) | 94% under pushes, 88% with randomized physics too (track003_14: 92%, 92%), 96% with randomized physics and no delay, noise or pushes (99%); not clearly different from track003_14 on further draws (lower in all three, p ≥ 0.18), and below it under pushes 1.5 times as strong (87.2% against 90.0%); axis error without pushes 5.7° (track003_14: 5.6°), position error 0.82 cm (0.78 cm); not kept |
| track003_18 | +30M on track003_14 | as track003_14 with a longer reward horizon, discount 0.99 → 0.995 (`--gamma 0.995`) | 96% under pushes (track003_14: 92%); on further draws 97.2% against 95.2% under pushes (p = 0.02; 97.0% against 94.7% with the 200 standard clips, p = 0.004), level with randomized physics (p ≥ 0.5) and under pushes 1.5 times as strong (90.1% against 90.0%); axis error without pushes 5.7° (5.6°), position error 0.84 cm (0.78 cm), shaking 0.179 → 0.199 rad/s |
| track003_19 | +30M on track003_18 | as track003_18 with discount 0.995 → 0.998 (`--gamma 0.998`) | 95% under pushes, 91% with randomized physics too (track003_18: 96%, 91%); on further draws level under pushes (96.5% against 97.2%, p = 0.29) and with randomized physics (p ≥ 0.5), higher under pushes 1.5 times as strong (92.4% against 90.1%, p = 0.02, about 0.09 allowing for the four comparisons); axis error without pushes 7.1° (track003_18: 5.7°), position error 0.97 cm (0.84 cm), shaking 0.205 rad/s (0.199); not kept |

## Usage

```bash
pip install -r requirements.txt
python train.py --name my_run --resume runs/track003_18/final_model.zip --timesteps 10_000_000 \
    --time-scale 1 --w-action-rate 0.5 --ema-alpha 0.3 --min-std 0.1 --randomize-physics --sim2real \
    --joint-vel diff --disturb 1.3 --disturb-ramp 0 0 --lr 5e-5 --gamma 0.995
python evaluate.py --model runs/track003_18/final_model.zip --sim2real on --joint-vel diff --random-physics 40
python evaluate.py --model runs/track003_18/final_model.zip --sim2real on --joint-vel diff --disturb 1.0 --s2r-clips 40
PY=python bash eval_suite.sh track003_18   # the standard suite, into runs/stage_eval/track003_18/
python evaluate.py --model runs/track003_8/final_model.zip --sim2real off --video out.mp4
python compare_video.py runs/track003_5/final_model.zip runs/track003_8/final_model.zip \
    --labels before after --sim2real on --seed 2 --out compare.mp4
python compare_video.py runs/track003_18/final_model.zip --labels track003_18 \
    --sim2real on --joint-vel diff --disturb 1 --seed 39 --out pushes.mp4   # as in videos/track003_18_pushes.mp4
                                       # (videos/track003_14_pushes.mp4 with track003_14)
python compare_video.py runs/track003_14/final_model.zip runs/track003_18/final_model.zip \
    --labels track003_14 track003_18 --sim2real on --joint-vel diff --disturb 1 --seed 0 \
    --out compare_pushes.mp4   # as in videos/track003_14_vs_18_pushes.mp4
```

A policy needs `final_model.zip`, `vecnormalize.pkl` (observation
normalization) and `env_kwargs.json` from its run folder.

## Deploying

- The pen's velocity in the observation must be computed exactly as in
  training: every control tick (0.02 s), the camera pose on hand minus the
  previous tick's, divided by 0.02 s with no timestamp correction, so zero
  while no new frame arrives. The angular velocity is the quaternion difference
  in the previous pose's frame over 0.02 s, rotated to the world frame.
- track003_10 and later were trained with the joint speeds differenced the same way: the
  reported angles minus the previous tick's, over 0.02 s, zero on the first
  tick (`env_kwargs.json`: `"joint_vel": "diff"`).
- The delay and noise magnitudes follow Wuji's hand2 config; the ORCA hardware
  is unmeasured. Measure the servo command latency (trained 0–40 ms) and the
  camera latency (trained 20–40 ms) first, and retrain with wider ranges if
  they fall outside.
