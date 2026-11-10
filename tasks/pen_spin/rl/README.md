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
- The final policy, `runs/track003_8/`, was trained with physics randomization
  and a sim2real model: servo command delay 0–40 ms, encoder bias and noise, and
  a camera model for the pen pose (calibration offset, jitter, 1–2 ticks of
  latency, dropped frames). About 210M steps in all, PPO with 10 environments.

## Results

Each test runs the whole clip from frame 0 after holding it for 1 s, the way a
deployment starts. `runs/track003_5` is the policy before sim2real training.

| Test | track003_5 | track003_8 (final) |
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

## Training history

| Run | Steps | Change | Outcome |
|---|---|---|---|
| track003_1 | 20M | first run, clip 3× slower | tracks (0.52 cm / 2.7°) but shakes |
| track003_2 | +15M | action-rate penalty 0.1→0.5, EMA alpha 0.5→0.3, min std 0.2→0.1 | shaking down 85% |
| track003_3 | +15M | clip at the human speed | 0.54 cm / 2.3° |
| track003_4 | +15M | pen angular velocity in the world frame | 0.49 cm / 2.3°, but fails at friction ×0.5 or pen mass ×2 |
| track003_5 | +20M | physics randomization (friction, pen mass, servo kp, joint damping) | robust to physics, not to delay |
| track003_6–8 | +40M, +45M, +40M | sim2real model; track003_8 at lr 5e-5 | as above |

## Usage

```bash
pip install -r requirements.txt
python train.py --name my_run --resume runs/track003_8/final_model.zip --timesteps 10_000_000 \
    --time-scale 1 --w-action-rate 0.5 --ema-alpha 0.3 --min-std 0.1 --randomize-physics --sim2real
python evaluate.py --model runs/track003_8/final_model.zip --sim2real on --random-physics 40
python evaluate.py --model runs/track003_8/final_model.zip --sim2real off --video out.mp4
python compare_video.py runs/track003_5/final_model.zip runs/track003_8/final_model.zip \
    --labels before after --sim2real on --seed 2 --out compare.mp4
```

A policy needs `final_model.zip`, `vecnormalize.pkl` (observation
normalization) and `env_kwargs.json` from its run folder.

## Deploying

- The pen's velocity in the observation must be computed exactly as in
  training: every control tick (0.02 s), the camera pose on hand minus the
  previous tick's, divided by 0.02 s with no timestamp correction, so zero
  while no new frame arrives. The angular velocity is the quaternion difference
  in the previous pose's frame over 0.02 s, rotated to the world frame.
- The delay and noise magnitudes follow Wuji's hand2 config; the ORCA hardware
  is unmeasured. Measure the servo command latency (trained 0–40 ms) and the
  camera latency (trained 20–40 ms) first, and retrain with wider ranges if
  they fall outside.
