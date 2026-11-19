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
- The best policy so far, `runs/track003_22/`, continues it with disturbances: pushes
  on the pen and kicks on the fingertips (Wuji's disturbance events, at the
  strengths of the group's config; `track_env.DISTURB`), and joint speeds
  differenced from the reported angles, as on the hand. 430M steps in all, the
  last 90M with the pushes and kicks 1.3 times as strong, the last 60M of
  those with a longer reward horizon (discount 0.995 instead of 0.99), and the
  last 30M with twice the samples per update; `runs/track003_10/`,
  `runs/track003_11/`, `runs/track003_12/`, `runs/track003_14/` and
  `runs/track003_18/` are earlier stages. track003_22 drops the pen less often
  than track003_14 and track003_18 under pushes but tracks slightly less
  closely and shakes slightly more than track003_14 (see Results);
  track003_14 is the alternative if shaking matters on the hand.

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

| Test | track003_8 | track003_9 | track003_10 | track003_11 | track003_12 | track003_13 | track003_14 | track003_18 | track003_22 (best so far) |
|---|---|---|---|---|---|---|---|---|---|
| No pushes, 100 clips | **100%**, 0.79 cm / 4.0° | **100%**, 0.79 cm / 5.9° | **100%**, 0.81 cm / 4.9° | 99%, 0.80 cm / 4.7° | **100%**, 0.79 cm / 5.1° | **100%**, 0.81 cm / 5.2° | **100%**, 0.78 cm / 5.6° | **100%**, 0.84 cm / 5.7° | **100%**, 0.83 cm / 5.8° |
| No pushes, 100 random start frames | **100%** reach the end | 99% | 98% | 98% | 98% | **100%** | **100%** | **100%** | 99% |
| No pushes, randomized physics, 200 clips | 96% | 93% | 93% | 96% | **98%** | 97% | **98%** | **98%** | **98%** |
| Pushes at half strength, 100 clips | 88% | 95% | 96% | 98% | 97% | 97% | 98% | **100%** | **100%** |
| … and randomized physics, 100 clips | 85% | 88% | 92% | 92% | 96% | 92% | 95% | 94% | **98%** |
| Pushes, 200 clips | 66% | 86% | 92% | 93% | 92% | 94% | 92% | 96% | **99%** |
| Pushes and randomized physics, 200 clips | 65% | 80% | 82% | 87% | 86% | 87% | 92% | 91% | **94%** |
| No delay, noise or pushes, randomized physics, 200 clips | 98% | 90% | 95% | 92% | 95% | 92% | **99%** | 98% | 98% |

track003_15–17, track003_19–21 and track003_23–34, later runs that were not
kept (see the training history for what each was trained on from), are left
out of this table; their outputs are in `runs/stage_eval/` too. A point or two is
within the noise of 100–200 clips, so the later stages were also run on
further draws, the same for each: seeds 200–999 for the clips and
10200–10599 with randomized physics, with the delay and noise on and the joint
speeds differenced (`evaluate.py --sim2real on --joint-vel diff --starts 0
--seed-offset 200 --s2r-clips 800 --random-physics 400`, with `--disturb 1.0`
for the pushes):

| Further draws | track003_11 | track003_12 | track003_13 | track003_14 | track003_15 | track003_16 | track003_17 | track003_18 | track003_19 | track003_20 | track003_21 | track003_22 | track003_23 | track003_24 | track003_25 | track003_26 | track003_27 | track003_28 | track003_29 | track003_30 | track003_31 | track003_32 | track003_33 | track003_34 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pushes, 800 clips | 94.5% | 95.5% | 94.6% | 95.2% | 96.1% | 94.2% | 94.5% | 97.2% | 96.5% | 98.0% | 95.8% | 99.1% | 99.5% | 98.8% | 97.4% | 97.9% | 98.5% | 98.4% | 98.9% | 98.4% | 98.2% | 99.2% | 99.2% | 97.4% |
| Pushes and randomized physics, 400 clips | 89.0% | 88.2% | 91.0% | 91.8% | 90.5% | 92.2% | 89.5% | 91.5% | 91.8% | 91.2% | 90.2% | 91.8% | 93.8% | 93.8% | 95.0% | 95.5% | 94.0% | 95.8% | 94.2% | 95.5% | 92.8% | 94.2% | 95.5% | 91.8% |
| No pushes, randomized physics, 400 clips | 94.5% | 97.2% | 97.2% | 98.2% | 96.8% | 98.8% | 97.8% | 97.5% | 98.2% | 96.8% | 98.0% | 98.2% | 98.0% | 98.5% | 99.0% | 98.0% | 98.8% | 99.2% | 99.0% | 98.5% | 98.0% | 98.0% | 97.8% | 98.2% |

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

track003_20, trained on from track003_18 with the action-rate penalty doubled
(`--w-action-rate 1.0`) to cut the shaking, shakes a little less with the delay
and noise on (0.191 rad/s against 0.199 without pushes, 0.193 against 0.201
under pushes), still more than track003_14 (0.179 and 0.183), and drops the
pen about as often: 98.0% of the 800 clips under pushes against 97.2% (14
clips only track003_20 completes, 8 only track003_18; p = 0.29), with
randomized physics 91.2% against 91.5% under pushes and 96.8% against 97.5%
without (p ≥ 0.6), and 90.2% against 90.1% under pushes 1.5 times as strong. Its axis
error without pushes is higher, 6.1° against 5.7° (position 0.83 cm against
0.84 cm).

track003_21, trained on from track003_18 with the same settings, does no
better, and under full pushes somewhat worse: 95.8% of the 800 clips against
97.2% (11 clips only
track003_21 completes, 23 only track003_18; p = 0.06), with randomized physics
90.2% against 91.5% under pushes and 98.0% against 97.5% without (p ≥ 0.5),
and 89.5% against 90.1% under pushes 1.5 times as strong. Its axis error
without pushes is higher, 6.2° against 5.7° (position 0.87 cm against
0.84 cm), and it shakes about as much (0.195 rad/s against 0.199). None of
these three continuations of track003_18 drops the pen significantly less often
under full pushes, and all three have a higher axis error without pushes
(6.1–7.1° against 5.7°).

track003_22, trained on from track003_18 with twice the samples per update
(`--n-steps 2048 --batch-size 5120`: each update uses 20480 steps instead of
10240, in minibatches of 5120 instead of 2560), drops the pen less often under
pushes: 99.1% of the 800 clips against 97.2% (18 clips only track003_22
completes, 3 only track003_18; p = 0.001), and with the 200 standard clips
added, 99.1% of 1000 against 97.0% (25 against 4; p = 0.0001, or 0.0004
allowing for the four continuations of track003_18 tried). Under pushes 1.5
times as strong the gap is wider, 95.9% against 90.1% (54 against 8). With
randomized physics the two are level (91.8% against 91.5% under pushes, 98.2%
against 97.5% without; p ≥ 0.5), so most of the drops left are with
randomized physics. It tracks about as closely (without pushes 0.83 cm / 5.8°
against 0.84 cm / 5.7°) and shakes slightly less (0.190 rad/s against 0.199;
0.192 against 0.201 under pushes), though still more than track003_14 (0.179
and 0.183).

track003_23, trained on from track003_22 with the samples per update doubled
again (`--n-steps 4096 --batch-size 10240`), is not clearly better: 99.5% of
the 800 clips under pushes against 99.1% (5 clips only track003_23 completes, 2
only track003_22; p = 0.45), with randomized physics 93.8% against 91.8% under
pushes (12 against 4; p = 0.08) and 98.0% against 98.2% without, and 96.1%
against 95.9% under pushes 1.5 times as strong. It tracks about as closely
(without pushes 0.85 cm / 5.9° against 0.83 cm / 5.8°) and shakes about as
much (0.187 rad/s against 0.190).

track003_24, trained on from track003_23 with the same settings, is not
clearly different from track003_22 either: 98.8% of the 800 clips under pushes
against 99.1% (4 clips only track003_24 completes, 7 only track003_22; p =
0.55), with randomized physics 93.8% against 91.8% under pushes (15 against 7;
p = 0.13) and 98.5% against 98.2% without, and 95.6% against 95.9% under
pushes 1.5 times as strong. Like track003_23 it completes more of the clips
with randomized physics and pushes than track003_22, but in neither run is the
gap significant. It tracks about as closely (without pushes 0.84 cm / 5.8°
against 0.83 cm / 5.8°) and shakes about as much (0.186 rad/s against 0.190).

track003_25, trained on from track003_24 at twice the learning rate (`--lr
1e-4`, as each update now uses four times the samples of track003_18), drops
the pen more often under pushes with nominal physics than track003_22: 97.4%
of the 800 clips against 99.1% (6 clips only track003_25 completes, 20 only
track003_22; p = 0.009), 96.9% of 1000 against 99.1% with the 200 standard
clips added (7 against 29; p = 0.0003), and 92.5% against 95.9% under pushes
1.5 times as strong (15 against 42; p = 0.0005). With randomized physics it
drops it less often, though not clearly so: 95.0% against 91.8% under pushes
(20 against 7; p = 0.02, or about 0.06 allowing for the three runs compared
with track003_22) and 99.0% against 98.2% without. Against track003_24, the
same run at the lower learning rate, the gap with randomized physics and
pushes is smaller (95.0% against 93.8%; p = 0.33), while the losses under
pushes with nominal physics are clear (97.4% against 98.8%, p = 0.02; 92.5%
against 95.6% under pushes 1.5 times as strong, p < 0.0001). Its axis error
without pushes is lower, 5.3° against 5.8° (position 0.86 cm against
0.83 cm), and it shakes slightly more (0.194 rad/s against 0.190; 0.198
against 0.192 under pushes). Its action noise grew over the run (standard
deviation 0.179 → 0.199 at the end of training).

track003_26, trained on from track003_25 back at the learning rate of 5e-5,
shows the same trade, keeping track003_25's gain with randomized physics
(95.5% against its 95.0%; p = 0.77) and losing less under pushes with nominal
physics. With randomized physics and pushes it drops the pen less often than
track003_22: 95.5% of the 400 clips against 91.8% (22 clips only track003_26
completes, 7 only track003_22; p = 0.008, or 0.03 allowing for the four runs
compared with track003_22), a gain not borne out on fresh draws (see Fresh
draws below); without pushes the two are level (98.0% against 98.2%). Under pushes with nominal physics it drops it more often:
97.9% of the 800 clips against 99.1% (7 against 17; p = 0.06), 97.2% of 1000
against 99.1% with the 200 standard clips added (8 against 27; p = 0.002), and
94.6% against 95.9% under pushes 1.5 times as strong (18 against 28; p =
0.18), though there it completes more than track003_25 (94.6% against 92.5%;
p = 0.003). It tracks more closely without pushes (0.80 cm / 5.3° against
0.83 cm / 5.8°) and shakes about as much (0.186 rad/s against 0.190). Along
the line track003_18 → 22 → 23 → 24 → 25 → 26 the policy's action noise grew
in every run (standard deviation at the end of training 0.147, 0.159, 0.169,
0.179, 0.199 and 0.210), and over the same runs the share of clips completed
with randomized physics and pushes rose from 91.5% to 95.5% on those draws;
track003_19–21's noise grew too (to 0.151–0.157) without that gain. The rise
is not borne out on fresh draws (see Fresh draws below): there track003_22,
track003_23 and track003_26 complete 94.2%, 94.1% and 93.6%.

track003_27, trained on from track003_26 with the entropy bonus off
(`--ent-coef 0` instead of 0.001), barely brought the action noise down
(standard deviation 0.210 → 0.206 at the end of training) and is not clearly
different from track003_26 in any test (p ≥ 0.2). Against track003_22 it
drops the pen more often under pushes with nominal physics: 98.5% of the 800
clips against 99.1% (3 clips only track003_27 completes, 8 only track003_22;
p = 0.23), 97.9% of 1000 against 99.1% with the 200 standard clips added (3
against 15; p = 0.008), and 94.6% against 95.9% under pushes 1.5 times as
strong (14 against 24; p = 0.14). With randomized physics it completes 94.0%
against 91.8% under pushes (21 against 12; p = 0.16) and 98.8% against 98.2%
without. It tracks more closely without pushes (0.78 cm / 5.4°
against 0.83 cm / 5.8°) and shakes about as much (0.188 rad/s against 0.190).

track003_28, trained on from track003_27 with the action noise capped
(`--max-std 0.15`, the entropy bonus still off), brought the noise down
(standard deviation 0.206 → 0.139 at the end of training, below
track003_22's 0.159) but is not clearly different from track003_27 or
track003_26 in any test (p ≥ 0.11): 30M steps at the lower noise neither won
back track003_22's results under pushes with nominal physics nor lost the
gain with randomized physics and pushes. Against track003_22 it shows the
same trade as track003_26. With randomized physics and pushes it drops the pen less often:
95.8% of the 400 clips against 91.8% (22 clips only track003_28 completes, 6
only track003_22; p = 0.004, or 0.02 allowing for the six runs compared with
track003_22), a gain not borne out on fresh draws (see Fresh draws below);
without pushes it completes 99.2% against 98.2% (5 against 1; p = 0.22). Under pushes with nominal physics it drops it more often: 98.4% of
the 800 clips against 99.1% (5 against 11; p = 0.21), 97.6% of 1000 against
99.1% with the 200 standard clips added (6 against 21; p = 0.006), and 93.8%
against 95.9% under pushes 1.5 times as strong (9 against 26; p = 0.006). It
tracks more closely without pushes (0.79 cm / 5.6° against 0.83 cm / 5.8°)
and shakes about as much (0.185 rad/s against 0.190).

track003_29, trained on from track003_24 (the last run before track003_25's
higher learning rate) with the action noise capped and the entropy bonus off
as in track003_28 (`--max-std 0.15 --ent-coef 0`), brought the noise down
(standard deviation 0.179 → 0.133 at the end of training). It is level with
track003_22 under full pushes with nominal physics: 98.9% of the 800 clips against 99.1% (6 clips only
track003_29 completes, 8 only track003_22; p = 0.79) and 98.8% of 1000
against 99.1% with the 200 standard clips added (7 against 10; p = 0.63).
With randomized physics it completes more, though not clearly so: 94.2%
against 91.8% under pushes (16 against 6; p = 0.05, or about 0.37 allowing
for the seven runs compared with track003_22) and 99.0% against 98.2% without
(4 against 1; p = 0.38). Under pushes 1.5 times as strong it drops the pen
more often, 94.0% against 95.9% (14 against 29; p = 0.03), and more often
than track003_24 too (94.0% against 95.6%; 9 against 22; p = 0.03), with
which it is otherwise level (p ≥ 0.6). Against track003_28 it completes more
under pushes with nominal physics: 98.9% of the 800 clips against 98.4% (9
against 5; p = 0.42) and 98.8% of 1000 against 97.6% with the 200 standard
clips added (17 against 5; p = 0.02). With randomized physics and pushes it
completes fewer (94.2% against 95.8%; 3 against 9; p = 0.15). Without pushes
it tracks about as closely as track003_22 (0.83 cm / 5.6° against 0.83 cm /
5.8°) and shakes slightly less (0.184 rad/s against 0.190; 0.186 against
0.192 under pushes).

track003_30, trained on from track003_29 with the same settings (standard
deviation of the action noise 0.133 → 0.131 at the end of training), is not
clearly different from track003_28 in any test (p ≥ 0.2). Against
track003_29 it drops the pen more often under pushes with nominal physics:
98.4% of the 800 clips against 98.9% (3 clips only track003_30 completes, 7
only track003_29; p = 0.34) and 97.7% of 1000 against 98.8% with the 200
standard clips added (3 against 14; p = 0.01), most of the gap being on the
200 standard clips (95.0% against 98.5%; 0 against 7; p = 0.02). With
randomized physics and pushes it completes 95.5% against 94.2% (12 against 7;
p = 0.36). Against track003_22 it completes more with randomized physics,
though not clearly so: 95.5% of the 400 clips against 91.8% under pushes (23
against 8; p = 0.01, or about 0.09 allowing for the eight runs compared with
track003_22) and 98.5% against 98.2% without (3 against 2; p = 1.0). Under
pushes with nominal physics it drops the pen more often: 98.4% of the 800
clips against 99.1% (4 against 10; p = 0.18), 97.7% of 1000 against 99.1%
with the 200 standard clips added (5 against 19; p = 0.007), and 94.8%
against 95.9% under pushes 1.5 times as strong (12 against 21; p = 0.16).
Without pushes it tracks as closely as track003_22 (0.83 cm / 5.8° against
0.83 cm / 5.8°) and shakes slightly less (0.181 rad/s against 0.190; 0.185
against 0.192 under pushes).

track003_31, trained on from track003_22 with track003_22's own settings
(`--n-steps 2048 --batch-size 5120`, the entropy bonus 0.001 and no cap on
the action noise), does not improve on it either. Under pushes with nominal
physics it completes 98.2% of the 800 clips against 99.1% (4 clips only
track003_31 completes, 11 only track003_22; p = 0.12), 98.0% of 1000 against
99.1% with the 200 standard clips added (5 against 16; p = 0.03), and 94.8%
against 95.9% under pushes 1.5 times as strong (18 against 27; p = 0.23).
With randomized physics it completes 92.8% against 91.8% under pushes (13
against 9; p = 0.52) and 98.0% against 98.2% without (3 against 4; p = 1.0).
Its action noise grew (standard deviation 0.159 → 0.172 at the end of
training) and it shakes more (0.200 rad/s against 0.190; 0.201 against 0.192
under pushes); without pushes its position error is 0.86 cm against 0.83 cm
and its axis error 5.7° against 5.8°.

track003_32, trained on from track003_23 at half the learning rate (2.5e-5)
with the action noise capped and the entropy bonus off (`--max-std 0.15
--ent-coef 0`; standard deviation of the action noise 0.169 → 0.129 at the
end of training), is not clearly different from track003_23 in any test on
these draws (p ≥ 0.17; on the fresh draws it drops the pen more often under
pushes with nominal physics, see Fresh draws below). Against track003_22 it
is level under pushes with nominal physics: it completes 99.2% of the 800
clips against 99.1% (4 clips only track003_32 completes, 3 only track003_22;
p = 1.0), 99.0% of 1000 against 99.1% with the 200 standard clips added (5
against 6; p = 1.0), and 95.8% against 95.9% under pushes 1.5 times as
strong (14 against 15; p = 1.0). With randomized physics it completes more,
though not clearly so: 94.2% against 91.8% under pushes (16 against 6;
p = 0.05, or about 0.52 allowing for the ten runs compared with
track003_22) and 98.0% against 98.2% without (2 against 3; p = 1.0). Without
pushes it tracks about as closely as track003_22 (0.84 cm / 5.9° against
0.83 cm / 5.8°) and shakes about as much (0.188 rad/s against 0.190; 0.190
against 0.192 under pushes).

track003_33, trained on from track003_22 with track003_32's settings (half the
learning rate, the action noise capped, the entropy bonus off and `--n-steps
4096 --batch-size 10240`; standard deviation of the action noise 0.159 → 0.124
at the end of training), completes more than track003_22 with randomized
physics and pushes on these draws: 95.5% of the 400 clips against 91.8% (19
clips only track003_33 completes, 4 only track003_22; p = 0.003, or about 0.03
allowing for the eleven runs compared with track003_22), a gain not borne out
on fresh draws, where it drops the pen more often than track003_22 (101 drops
in the 2800 clips against 75, p = 0.009; see Fresh draws below); without
pushes the two are level (97.8% against 98.2%; 2 against 4; p = 0.69). Under
full pushes with nominal physics it is level, 99.2% of the 800 clips against
99.1% (4 against 3; p = 1.0) and 99.1% of 1000 against 99.1% with the 200
standard clips added (5 against 5; p = 1.0), but under pushes 1.5 times as
strong it drops the pen more often, 94.5% against 95.9% (7 against 18; p =
0.04). Without pushes it tracks about as closely as track003_22 (0.85 cm /
5.9° against 0.83 cm / 5.8°) and shakes about as much (0.189 rad/s against
0.190; 0.190 against 0.192 under pushes).

track003_34, trained on from track003_22 with track003_22's own settings but
GAE lambda 0.98 instead of 0.95 (`--gae-lambda 0.98`; standard deviation of
the action noise 0.159 → 0.179 at the end of training), drops the pen more
often under pushes with nominal physics: 97.4% of the 800 clips against
track003_22's 99.1% (4 clips only track003_34 completes, 18 only track003_22;
p = 0.004), 96.8% of 1000 against 99.1% with the 200 standard clips added (4
against 27; p < 0.0001), and 91.9% against 95.9% under pushes 1.5 times as
strong (13 against 45; p < 0.0001). It is also below track003_31, trained on
from track003_22 with the same settings apart from GAE lambda, under pushes
1.5 times as strong (91.9% against 94.8%; 14 against 37; p = 0.002) and over
the 1000 clips (96.8% against 98.0%; 7 against 19; p = 0.03), and not clearly
different from it otherwise (p ≥ 0.14). With randomized physics it is level
with track003_22: 91.8% of the 400 clips against 91.8% under pushes (21
against 21; p = 1.0) and 98.2% against 98.2% without (4 against 4; p = 1.0).
Without pushes it tracks less closely than track003_22 (0.87 cm / 6.3° against
0.83 cm / 5.8°) and shakes more (0.201 rad/s against 0.190; 0.204 against
0.192 under pushes), about as much as track003_31 (0.200; 0.201 under pushes).

#### Fresh draws

Every run from track003_24 to track003_31 completed fewer of the clips under pushes with
nominal physics than track003_22 did on the draws above (track003_23 more,
99.5% against 99.1%, though not clearly), track003_31 with track003_22's own
settings included. track003_22 was chosen on those same draws, so its figure
there could be flattered. Seven of the runs, track003_22 and six later ones
(track003_24, track003_25 and track003_27 left out), were therefore also run
on draws none of them was chosen on: seeds 1000–2999 under pushes with
nominal physics (2000 clips; `evaluate.py --sim2real on --joint-vel diff
--disturb 1.0 --starts 0 --seed-offset 1000 --s2r-clips 2000`) and seeds
10600–11399 under pushes with randomized physics (800 clips; `--seed-offset
600 --s2r-clips 0 --random-physics 800`). The test was fixed before the
results were seen: a run would replace track003_22 only if it dropped the pen
significantly less often over the 2800 clips together (exact McNemar p < 0.05
allowing for the six runs compared with track003_22) and not significantly
more often (p < 0.05, not adjusted) in either set alone.

| Pen dropped, fresh draws | track003_22 | track003_23 | track003_26 | track003_28 | track003_29 | track003_30 | track003_31 | track003_32 (after the test) | track003_33 (after the test) | track003_34 (after the test) |
|---|---|---|---|---|---|---|---|---|---|---|
| Pushes, 2000 clips | 29 | **22** | 51 | 48 | 34 | 37 | 37 | 37 | 41 | 59 |
| Pushes and randomized physics, 800 clips | 46 | 47 | 51 | 49 | **43** | 46 | 51 | 44 | 60 | 76 |
| Both, 2800 clips | 75 | **69** | 102 | 97 | 77 | 83 | 88 | 81 | 101 | 135 |

No run of the six passes. track003_23 drops the pen least often (69 against 75; 33
clips only track003_23 completes, 27 only track003_22; p = 0.52). track003_26
and track003_28 drop it more often under pushes with nominal physics (51 and
48 against 29; p = 0.003 and 0.004), and their gain with randomized physics
and pushes on the earlier draws (95.5% and 95.8% against 91.8%) does not hold
up here (51 and 49 drops against 46; p = 0.58 and 0.78): on these draws
track003_22 completes 94.2% of the clips with randomized physics and pushes
(91.8% on the earlier draws), and track003_26 and track003_28 93.6% and 93.9%
(95.5% and 95.8%). Over the 2800 clips track003_26 and track003_28 drop it
more often (102 and 97 against 75; p = 0.01 and 0.03, about 0.06 and 0.17
allowing for the six runs), and the other four runs of the test are not
clearly different from track003_22 (p ≥ 0.2). Under pushes with nominal physics track003_22 drops
the pen in 1.45% of the fresh clips against 0.9% on the earlier draws, a
rise in line with the other six runs' (0.1–0.8 points), so its figure there
was not much flattered. Under the test fixed beforehand track003_22 stays the
policy to use; track003_23 is level with it rather than worse (fewer drops
under pushes with nominal physics on both the earlier and the fresh draws, 4
against 7 and 22 against 29, neither significant). The paired counts, and the
breakdown below, are in `runs/stage_eval/fresh_draws.txt`.

track003_32, track003_33 and track003_34, trained after the test, were run on
the same fresh draws (last three columns; not part of the test). track003_32 is not
clearly different from track003_22 there: 81 drops in the 2800 clips against 75 (32 clips only track003_32
completes, 38 only track003_22; p = 0.55), 37 against 29 under pushes with
nominal physics (p = 0.20) and 44 against 46 with randomized physics and
pushes (p = 0.87). Under pushes with nominal physics it drops the pen more
often than track003_23, which it was trained on from (37 against 22; 8 clips
only track003_32 completes, 23 only track003_23; p = 0.01). track003_33
drops the pen more often than track003_22: 101 times in the 2800 clips
against 75 (33 clips only track003_33 completes, 59 only track003_22;
p = 0.009), 41 against 29 under pushes with nominal physics (p = 0.08) and
60 against 46 with randomized physics and pushes (p = 0.07). As with
track003_26 and track003_28, its gain with randomized physics and pushes on
the earlier draws (95.5% against 91.8%) does not hold up: on these draws it
completes 92.5% of those clips against track003_22's 94.2%. track003_34
drops the pen more often than track003_22 in both sets: 135 times in the 2800
clips against 75 (32 clips only track003_34 completes, 92 only track003_22;
p < 0.0001), 59 against 29 under pushes with nominal physics (p < 0.0001) and
76 against 46 with randomized physics and pushes (p = 0.0004); it also drops
it more often than track003_31 (135 against 88; 32 against 79; p < 0.0001).

Where track003_22 drops the pen with randomized physics and pushes, over the
earlier and the fresh draws together (the 1200 clips of seeds 10200–11399; not
part of the test): in 12.0% of the quarter with the heaviest pens
(1.39–2 times the nominal mass) against 4.3–5.3% in the other three quarters,
and in 14.0% of the clips with both a pen in the heaviest third (1.25–2
times) and friction in the highest third (1.05–1.5 times; 17 of 121). Across
the quarters of friction, servo stiffness and joint damping, and across servo
delays, the rate stays between 4.3% and 8.3%.

#### Videos

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
`videos/track003_22_pushes.mp4` is track003_22 on that draw too: it completes
the clip with a mean error of 0.83 cm / 6.8°. `videos/track003_18_vs_22_pushes.mp4`
puts track003_18 and track003_22 under the same draw (no servo delay, 2 ticks
of camera lag) and the same pushes: track003_18 drops the pen at frame 127 and
track003_22 completes the clip. Of seeds 0–199, track003_18 drops eight and
track003_22 two: track003_22 completes seven of the eight that track003_18
drops (both drop seed 80) and drops one (seed 23) that track003_18 completes.


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
| track003_20 | +30M on track003_18 | as track003_18 with the action-rate penalty 0.5 → 1.0 (`--w-action-rate 1.0`) | 97% under pushes, 88% with randomized physics too (track003_18: 96%, 91%); on further draws level with track003_18 (98.0% against 97.2% under pushes, p = 0.29; p ≥ 0.6 with randomized physics; 90.2% against 90.1% under pushes 1.5 times as strong); shaking 0.199 → 0.191 rad/s, axis error without pushes 6.1° (5.7°), position error 0.83 cm (0.84 cm); not kept |
| track003_21 | +30M on track003_18 | as track003_18 | 96% under pushes, 92% with randomized physics too (track003_18: 96%, 91%); on further draws 95.8% against 97.2% under pushes (p = 0.06), level with randomized physics (p ≥ 0.5) and under pushes 1.5 times as strong (89.5% against 90.1%); axis error without pushes 6.2° (5.7°), position error 0.87 cm (0.84 cm), shaking 0.195 rad/s (0.199); not kept |
| track003_22 | +30M on track003_18 | as track003_18 with twice the samples per update (`--n-steps 2048 --batch-size 5120`) | 99% under pushes, 94% with randomized physics too (track003_18: 96%, 91%); on further draws 99.1% against 97.2% under pushes (p = 0.001; 99.1% against 97.0% with the 200 standard clips, p = 0.0001) and 95.9% against 90.1% under pushes 1.5 times as strong, level with randomized physics (p ≥ 0.5); axis error without pushes 5.8° (5.7°), position error 0.83 cm (0.84 cm), shaking 0.199 → 0.190 rad/s |
| track003_23 | +30M on track003_22 | as track003_22 with the samples per update doubled again (`--n-steps 4096 --batch-size 10240`) | 199 of 200 clips under pushes (track003_22: 198), 95% with randomized physics too (94%); on further draws not clearly different from track003_22 (99.5% against 99.1% under pushes, p = 0.45; 93.8% against 91.8% with randomized physics too, p = 0.08; 96.1% against 95.9% under pushes 1.5 times as strong); not clearly different from it on fresh draws either (69 drops in 2800 clips against 75, p = 0.52); axis error without pushes 5.9° (5.8°), position error 0.85 cm (0.83 cm), shaking 0.187 rad/s (0.190); not kept |
| track003_24 | +30M on track003_23 | as track003_23 | 99% under pushes, 94% with randomized physics too (track003_22: 99%, 94%); on further draws not clearly different from track003_22 (98.8% against 99.1% under pushes, p = 0.55; 93.8% against 91.8% with randomized physics too, p = 0.13; 95.6% against 95.9% under pushes 1.5 times as strong); axis error without pushes 5.8° (5.8°), position error 0.84 cm (0.83 cm), shaking 0.186 rad/s (0.190); not kept |
| track003_25 | +30M on track003_24 | as track003_24 at twice the learning rate (`--lr 1e-4`) | 95% under pushes, 93% with randomized physics too (track003_22: 99%, 94%); on further draws below track003_22 under pushes (97.4% against 99.1%, p = 0.009; 92.5% against 95.9% under pushes 1.5 times as strong, p = 0.0005) and above it with randomized physics and pushes, though not clearly (95.0% against 91.8%, p = 0.02, about 0.06 allowing for the three runs compared with track003_22); axis error without pushes 5.3° (5.8°), position error 0.86 cm (0.83 cm), shaking 0.194 rad/s (0.190); not kept |
| track003_26 | +30M on track003_25 | as track003_25 back at the learning rate 5e-5 | 94% under pushes, 94% with randomized physics too (track003_22: 99%, 94%); on further draws above track003_22 with randomized physics and pushes (95.5% against 91.8%, p = 0.008, 0.03 allowing for the four runs compared with track003_22) but below it under pushes with nominal physics (97.9% against 99.1%, p = 0.06; 97.2% against 99.1% with the 200 standard clips, p = 0.002; 94.6% against 95.9% under pushes 1.5 times as strong); on fresh draws the gain with randomized physics and pushes not borne out (51 drops in 800 clips against 46, p = 0.58) and below track003_22 under pushes with nominal physics (51 drops in 2000 against 29, p = 0.003); axis error without pushes 5.3° (5.8°), position error 0.80 cm (0.83 cm), shaking 0.186 rad/s (0.190); not kept |
| track003_27 | +30M on track003_26 | as track003_26 with the entropy bonus 0.001 → 0 (`--ent-coef 0`) | 96% under pushes, 96% with randomized physics too (track003_22: 99%, 94%); action noise barely lower (standard deviation 0.206 against 0.210); on further draws not clearly different from track003_26 (p ≥ 0.2); below track003_22 under pushes with nominal physics (98.5% against 99.1%, p = 0.23; 97.9% against 99.1% with the 200 standard clips, p = 0.008; 94.6% against 95.9% under pushes 1.5 times as strong, p = 0.14), 94.0% against 91.8% with randomized physics too (p = 0.16); axis error without pushes 5.4° (5.8°), position error 0.78 cm (0.83 cm), shaking 0.188 rad/s (0.190); not kept |
| track003_28 | +30M on track003_27 | as track003_27 with the action noise capped (`--max-std 0.15`) | 94% under pushes, 94% with randomized physics too (track003_22: 99%, 94%); action noise lower (standard deviation 0.139 against 0.206); on further draws not clearly different from track003_27 or track003_26 (p ≥ 0.11); above track003_22 with randomized physics and pushes (95.8% against 91.8%, p = 0.004, 0.02 allowing for the six runs compared with track003_22) but below it under pushes with nominal physics (98.4% against 99.1%, p = 0.21; 97.6% against 99.1% with the 200 standard clips, p = 0.006; 93.8% against 95.9% under pushes 1.5 times as strong, p = 0.006); on fresh draws the gain with randomized physics and pushes not borne out (49 drops in 800 clips against 46, p = 0.78) and below track003_22 under pushes with nominal physics (48 drops in 2000 against 29, p = 0.004); axis error without pushes 5.6° (5.8°), position error 0.79 cm (0.83 cm), shaking 0.185 rad/s (0.190); not kept |
| track003_29 | +30M on track003_24 | as track003_24 with the action noise capped and the entropy bonus off (`--max-std 0.15 --ent-coef 0`) | 98% under pushes, 96% with randomized physics too (track003_22: 99%, 94%); action noise lower (standard deviation 0.133 against track003_24's 0.179); on further draws below track003_24 under pushes 1.5 times as strong (94.0% against 95.6%, p = 0.03) and otherwise level with it (p ≥ 0.6); level with track003_22 under full pushes with nominal physics (98.9% against 99.1%, p = 0.79; 98.8% against 99.1% with the 200 standard clips, p = 0.63), above it with randomized physics and pushes, though not clearly (94.2% against 91.8%, p = 0.05, about 0.37 allowing for the seven runs compared with track003_22), and below it under pushes 1.5 times as strong (94.0% against 95.9%, p = 0.03); level with it on fresh draws (77 drops in 2800 clips against 75, p = 0.91; 43 against 46 with randomized physics and pushes); axis error without pushes 5.6° (5.8°), position error 0.83 cm (0.83 cm), shaking 0.184 rad/s (0.190); not kept |
| track003_30 | +30M on track003_29 | as track003_29 | 95% under pushes, 94% with randomized physics too (track003_22: 99%, 94%); action noise about the same (standard deviation 0.131 against track003_29's 0.133); on further draws not clearly different from track003_28 (p ≥ 0.2) and below track003_29 under pushes with nominal physics (98.4% against 98.9%, p = 0.34; 97.7% against 98.8% with the 200 standard clips, p = 0.01); above track003_22 with randomized physics and pushes, though not clearly (95.5% against 91.8%, p = 0.01, about 0.09 allowing for the eight runs compared with track003_22), but below it under pushes with nominal physics (98.4% against 99.1%, p = 0.18; 97.7% against 99.1% with the 200 standard clips, p = 0.007; 94.8% against 95.9% under pushes 1.5 times as strong, p = 0.16); on fresh draws level with it with randomized physics and pushes (46 drops in 800 clips against 46) and not clearly different over the 2800 clips (83 against 75, p = 0.43); axis error without pushes 5.8° (5.8°), position error 0.83 cm (0.83 cm), shaking 0.181 rad/s (0.190); not kept |
| track003_31 | +30M on track003_22 | as track003_22 | 97% under pushes, 94% with randomized physics too (track003_22: 99%, 94%); action noise higher (standard deviation 0.172 against 0.159); on further draws below track003_22 under pushes with nominal physics (98.2% against 99.1%, p = 0.12; 98.0% against 99.1% with the 200 standard clips, p = 0.03; 94.8% against 95.9% under pushes 1.5 times as strong, p = 0.23), level with randomized physics (92.8% against 91.8% under pushes, p = 0.52; 98.0% against 98.2% without); not clearly different from it on fresh draws (88 drops in 2800 clips against 75, p = 0.20); axis error without pushes 5.7° (5.8°), position error 0.86 cm (0.83 cm), shaking 0.200 rad/s (0.190); not kept |
| track003_32 | +30M on track003_23 | as track003_23 at half the learning rate, with the action noise capped and the entropy bonus off (`--lr 2.5e-5 --max-std 0.15 --ent-coef 0`) | 98% under pushes, 95% with randomized physics too (track003_22: 99%, 94%); action noise lower (standard deviation 0.129 against track003_23's 0.169); on further draws not clearly different from track003_23 (p ≥ 0.17); level with track003_22 under pushes with nominal physics (99.2% against 99.1%, p = 1.0; 95.8% against 95.9% under pushes 1.5 times as strong, p = 1.0), above it with randomized physics and pushes, though not clearly (94.2% against 91.8%, p = 0.05, about 0.52 allowing for the ten runs compared with track003_22); not clearly different from track003_22 on fresh draws (81 drops in 2800 clips against 75, p = 0.55), but below track003_23 there under pushes with nominal physics (37 drops in 2000 clips against 22, p = 0.01); axis error without pushes 5.9° (5.8°), position error 0.84 cm (0.83 cm), shaking 0.188 rad/s (0.190); not kept |
| track003_33 | +30M on track003_22 | as track003_32 (`--lr 2.5e-5 --max-std 0.15 --ent-coef 0 --n-steps 4096 --batch-size 10240`) | 98% under pushes, 90% with randomized physics too (track003_22: 99%, 94%); action noise lower (standard deviation 0.124 against 0.159); on further draws above track003_22 with randomized physics and pushes (95.5% against 91.8%, p = 0.003, about 0.03 allowing for the eleven runs compared with track003_22), level with it under full pushes with nominal physics (99.2% against 99.1%, p = 1.0; 99.1% against 99.1% with the 200 standard clips, p = 1.0) and below it under pushes 1.5 times as strong (94.5% against 95.9%, p = 0.04); on fresh draws the gain with randomized physics and pushes not borne out (60 drops in 800 clips against 46, p = 0.07) and below track003_22 over the 2800 clips (101 drops in 2800 clips against 75, p = 0.009); axis error without pushes 5.9° (5.8°), position error 0.85 cm (0.83 cm), shaking 0.189 rad/s (0.190); not kept |
| track003_34 | +30M on track003_22 | as track003_22 with GAE lambda 0.95 → 0.98 (`--gae-lambda 0.98`) | 94% under pushes, 92% with randomized physics too (track003_22: 99%, 94%); action noise higher (standard deviation 0.179 against 0.159); on further draws below track003_22 under pushes with nominal physics (97.4% against 99.1%, p = 0.004; 96.8% against 99.1% with the 200 standard clips, p < 0.0001; 91.9% against 95.9% under pushes 1.5 times as strong, p < 0.0001) and level with it with randomized physics (91.8% against 91.8% under pushes, p = 1.0; 98.2% against 98.2% without); below track003_22 on fresh draws too (135 drops in 2800 clips against 75, p < 0.0001); axis error without pushes 6.3° (5.8°), position error 0.87 cm (0.83 cm), shaking 0.201 rad/s (0.190); not kept |

## Usage

```bash
pip install -r requirements.txt
python train.py --name my_run --resume runs/track003_22/final_model.zip --timesteps 10_000_000 \
    --time-scale 1 --w-action-rate 0.5 --ema-alpha 0.3 --min-std 0.1 --randomize-physics --sim2real \
    --joint-vel diff --disturb 1.3 --disturb-ramp 0 0 --lr 5e-5 --gamma 0.995 --n-steps 2048 --batch-size 5120
python evaluate.py --model runs/track003_22/final_model.zip --sim2real on --joint-vel diff --random-physics 40
python evaluate.py --model runs/track003_22/final_model.zip --sim2real on --joint-vel diff --disturb 1.0 --s2r-clips 40
PY=python bash eval_suite.sh track003_22   # the standard suite, into runs/stage_eval/track003_22/
python evaluate.py --model runs/track003_8/final_model.zip --sim2real off --video out.mp4
python compare_video.py runs/track003_5/final_model.zip runs/track003_8/final_model.zip \
    --labels before after --sim2real on --seed 2 --out compare.mp4
python compare_video.py runs/track003_22/final_model.zip --labels track003_22 \
    --sim2real on --joint-vel diff --disturb 1 --seed 39 --out pushes.mp4   # as in videos/track003_22_pushes.mp4
                                       # (videos/track003_14_pushes.mp4 and track003_18_pushes.mp4 with those policies)
python compare_video.py runs/track003_18/final_model.zip runs/track003_22/final_model.zip \
    --labels track003_18 track003_22 --sim2real on --joint-vel diff --disturb 1 --seed 27 \
    --out compare_pushes.mp4   # as in videos/track003_18_vs_22_pushes.mp4
                               # (videos/track003_14_vs_18_pushes.mp4: track003_14 and track003_18, --seed 0)
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
