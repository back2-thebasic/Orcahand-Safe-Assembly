# OrcaHand Pen Spin

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

OrcaHand v1 pen spinning: models, official demonstrations, demonstration repair, policies, and deployment.

```text
pen_spin/
├── models/
│   ├── pen/              Official pen model and textures
│   └── hand/             OrcaHand meshes
├── hardware/             Pen STEP and 3MF print files, visual markers
├── data/
│   ├── official/         40 official demonstrations (5 groups), indexes, recipes
│   └── demo_repair/      Demonstration repair pipeline
│       ├── scripts/      Scripts from initial conversion to training-data export
│       ├── assets/       Retargeting config
│       ├── base_scene.xml  Base scene for alignment
│       └── results/      Per-step results for clip_002 and clip_004
├── policies/             Trained policies
├── deploy/               Bundle export (export_bundle.py) and runtime (penspin_deploy/)
├── rl/                   Residual tracking of demonstration 003 with SB3 PPO on CPU, robust to delay and noise (rl/README.md)
├── view.py               Playback tool
└── requirements.txt      Dependencies of view.py
```

- The repaired clip_002 and clip_004 are in `data/demo_repair/results/trajectory/` and have been exported to `reorient/wuji-mjlab/src/wuji_mjlab/tasks/pen_spin/data/`.
- The pen model `models/pen/assets/pen_tracking.xml` is the original official XML. The 30 g mass is set by the loading code, not in the XML.

## Policies

All three policies are trained on clip_002 + clip_004. Each folder has `best.pt` and `log/params/` (`env.yaml`, `agent.yaml`).

| Folder | Training | Checkpoint |
|---|---|---|
| `clip002_004/` | No randomization, no disturbances, 5000 iterations | `model_4000.pt` |
| `clip002_004_dr_finetune/` | Resumed from `clip002_004` iteration 4000 for 6000 more, with randomization and disturbances | `model_10000.pt` |
| `clip002_004_dr_scratch/` (recommended) | Randomization, disturbances and curriculum, from scratch, 10000 iterations | `model_9999.pt` |

dr = domain randomization. Training settings of the dr policies:

- Randomization: joint damping 0.6–1.6×, rotor inertia 0.5–2×, joint friction 0–0.03 N·m, actuator force limit 0.6–1.1×, finger link mass and inertia, encoder bias ±0.01 rad, pen length 205–265 mm, pen diameter 14–16 mm.
- Disturbances: pushes on the pen (1–2× pen weight), kicks on the fingertips (up to 4× fingertip link weight).
- Curriculum: disturbance strength ramps from zero to full.

### Evaluation

Complete runs in the training simulation (64 per clip, 128 in total), and tracking error under nominal conditions:

| Policy | Nominal | Physics randomization | Randomization + full disturbances | Position error | Orientation error |
|---|---|---|---|---|---|
| `clip002_004` | 127/128 | 37/128 | 0/128 | 0.8–1.1 cm | 3.5–4° |
| `clip002_004_dr_finetune` | 128/128 | 123/128 | 158/256 | 1.2–1.5 cm | 5–6° |
| `clip002_004_dr_scratch` | 128/128 | 127/128 | 179/256 | 1.3–1.5 cm | 5–6° |

The full-disturbance column combines two rounds (256 runs) and adds observation noise.

## Deployment

`policies/clip002_004_dr_scratch/deploy/` is the deploy bundle of the recommended policy (ONNX policy, simulation scene, clips, parity data), used by the programs in `deploy/`. On a standalone MuJoCo twin with camera noise: 20/20 complete runs on both clips, position error 1.3–1.4 cm; actuation delay up to 20 ms has little effect, at 40 ms the pen is mostly dropped.
