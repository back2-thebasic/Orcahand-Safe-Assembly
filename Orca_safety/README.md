# OrcaHand - Safety Layer

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

The Safety Layer reduces finger self-collision during OrcaHand teleoperation and control. It sits between the existing retargeter and MuJoCo, preserving the original command as much as possible while enforcing collision constraints and joint limits. **v2 refers to the safety software; the robot model is the v1 right hand.**

## Safety Layer Evaluation

- [Four OFF/ON motion video comparisons](output/video/demo_video/README.md): visualize behavior in MuJoCo using recorded retargeting inputs and a synthetic motion.
- [Seven synthetic-motion OFF/ON data comparisons](output/comparison/README.md): compare collisions, clearance violations, interventions, and runtime for identical target trajectories.

## Original Pipeline

The Adaptive Analytical Retargeter generates joint targets from hand landmarks and sends them directly to the simulation.

```text
Human Hand
    ↓
Hand Tracking
    ↓
Adaptive Analytical Retargeting
    ↓
q_nominal
    ↓
OrcaHand / MuJoCo
```

## Pipeline with the Safety Layer

The retargeter is unchanged. The layer combines the nominal target `q_nominal` with the measured simulation state `q_current` to produce the dispatched target `q_safe`.

```text
Human Hand
    ↓
Hand Tracking
    ↓
Adaptive Analytical Retargeting
    ↓
q_nominal
    ↓
┌─────────────────────────┐
│ Collision Safety Layer  │
│-------------------------│
│ Distance Checking       │
│      ↓                  │
│ CBF Safety Constraint   │
│      ↓                  │
│ QP Safety Filter        │
└─────────────────────────┘
    ↓
q_safe
    ↓
MuJoCo
    ↓
Real OrcaHand
```

The hardware path shown in the diagram is a future extension; the current safety integration is limited to MuJoCo simulation.

## Core Components

- **Distance Checking**: Pinocchio/FCL queries the configured collision link pairs. Distance gradients are computed only for nearby pairs to determine whether joint motion brings them closer together or farther apart.
- **CBF Safety Constraint**: define the clearance margin $h(q)=d(q)-d_{safe}$ and constrain nearby pairs:    $$
  \nabla d(q)^T\Delta q \geq -\eta h(q)
  $$    This limits the reduction in distance at each step. At the safety boundary, the local linear model disallows motion that further reduces clearance. The default safe distance is 5 mm; constraints activate below 15 mm. Both are configurable.

- **QP Safety Filter**: OSQP solves for a joint increment close to $\Delta q_{nom}=q_{nominal}-q_{current}$ while satisfying CBF, joint-angle, and per-step motion limits. The output is $q_{safe}=q_{current}+\Delta q$. With no binding constraints, it remains close to the nominal target.
- **Nonlinear Command Validation and Fallback**: recompute collision distances at the candidate target. If its clearance is insufficient, halve the motion up to eight times, checking CBF, joint/step limits, and all configured collision pairs each time. Execute a valid smaller step if available. If solving fails or no valid candidate is found, reuse the last validated target instead of forwarding the nominal command.

## Reading Guide

- [output/README.md](output/README.md): output directory index, file purposes, and reading notes.
- [baseline.md](docs/baseline.md): action pipeline, safety interfaces, joint mapping, units, and model limits.
- [implementation.md](docs/implementation.md): implementation, startup fixes, recording/replay features, validation, and limitations.
- [Configuration](src/orca_safety_v2/configs/v1_right.yaml): 108 collision link pairs and clearance, CBF, QP, and backtracking parameters.

## Directory

- **`src/orca_safety_v2/`**: collision distances, CBF-QP filtering, simulation integration, configuration, and replay state reconstruction.
- **`scripts/`**: geometry verification, motion generation, OFF/ON data experiments, video export, and interactive MuJoCo replay.
- **`tests/`**: geometry, filtering, startup synchronization, integration, and replay tests.
- **`docs/`**: original interfaces, current implementation, subsequent changes, and validation scope.
- **`output/`**: experimental data, motion inputs, teleoperation logs, webcam recordings, OFF/ON demos, and environment snapshots; see the [index](output/README.md).

## Install

Run from the Orcahand workspace root:

```bash
uv pip install --python orca_teleop/.venv/bin/python -e './Orca_safety[geometry,safety,test]'
```

## Teleoperation

Starting in `Orca_safety`, switch to the sibling `orca_teleop` directory:

```bash
cd ../orca_teleop
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right --version v1 --hand right --local --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config ../orca_adaptive_test/configs/baseline.yaml \
  --collision-safety
```

- Safety OFF: remove `--collision-safety`.
- Passive OFF baseline: use `--collision-monitor` to record distances without modifying commands.
- Safety ON: use `--collision-safety`.
- Optional: `--collision-safety-config PATH` and `--collision-safety-log NEW_PATH`.
- Optional `--record-video NEW_FILE.mp4` (requires `--local`): record the webcam feed and timing. Without it, no video is recorded. See [recording and clip selection](scripts/README.md#optional-webcam-recording).
- Use `replay_viewer.py --mode OFF` or `--mode ON` to rotate the view, pause, and inspect individual steps; see [interactive replay](scripts/README.md#interactive-mujoco-replay).

Default logs are stored in `Orca_safety/output/teleop/`. Each JSONL frame includes measured state, nominal/safe targets, minimum clearance, active constraints, interventions, solver status, runtime, and post-step measurements.

### Terminal Messages

The examples below assume the default **5 mm** safety clearance. Use your configured threshold if it differs.

**1. Candidate rejected; fallback applied**

```text
COLLISION SAFETY | FALLBACK: nonlinear_command_margin_violation | min=5.77 mm | right_middle_mp__right_ring_mp
```

Geometric validation found a collision or insufficient clearance, so the previous validated target was reused. `min=5.77 mm` is the actual minimum distance after executing the fallback, not the rejected candidate's distance. A fallback can therefore occur even when the displayed distance exceeds 5 mm. The pair identifies the middle-finger `mp` link and ring-finger `mp` link; `__` separates the names.

The rejected candidate's minimum distance is available as `rejected_min_distance_after` in JSONL. `solver_status=solved` only indicates QP success, not successful nonlinear validation.

**2. Measured state below the safety clearance**

```text
COLLISION SAFETY | MEASURED MARGIN VIOLATION | min=4.98 mm | right_ring_ip__right_pinky_ip
```

After a control step, the actual minimum gap between the ring- and pinky-finger `ip` links is 4.98 mm, below the default 5 mm. This is a margin violation, not necessarily a geometric collision; check `measured_collision` in JSONL.

When both fallback and a measured margin violation occur, the terminal prioritizes `FALLBACK`. Thus, `FALLBACK ... | min=4.93 mm` indicates both candidate rejection and insufficient post-step clearance.

**Other messages**

- `COLLISION SAFETY READY`: filtering is enabled; the log path follows. `COLLISION SAFETY MONITOR` means monitoring only.
- `COLLISION SAFETY FALLBACK #100: ...`: 100 accumulated fallbacks since initialization/reset. The reason follows the colon; this counter is printed on the first fallback and every 100 fallbacks.
- `FALLBACK: QP primal infeasible`: QP constraints are infeasible; reuse the previous validated target. Solver errors or invalid inputs also trigger fallback, with details in `error`.
- `Retargeter | 11.9 fps | retarget 84.22 ms`: average processing rate and call duration during the reporting interval. These are neither rendering FPS nor full teleoperation latency.

## Reproducing Safety Layer Tests

See [tests/README.md](tests/README.md) for automated tests and [scripts/README.md](scripts/README.md) for pose verification, input generation, and OFF/ON experiments.
