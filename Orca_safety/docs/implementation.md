# Current Implementation and Validation

[English](implementation.md) | [中文](implementation.zh.md) | [한국어](implementation.ko.md)

The safety layer targets the v1 right-hand MuJoCo simulation. See [baseline.md](baseline.md) for interface conventions, and the [main README](../README.md) and [scripts README](../scripts/README.md) for installation and usage.

## Safety Layer Implementation

- **Collision distances and CBF**: Pinocchio/FCL checks 108 configured link pairs. The default safety clearance is 5 mm; constraints activate below 15 mm. A conservative AABB lower bound screens distant pairs, and distance gradients are computed for nearby pairs.
- **QP filtering**: OSQP preserves the original target as much as possible while satisfying CBF constraints, joint ranges, and per-step motion limits.
- **Nonlinear Command Validation and Fallback**: candidate distances are checked again. If invalid, the motion is halved up to 8 times, with all constraints checked at each attempt. If no valid candidate is found or the solver fails, the last validated target is reused.
- **Optional modes and logging**: `--collision-safety` enables filtering; `--collision-monitor` only monitors. JSONL logs record inputs, outputs, actual states, distances, and interventions.

## Subsequent Changes

- **Startup synchronization fix**: while waiting for retargeter calibration, the simulation holds its actual MuJoCo reset pose, avoiding a mismatch with the core neutral pose. Progressively smaller candidate motions reduce repeated validation failures that previously kept the old target indefinitely. Set `nonlinear_backtracking_steps=0` to disable these attempts.
- **Optional webcam recording**: `--record-video` saves the human-hand video and timing metadata, allowing replay clips to be selected using original-video timestamps.
- **Video comparisons**: `replay_video.py` generates OFF, ON, and side-by-side videos from the same nominal inputs, with clipping, camera controls, and slow playback.
- **Interactive replay**: `replay_viewer.py` displays OFF/ON separately in MuJoCo, with a free camera, pause, and frame stepping. Trajectories are computed before playback; display speed does not change the safety control step.

## Validation and Demonstrations

- **Automated tests**: the latest recorded run passed 26 tests covering geometry, QP, fallback, startup synchronization, and replay history. See the [tests README](../tests/README.md) for commands.
- **Static geometry checks**: [passive/](../output/passive/) contains pose, distance, and gradient checks.
- **Historical experiments**: [seven synthetic-motion comparisons](../output/comparison/README.md) and the [retarget replay summary](../output/retarget-comparison/summary.json) contain OFF/ON results.
- **Visual demonstrations**: [four demos](../output/video/demo_video/README.md) include recorded human motions and the synthetic index–middle crossing motion.

## Current Limitations

- Only configured collision pairs are covered. Within-finger pairs and some structural contacts are excluded; the 5 mm clearance also restricts fingertip contact.
- Target validation and control-step sampling do not guarantee safety throughout a continuous trajectory. Actual dynamic motion can still overshoot.
- Mesh distance queries, gradient calculations, and backtracking add latency. Stable real-time teleoperation performance has not been demonstrated.
- OFF retains MuJoCo's own control limits and contact forces, so crossing targets may produce contact and compression. ON preventing completion of a dangerous target is an expected intervention.
- Results apply only to simulation; physical-hand safety has not been validated. Passing the safety module's tests does not imply that all tests in other projects pass.
