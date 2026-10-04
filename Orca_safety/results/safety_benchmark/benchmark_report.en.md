# Retargeting + Safety Layer Benchmark

Hardware model: v1 right hand; safety layer: orca_safety_v2

OFF and ON use the same nominal trajectory for each scenario.

ON reruns the production safety path; OFF only monitors, retaining MuJoCo target clamping and contact dynamics.

## Motion descriptions

The images below are existing static target-pose renderings from geometry validation, not screenshots of this OFF/ON dynamic execution.

### open

Hold the open-hand pose from the model reset to check whether a normal safe pose is modified.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

![open: existing static target-pose rendering](<../../output/passive/open.png>)

### fist

Set the MCP/PIP targets of the four non-thumb fingers to 1.2 rad and execute open hand → fist → open hand.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

![fist: existing static target-pose rendering](<../../output/passive/fist.png>)

### thumb_index_pinch

Use a fixed-seed optimization search to obtain a thumb–index target with an approximately 1 mm fingertip gap, gradually approach it, and return.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

![thumb_index_pinch: existing static target-pose rendering](<../../output/passive/thumb_index_pinch.png>)

### thumb_middle_pinch

Use a fixed-seed optimization search to obtain a close thumb–middle pinch target, gradually approach it, and return.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

![thumb_middle_pinch: existing static target-pose rendering](<../../output/passive/thumb_middle_pinch.png>)

### intentional_collision

Use a fixed-seed random search to obtain a target with overlapping index/middle finger geometry, gradually approach it, and return to test collision intervention.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

![intentional_collision: existing static target-pose rendering](<../../output/passive/intentional_collision.png>)

### spread

Set abduction targets for four fingers from the open-hand pose and execute open hand → spread target → open hand. Parameters come from the existing synthetic experiment.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

### assembly_like_synthetic

Manually set MCP/PIP flexion targets for four fingers to simulate grasping, then return to the open-hand pose.

Data source: [poses.json](<../../output/passive/poses.json>)

120 control frames, totaling 1.20 seconds of simulation time.

### adaptive_analytical_replay

Generate synthetic 21-point landmarks in MediaPipe format, then obtain targets through 30 calibration frames and the original AdaptiveAnalyticalRetargeter. Reuse the existing NPZ; this is not a human recording.

Data source: [retarget-replay.npz](<../../output/replay/retarget-replay.npz>)

120 control frames, totaling 1.20 seconds of simulation time.

### index_middle_crossing

Change only the index/middle abduction angles and execute open hand → approach → hold crossing target → return → open hand. Reuse the existing synthetic JSONL.

Data source: [motion.jsonl](<../../output/synthetic/index-middle-crossing/motion.jsonl>)

600 control frames, totaling 6.00 seconds of simulation time.

### retarget-session-01

Replay the original q_nominal from human teleoperation recording 1 in full. Motion: thumb_index_pinch.

Data source: [retarget-session-01.jsonl](<../../output/teleop/retarget-session-01.jsonl>)

440 control frames, totaling 4.40 seconds of simulation time.

### retarget-session-02

Replay the original q_nominal from human teleoperation recording 2 in full. Motion: thumb_middle_pinch.

Data source: [retarget-session-02.jsonl](<../../output/teleop/retarget-session-02.jsonl>)

449 control frames, totaling 4.49 seconds of simulation time.

### retarget-session-03

Replay the original q_nominal from human teleoperation recording 3 in full. Motion: thumb_index_middle_pinch.

Data source: [retarget-session-03.jsonl](<../../output/teleop/retarget-session-03.jsonl>)

521 control frames, totaling 5.21 seconds of simulation time.

## Six metrics explained

N denotes the number of control frames included for the scenario. The first three metrics assess the safety of the actual pose after execution; motion modification measures target changes; fallback and runtime measure execution costs.

### 1. Minimum actual collision distance (mm)

- **Calculation**: After each control step, measure the minimum distance over the configured collision pairs, then take the minimum over the entire run: `min(min_distance_measured) × 1000`.
- **Meaning**: The actual clearance at the most dangerous point in the motion. Compare it with the configured safety distance: for example, a 3 mm minimum distance with a 5 mm safety distance indicates insufficient clearance, but does not necessarily indicate a collision.
- **Comparison**: Check whether ON keeps clearance near or above the safety distance. Larger clearance may also result from more motion modification, so consider metrics 4 and 5 together. Distance during overlap cannot be directly interpreted as a reliable penetration depth.

### 2. Actual collision rate (%)

- **Calculation**: `number of frames with measured_collision=True / N × 100%`. Multiple colliding pairs within one frame still count as one frame.
- **Meaning**: The proportion of sampled frames with a collision during execution. `12.5% (15/120)` means 15 of 120 frames contain collisions, not 15 separate collision events.
- **Comparison**: Lower is better; focus on the reduction from OFF to ON. A value of 0% means no collisions were found in the checked control frames, but does not guarantee no collisions between control steps.

### 3. Safety-margin violation rate (%)

- **Calculation**: `number of frames with actual minimum distance strictly below configured safe_distance / N × 100%`, using `measured_margin_violation` from production logs.
- **Meaning**: Checks whether fingers enter the safety buffer. For example, a 2 mm actual gap with a configured 5 mm safety distance is already a violation even without a collision. Thus, a zero collision rate can coexist with a nonzero margin-violation rate.
- **Comparison**: Lower is better. Read this together with collision rate to judge whether the safety layer both avoids contact and preserves the required clearance.

### 4. Mean motion modification (rad)

- **Calculation**: Compute `||q_safe − q_nominal||₂` across all 17 joints for each frame, then average over N frames.
- **Meaning**: Measures how much the safety layer changes the original target. This is the difference between whole-hand joint vectors, not the average error of a single joint or the tracking error between actual pose and target.
- **Comparison**: Once safety requirements are met, smaller values indicate better preservation of the original motion. OFF does not apply safety filtering, so this value is zero, although the actual pose may still be affected by MuJoCo clamping and contact forces. Dangerous targets require modification; a larger modification alone does not imply poor ON performance. This metric does not assess motion completion either.

### 5. Fallback rate (%)

- **Calculation**: `number of frames with fallback=True / N × 100%`.
- **Meaning**: No new target passed all checks in this frame, so the last validated safe target was reused. QP errors, infeasibility, failed residual checks, or a final nonlinear-validation failure can trigger fallback. Finding an acceptable target after step reduction does not count as fallback.
- **Comparison**: Lower values generally mean less reliance on fallback. If collision rate is low but fallback rate is high, check whether safety is achieved mainly by holding an old target. Fallback is a protective mechanism, not an actual collision. OFF has zero fallback because filtering is disabled; this does not mean OFF is more reliable.

### 6. Safety-layer p95 runtime (ms)

- **Calculation**: Take the 95th percentile of the per-frame production timing `total_safety_time_ms` for ON, using NumPy's default linear interpolation.
- **Meaning**: Approximately 95% of the included frames have safety computation times at or below this value. It includes distance, gradient, QP, validation, and fallback computations inside the filter. It excludes simulation stepping, post-execution measurement, logging, retargeting, and queue waiting.
- **Comparison**: Lower is better; compare it with the control period. For example, with a 10 ms control period and a 50 ms p95, slower safety-computation frames exceed the single-period budget, so this cannot be regarded as meeting the 100 Hz real-time requirement. OFF does not run filtering and displays N/A. p95 is neither the maximum runtime nor the full teleoperation latency.

The first three metrics indicate safety benefits; motion modification and fallback indicate motion costs; p95 indicates the computation budget. This is an offline simulation: even when computation is slow, simulation advances with a fixed time step without skipping input frames.

## Experimental results

| Scenario | Mode | Minimum actual distance mm | Collision rate (frames) | Margin-violation rate (frames) | Mean modification rad | Fallback rate (frames) | Safety runtime p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| open | OFF | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000000 | 0.00%（0/120） | N/A |
| open | ON | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000000 | 0.00%（0/120） | 27.908 |
| fist | OFF | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000000 | 0.00%（0/120） | N/A |
| fist | ON | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.018923 | 0.00%（0/120） | 77.066 |
| thumb_index_pinch | OFF | 0.987 | 0.00%（0/120） | 34.17%（41/120） | 0.000000 | 0.00%（0/120） | N/A |
| thumb_index_pinch | ON | 5.726 | 0.00%（0/120） | 0.00%（0/120） | 0.039473 | 0.00%（0/120） | 42.116 |
| thumb_middle_pinch | OFF | 0.000 | 35.83%（43/120） | 43.33%（52/120） | 0.000000 | 0.00%（0/120） | N/A |
| thumb_middle_pinch | ON | 5.582 | 0.00%（0/120） | 0.00%（0/120） | 0.125778 | 0.00%（0/120） | 43.031 |
| intentional_collision | OFF | 0.000 | 46.67%（56/120） | 60.00%（72/120） | 0.000000 | 0.00%（0/120） | N/A |
| intentional_collision | ON | 5.418 | 0.00%（0/120） | 0.00%（0/120） | 0.125593 | 0.00%（0/120） | 44.884 |
| spread | OFF | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000000 | 0.00%（0/120） | N/A |
| spread | ON | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000137 | 0.00%（0/120） | 28.153 |
| assembly_like_synthetic | OFF | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000000 | 0.00%（0/120） | N/A |
| assembly_like_synthetic | ON | 6.228 | 0.00%（0/120） | 0.00%（0/120） | 0.000000 | 0.00%（0/120） | 28.311 |
| adaptive_analytical_replay | OFF | 3.227 | 0.00%（0/120） | 65.00%（78/120） | 0.000000 | 0.00%（0/120） | N/A |
| adaptive_analytical_replay | ON | 5.432 | 0.00%（0/120） | 0.00%（0/120） | 0.394195 | 0.00%（0/120） | 105.449 |
| index_middle_crossing | OFF | 0.000 | 41.00%（246/600） | 45.33%（272/600） | 0.000000 | 0.00%（0/600） | N/A |
| index_middle_crossing | ON | 5.027 | 0.00%（0/600） | 0.00%（0/600） | 0.153788 | 0.00%（0/600） | 57.403 |
| retarget-session-01 | OFF | 0.000 | 9.09%（40/440） | 60.91%（268/440） | 0.000000 | 0.00%（0/440） | N/A |
| retarget-session-01 | ON | 5.151 | 0.00%（0/440） | 0.00%（0/440） | 0.286728 | 0.68%（3/440） | 96.331 |
| retarget-session-02 | OFF | 0.000 | 15.37%（69/449） | 48.11%（216/449） | 0.000000 | 0.00%（0/449） | N/A |
| retarget-session-02 | ON | 5.801 | 0.00%（0/449） | 0.00%（0/449） | 0.364166 | 0.22%（1/449） | 103.809 |
| retarget-session-03 | OFF | 0.000 | 24.95%（130/521） | 56.81%（296/521） | 0.000000 | 0.00%（0/521） | N/A |
| retarget-session-03 | ON | 5.238 | 0.00%（0/521） | 0.00%（0/521） | 0.220639 | 0.38%（2/521） | 93.952 |

## Errors and fallback reasons

- retarget-session-01 / ON: {"QP solution fails residual check": 3}
- retarget-session-02 / ON: {"QP solution fails residual check": 1}
- retarget-session-03 / ON: {"QP solution fails residual check": 2}
