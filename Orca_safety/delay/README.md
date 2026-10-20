# Safety Layer Latency Experiments

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

Record the computation time of each safety-layer component over three complete motions to compare subsequent optimizations.

Latest results: **After the first round of analytical-gradient optimization, the three motions average 19.43–20.79 ms/frame, with an overall speedup of 2.75–3.04×.**

## 1. Experimental Setup

- **Model**: OrcaHand v1 right hand, 17 joints and 108 collision pairs
- **Safety parameters**: 5 mm safety clearance; constraints activate at 15 mm
- **Data**: Offline Safety ON replays of three previously recorded human-motion sequences; each step advances simulation time by 10 ms

## 2. How Runtime Is Measured

Use `time.perf_counter()` to measure the elapsed runtime of safety filtering for each frame, in ms.

```text
Current-pose distance queries → Distance gradients → QP → Candidate-target distance validation → Safe target
```

- **Distance**: Distance queries for the current pose, candidate target, and backtracking candidates
- **Gradient**: The baseline uses finite differences, including repeated internal distance queries; round one uses analytical gradients from closest points, retaining finite differences for degenerate cases
- **QP**: Solver setup or update, warm start, and solve
- **Other**: Total runtime minus the three components above, including matrix construction, residual checks, fallback handling, and other operations without separate timers

Only **computation inside the Safety Layer** is measured. Policy inference, queueing, simulation stepping, post-execution measurement, logging, and hardware communication are excluded.

## 3. Pre-optimization Baseline: baseline-001

All values are **runtime per frame, in ms**. All frames are included, including backtracking and fallback frames. p95 means that approximately 95% of frames have a total runtime at or below that value.

| Motion ID | Motion | Frames | Distance mean | Gradient mean | QP mean | Other mean | Total mean | Total p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | Thumb–index pinch | 440 | 19.13 | 42.05 | 0.24 | 0.23 | **61.65** | **96.33** |
| 02 | Thumb–middle pinch | 449 | 19.84 | 37.15 | 0.23 | 0.18 | **57.40** | **103.81** |
| 03 | Thumb–index–middle pinch | 521 | 18.38 | 40.24 | 0.21 | 0.19 | **59.02** | **93.95** |

All three motions share the same main bottleneck: finite-difference gradients, followed by distance queries. QP runtime is small.

No collisions or violations of the 5 mm safety margin were detected in the sampled frames of any motion. Fallback occurred in 3/440, 1/449, and 2/521 frames, respectively.

## 4. First Optimization Round: round-001

Changes: `collision.py` reuses closest points and computes analytical distance gradients; `filter.py` calls the new gradient interface. The original meshes, collision pairs, CBF/QP parameters, and candidate-target validation remain unchanged.

Both versions were rerun three times for each motion. Means and p95 values are computed by pooling all frames from the three runs. Values below are in ms/frame, with each cell showing **before → after**. The before values come from this round's new measurements, not the historical data in Section 3.

| Motion ID | Distance mean | Gradient mean | QP mean | Total mean | Total p95 | Overall speedup |
|---|---:|---:|---:|---:|---:|---:|
| 01 | 19.07 → 19.27 | 41.88 → **0.51** | 0.24 → 0.23 | 61.41 → **20.21** | 95.41 → **30.12** | **3.04×** |
| 02 | 19.80 → 19.96 | 37.06 → **0.48** | 0.23 → 0.21 | 57.27 → **20.79** | 103.25 → **35.79** | **2.75×** |
| 03 | 18.37 → 18.53 | 40.20 → **0.51** | 0.22 → 0.21 | 58.98 → **19.43** | 93.75 → **24.43** | **3.04×** |

- **Safety results**: Both versions had zero collisions and zero 5 mm safety-margin violations across all sampled frames in the three replays
- **Fallback frames, summed over three runs**: Motion 01: 9 → 9; motion 02: 3 → 0; motion 03: 6 → 6
- **Validation**: 37 tests passed; 894 active gradient rows were compared against finite differences, with a maximum error of approximately 2.96e-11 m/rad
- **Motion differences**: A change in fallback behavior for motion 02 produced a maximum target difference of 0.254 rad and a maximum measured-state difference of 0.0286 rad, both measured as the L2 norm of the 17-joint vector. Outputs are not identical

Gradient runtime is now approximately 0.5 ms, yielding an overall speedup of roughly 3×. Distance queries are now the main bottleneck; the 10 ms control budget has not yet been met.
