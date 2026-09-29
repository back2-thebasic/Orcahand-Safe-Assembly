# Demo videos / 演示视频

1280×720, no text overlay. All policies run in MuJoCo (orca_sim).
1280×720，画面中没有文字。全部为 MuJoCo 仿真（orca_sim）。

| File | Task · 任务 | Hand · 手 | Model | Speed · 速度 |
|---|---|---|---|---|
| `1_cube_v2.mp4` | In-hand cube reorientation, 45° goals · 手内转方块，45° 目标 | v2 | `runs/run20` | 1/2 |
| `2_cube_v1.mp4` | Same, v1 hand · 同上，v1 手 | v1 | `runs/run23` | 1/2 |
| `3_pickup.mp4` | Pick up red cube, 4 episodes · 抓起红方块，4 回合 | v1 | taskgen `pickup_reach_s0` | 1/4 |
| `4_fist.mp4` | Open → half → fist → open, 3 episodes · 张开→半握→握拳→张开，3 回合 | v1 | taskgen `fist_prec_s0` | 1/4 |
| `5_place.mp4` | Pick up cylinder and place in target, 4 episodes · 抓起圆柱放入目标区，4 回合 | v1 | taskgen `place_rel_s0` | 1/4 |

- Cube videos: green arrow = where the red face should point, red arrow = where it points now; the border flashes green on each solve. The 3 episodes were **selected** as the best of reset seeds 0–11, so they look better than average. Real numbers: 45° goal success 63% (v2) and 41% (v1, mean of 2 seeds), see `../results/`.
- 转方块：绿箭头 = 红面应朝的方向，红箭头 = 当前朝向；每完成一个目标边框闪绿。3 个回合是从复位 seed 0–11 中**挑出**完成目标最多的，比平均水平好看。真实水平：45° 成功率 v2 63%，v1 41%（两个 seed 平均），见 `../results/`。
- Taskgen videos use reset seeds 0–3 without selection; every episode shown succeeds. The green disc in `5_place.mp4` marks the target region (drawn in, since the scene's marker does not render).
- 组员任务的视频用复位 seed 0–3，未挑选，每个回合都成功。`5_place.mp4` 中的绿色圆盘是补画的目标区。
- The successful taskgen policies move unrealistically (they fling or toss the object); see `../results/2026-09-29-taskgen.md`.
- 组员任务的成功策略动作不真实（甩、抛物体），见 `../results/2026-09-29-taskgen.md`。

Regenerate · 重新生成: `python -m orca_rl.record_clean` (cube) and `taskgen_rl/render_clean.py` (taskgen).
