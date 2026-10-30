# Demo videos / 演示视频

Videos 1–5: 1280×720, no text overlay. All policies run in MuJoCo (orca_sim).
视频 1–5：1280×720，画面中没有文字。全部为 MuJoCo 仿真（orca_sim）。

| File | Task · 任务 | Hand · 手 | Model | Speed · 速度 |
|---|---|---|---|---|
| `1_cube_v2.mp4` | In-hand cube reorientation, 45° goals · 手内转方块，45° 目标 | v2 | `runs/run20` | 1/2 |
| `2_cube_v1.mp4` | Same, v1 hand · 同上，v1 手 | v1 | `runs/run23` | 1/2 |
| `3_pickup.mp4` | Pick up red cube, 4 episodes · 抓起红方块，4 回合 | v1 | taskgen `pickup_reach_s0` | 1/4 |
| `4_fist.mp4` | Open → half → fist → open, 3 episodes · 张开→半握→握拳→张开，3 回合 | v1 | taskgen `fist_prec_s0` | 1/4 |
| `5_place.mp4` | Pick up cylinder and place in target, 4 episodes · 抓起圆柱放入目标区，4 回合 | v1 | taskgen `place_rel_s0` | 1/4 |
| `6_cube_v1_safety_off_on.mp4` | Cube reorientation with the Orca_safety layer off (left) and on (right), same policy and seeds · 转方块，安全层关（左）/ 开（右），同一策略同一批回合 | v1 | `runs/run23` | 1/2 |
| `7_pencil_v1.mp4` | Spin a pencil about the vertical axis, 3 episodes · 绕竖直轴转铅笔，3 回合 | v1 | `models/pencil1` | 1/2 |
| `8_pencil_compare.mp4` | Pencil spinning, four panels from the same 2 starts: no policy, pencil1, pencil2, pencil3 · 转铅笔四格对比，同两个起点：不动的手、pencil1、pencil2、pencil3 | v1 | `models/pencil1`, `pencil2`, `pencil3` | 1/2 |
| `9_pencil3_30m_vs_60m.mp4` | pencil3 after 30M (left) and 60M (right) steps, 20 s episodes · pencil3 训练 3000 万步（左）和 6000 万步（右），每回合 20 秒 | v1 | `models/pencil3`, `models/pencil3_60m` | 1/2 |
| `10_pencil4_vs_pencil5.mp4` | pencil4 (left) and pencil5 (right), 20 s episodes · pencil4（左）和 pencil5（右），每回合 20 秒 | v1 | `models/pencil4`, `models/pencil5` | 1/2 |
| `11_pencil5_seed0_vs_seed1.mp4` | pencil5 trained with seed 0 (left) and seed 1 (right), 20 s episodes · pencil5 用 seed 0（左）和 seed 1（右）训练，每回合 20 秒 | v1 | `models/pencil5`, `models/pencil5_s1` | 1/2 |
| `12_pencil5_s1_20s.mp4` | pencil5_s1 spinning the pencil continuously, one 20 s episode · pencil5_s1 连续转铅笔，一个 20 秒的回合 | v1 | `models/pencil5_s1` | 1/2 |

- Cube videos: green arrow = where the red face should point, red arrow = where it points now; the border flashes green on each solve. The 3 episodes were **selected** as the best of reset seeds 0–11, so they look better than average. Real numbers: 45° goal success 63% (v2) and 41% (v1, mean of 2 seeds), see `../results/`.
- 转方块：绿箭头 = 红面应朝的方向，红箭头 = 当前朝向；每完成一个目标边框闪绿。3 个回合是从复位 seed 0–11 中**挑出**完成目标最多的，比平均水平好看。真实水平：45° 成功率 v2 63%，v1 41%（两个 seed 平均），见 `../results/`。
- Taskgen videos use reset seeds 0–3 without selection; every episode shown succeeds. The green disc in `5_place.mp4` marks the target region (drawn in, since the scene's marker does not render).
- 组员任务的视频用复位 seed 0–3，未挑选，每个回合都成功。`5_place.mp4` 中的绿色圆盘是补画的目标区。
- The successful taskgen policies move unrealistically (they fling or toss the object); see `../results/2026-09-29-taskgen.md`.
- 组员任务的成功策略动作不真实（甩、抛物体），见 `../results/2026-09-29-taskgen.md`。

Regenerate · 重新生成: `python -m orca_rl.record_clean` (cube) and `taskgen_rl/render_clean.py` (taskgen).

- `6_cube_v1_safety_off_on.mp4`: the bar under each panel is the hand's smallest self-clearance (red = links in contact, orange = under the 5 mm margin, green = clear; white tick = 5 mm). Reset seeds 0 and 1, not selected. Numbers: `../results/2026-10-12-safety-layer.md`. Regenerate with `python -m orca_rl.record_safety`.
- `6_cube_v1_safety_off_on.mp4`：每个画面下方的条是手指间最小间距（红 = 连杆接触，橙 = 小于 5 mm，绿 = 安全；白线 = 5 mm）。复位 seed 0 和 1，未挑选。数据见 `../results/2026-10-12-safety-layer.md`。

- `7_pencil_v1.mp4`: 1280×720, no text. Reset seeds 0–2, not selected (0.22 revolutions per 10 s episode on average, against 0.29 over 100 episodes). The policy turns the pencil once, then closes the hand around it and holds still; see `../PENCIL.md`. Regenerate with `python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --seed 0 --video 7_pencil_v1.mp4 --clean`.
- `7_pencil_v1.mp4`：1280×720，无文字。复位 seed 0–2，未挑选（平均每个 10 秒的 episode 转 0.22 圈，100 个 episode 的平均是 0.29）。策略推转一下之后就握拳把铅笔夹住不动，见 `../PENCIL.zh-CN.md`。

- `8_pencil_compare.mp4`: 1280×720, four 640×360 panels with text (model on top; time and degrees turned at the bottom). Reset seeds 7 and 8, picked before recording, not by result; on these two starts every model turned 118°–154°, more than its 100-episode average (80°–105°). Each model turns the pencil once and then parks it: pencil1 in a fist, pencil2 standing in a finger gap, pencil3 flat on the open hand. See `../PENCIL.md` Section 1.2. Made with a one-off script that is not in the repo.
- `8_pencil_compare.mp4`：1280×720，四个 640×360 画面，带文字（上方是模型，下方是时间和已转角度）。复位 seed 7 和 8，录之前选定，不是按结果挑的；这两个起点上每个模型都转了 118°–154°，比各自 100 个回合的平均（80°–105°）好。每个模型都是推转一下就停：pencil1 握拳，pencil2 把铅笔立在指缝里，pencil3 平放在张开的手上。见 `../PENCIL.zh-CN.md` 1.2 节。用一次性脚本生成，没有放进仓库。
- `9_pencil3_30m_vs_60m.mp4`: 1280×360, two panels with text, 20 s episodes. Reset seed 7 (both models turn about 130°; the 60M model's pencil ends sunk in a finger gap) and seed 1240, **picked** to show the 60M model's pencil standing up at 17 s, which happens in 24 of 40 such 20 s episodes. Numbers: `../results/2026-10-27-pencil3.md`.
- `9_pencil3_30m_vs_60m.mp4`：1280×360，两个画面，带文字，每回合 20 秒。复位 seed 7（两个模型都转约 130°，6000 万步的铅笔最后沉在指缝里）和 seed 1240——**特意挑的**，用来展示 6000 万步的模型在第 17 秒把铅笔立起来；40 个 20 秒的回合里有 24 个这样结束。数据见 `../results/2026-10-27-pencil3.md`。

- `10_pencil4_vs_pencil5.mp4`: 1280×368, two panels with text, 20 s episodes; the bottom line says "stalled" while the pencil has gained under 10° of new ground in the last second. Reset seeds 7 and 8, the starts of video 8, chosen before recording. Seed 7: pencil4 turns 173°, pencil5 94° and then stands the pencil up at 2.4 s (a pencil5 failure, 24% of its episodes end early). Seed 8: pencil4 159°, pencil5 202°. Both models push once and park. Numbers: `../results/2026-10-29-pencil5.md`.
- `10_pencil4_vs_pencil5.mp4`：1280×368，两个画面，带文字，每回合 20 秒；最近 1 秒新转出不到 10° 时，底部显示「stalled」。复位 seed 7 和 8，和视频 8 的起点相同，录之前选定。seed 7：pencil4 转 173°，pencil5 转 94° 后在第 2.4 秒把铅笔立起来（pencil5 的失败例子，它 24% 的回合提前结束）。seed 8：pencil4 159°，pencil5 202°。两个模型都是推一下就停。数据见 `../results/2026-10-29-pencil5.md`。
- `11_pencil5_seed0_vs_seed1.mp4`: 1280×368, two panels with text, 20 s episodes. Reset seeds 1234 and 1235, the first two evaluation episodes, not selected. Seed 1234: seed 0 turns 209°, seed 1 1122°. Seed 1235: seed 0 turns 68° and stands the pencil up at 7.4 s, seed 1 turns 1234°. Same settings, different training seed; see `../PENCIL.md` Section 1.3.
- `11_pencil5_seed0_vs_seed1.mp4`：1280×368，两个画面，带文字，每回合 20 秒。复位 seed 1234 和 1235，即评估的前两个回合，未挑选。seed 1234：seed 0 转 209°，seed 1 转 1122°。seed 1235：seed 0 转 68° 后在第 7.4 秒立起，seed 1 转 1234°。设置相同，只是训练 seed 不同；见 `../PENCIL.zh-CN.md` 1.3 节。
- `12_pencil5_s1_20s.mp4`: 1280×720, no text. Reset seed 1234, the first evaluation episode, not selected; 3.12 revolutions in 20 s (the reward's rate cap allows 3.18). The four fingers curl and straighten in turn and walk the pencil round. Regenerate with `python -m orca_rl.pencil_eval --model models/pencil5_s1/final_model.zip --episodes 1 --max-episode-steps 2000 --video 12_pencil5_s1_20s.mp4 --clean`.
- `12_pencil5_s1_20s.mp4`：1280×720，无文字。复位 seed 1234，即评估的第一个回合，未挑选；20 秒转了 3.12 圈（奖励的速率上限最多 3.18 圈）。四指轮流弯曲、伸直，把铅笔一圈一圈往前拨。
