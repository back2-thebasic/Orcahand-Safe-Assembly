# 实验与回放脚本

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

先按[主 README](../README.zh.md)安装依赖。除摄像头录制示例外，以下命令均从包含 `Orca_safety` 和 `orca_teleop` 的工作区根目录运行。

## 脚本用途

- `verify_geometry.py`：静态姿态、碰撞距离和梯度验证。
- `compare_simulation.py`：相同目标轨迹的 OFF/ON 数据实验。
- `generate_retarget_replay.py`：合成关键点经过原 retargeter，生成 NPZ 回放输入。
- `generate_crossing_motion.py`：生成食指—中指交叉的 JSONL 目标输入。
- `replay_video.py`：读取真人或合成 JSONL，导出 OFF、ON 和并排 MP4。
- `replay_viewer.py`：在 MuJoCo 窗口交互查看 OFF 或 ON。

## MuJoCo 交互式回放

macOS 使用 `mjpython` 启动窗口：

```bash
orca_teleop/.venv/bin/mjpython Orca_safety/scripts/replay_viewer.py \
  Orca_safety/output/teleop/retarget-session-02.jsonl \
  --mode OFF --view front --speed 0.5 --paused
```

把 OFF 改为 ON 即可对比。空格播放/暂停，N/B 前进/后退一步，R 回到开头；鼠标左键旋转、右键平移、滚轮缩放。脚本先计算轨迹再打开窗口，ON 准备阶段可能需要等待。

回放[合成交叉动作](../output/synthetic/index-middle-crossing/README.zh.md)时，将输入换为 `Orca_safety/output/synthetic/index-middle-crossing/motion.jsonl`，无需摄像头时间文件。

## 导出视频

```bash
orca_teleop/.venv/bin/python Orca_safety/scripts/replay_video.py \
  Orca_safety/output/teleop/retarget-session-02.jsonl \
  --video-timing Orca_safety/output/video/camera/session-02.timing.json \
  --clip 7:14 --view front --speed 0.5 \
  --output Orca_safety/output/video/my-key-actions
```

输出 `OFF.mp4`、`ON.mp4`、`OFF-ON.mp4`。以上片段只是示例，按原视频选择实际范围；导出完整记录时去掉 `--clip`。

交互与视频回放共用以下参数：

- `--clip 起点:终点`：可重复指定。配合 `--video-timing` 使用摄像头视频秒数；否则使用首条帧日志起算的现实秒数，合成输入使用轨迹时间。片段之前的仿真历史仍会计算。
- `--speed 0.5`：按仿真时间半速播放，不改变控制步长。ON 重新计算安全目标，不直接播放日志中的 `q_safe`。
- `--view front`：掌面正视角；可用 `--azimuth`、`--elevation` 调整方向。`--camera-distance` 越大越远，当前 v1 默认约 0.45 m。

## 可选录制摄像头原视频

在 `orca_teleop` 目录运行遥操作时，给 `teleop_sim.py --local` 的原命令添加：

```bash
--collision-safety-log ../Orca_safety/output/teleop/retarget-session-04.jsonl \
--record-video ../Orca_safety/output/video/camera/session-04.mp4
```

不传 `--record-video` 就不录制，与 `--show-video` 独立。正常按 Ctrl+C 结束；同时生成 `.timing.json`，供 `--video-timing` 对齐片段时间。视频与执行后日志是近似时间对齐，包含识别和队列延迟。

## 数据实验与输入生成

```bash
orca_teleop/.venv/bin/python Orca_safety/scripts/verify_geometry.py \
  --output Orca_safety/output/my-passive --render
orca_teleop/.venv/bin/python Orca_safety/scripts/compare_simulation.py \
  --steps 120 --poses Orca_safety/output/my-passive/poses.json \
  --output Orca_safety/output/my-comparison
orca_teleop/.venv/bin/python Orca_safety/scripts/generate_retarget_replay.py \
  --steps 120 --output Orca_safety/output/replay/my-replay.npz
orca_teleop/.venv/bin/python Orca_safety/scripts/compare_simulation.py \
  --replay Orca_safety/output/replay/my-replay.npz \
  --output Orca_safety/output/my-retarget-comparison
orca_teleop/.venv/bin/python Orca_safety/scripts/generate_crossing_motion.py \
  --output Orca_safety/output/synthetic/my-crossing
```

120 个控制步对应 1.2 秒仿真时间。对照、视频和交叉动作生成请使用新输出目录；静态验证同名文件会覆盖。macOS 渲染需要图形权限；离线实验无需摄像头或实体手。

实验输出见 [output/README.md](../output/README.zh.md)，代码测试见 [tests/README.md](../tests/README.zh.md)，验证边界见 [implementation.md](../docs/implementation.md)。
