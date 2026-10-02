# 食指与中指交叉：合成目标动作

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

`motion.jsonl` 是 v1 右手的合成 nominal 输入，不是人手 retarget 录制，不包含实际执行结果。兼容 `replay_viewer.py` 和 `replay_video.py`，同一文件用于 OFF 和 ON。

共 600 个控制步，每步 0.01 秒，共 6 秒仿真时间：

- 0～1 秒：保持五指张开的模型 reset 姿态。
- 1～2.5 秒：食指、中指逐渐靠近并交叉。
- 2.5～3.5 秒：保持交叉目标。
- 3.5～5 秒：逐渐恢复张手。
- 5～6 秒：保持张手，给实际运动留出稳定时间。

只修改食指与中指的外展关节，其他关节目标保持不变。交叉目标是食指 abd=0.23 rad（约 13.18°）、中指 abd=−0.40 rad（约 −22.92°），采用平滑的三次插值。这些是模型中的绝对角度，已包含该模型的坐标约定，不是从零姿态计算的增量。

目标在关节范围内，但配置的食指—中指碰撞几何会相交，用于观察自碰撞安全层的干预。OFF 仍保留 MuJoCo 自身的控制范围和接触动力学，接触力可能让手指停在挤压姿态，无法完全实现交叉目标；ON 应提前阻止手指继续靠近，未完成交叉是预期行为。两指近距离观察时可用鼠标缩放和旋转。实际运动以回放结果为准。

从工作区根目录运行：

```bash
orca_teleop/.venv/bin/mjpython Orca_safety/scripts/replay_viewer.py \
  Orca_safety/output/synthetic/index-middle-crossing/motion.jsonl \
  --mode OFF --view front --speed 0.5 --paused
```

把 OFF 改为 ON 即可观察安全层效果。macOS 必须使用 mjpython 启动交互窗口。空格播放/暂停，N/B 前进/后退一步，R 回到开头；鼠标自由调整视角。无需 `--video-timing`，因为本数据没有摄像头视频；`--clip` 如需使用，按本合成轨迹的时间筛选。

Safety ON 是对合成目标重新运行当前版本的安全层，不是预先手工编排的一条避碰动作。安全层只检查配置范围和采样状态，不保证任何连续轨迹或实体手安全。本数据只用于仿真展示。
