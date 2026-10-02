# 实验输出与回放数据

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

本目录保存 Safety Layer 的实验结果、动作输入、遥操作记录和演示视频。运行方法见 [scripts/README.md](../scripts/README.zh.md)。

## 目录

```text
output/
├── comparison/          七组合成动作的 OFF/ON 日志、汇总与图表
├── video/
│   ├── demo_video/      四组动作的 OFF/ON 演示视频
│   └── camera/          时间对齐文件（不包含摄像头原视频）
├── passive/             五种静态姿态图像、poses.json、距离与梯度验证
├── replay/              合成手部关键点经 retargeter 生成的回放输入
├── retarget-comparison/ 上述回放输入的 OFF/ON 日志与汇总
├── synthetic/           直接合成的动作输入，包括食指—中指交叉
├── teleop/              真人遥操作的关节目标、实际状态与安全日志
└── environment/         生成历史实验结果时的 Python 和依赖版本
```

## 阅读入口

- [4组动作视频 Demo](video/demo_video/README.zh.md)：直观查看 OFF/ON 的动作表现
- [7组合成动作对照](comparison/README.zh.md)：查看碰撞、间隙违规、动作干预和耗时结果
- [Retarget 回放汇总](retarget-comparison/summary.json)：合成关键点经原 retargeter 输出后的 OFF/ON 结果，不是真实采集实验

## 文件介绍

- **`.jsonl`**：逐步记录或合成动作输入。运行日志通常以 `event=start` 保存配置、关节顺序和单位，后续 `event=frame` 记录每一步。
- **`.npy` / `.npz`**：NumPy 格式的目标轨迹或关键点数据；`summary.json` 保存实验汇总。
- **摄像头 `.timing.json`**：用于把原视频中的片段时间对应到遥操作日志，配合回放脚本的 `--video-timing` 使用。

常用日志字段：

`q_nominal` 是原始目标，`q_safe` 是安全层返回目标，`q_measured` 是执行后实际状态；`min_distance_measured` 是执行后最小间隙，`fallback` 和 `error` 表示回退及原因

角度使用 rad、距离使用 m。关节数组顺序以文件中的 `joint_names` 为准。目标距离与实际运动距离不同，低于安全裕量也不等于已经碰撞。
