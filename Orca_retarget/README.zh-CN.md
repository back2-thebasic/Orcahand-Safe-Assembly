# ORCA Hand 重定向改进与仿真对比

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

> 本次先更新中文版，英文和韩文版暂保留前期内容。

文档记录从官方参考配置出发，对指尖参考点、优化权重、关节限位和捏合目标逐步改进的过程。当前重点是改善拇指与其他手指的配合，并观察引入碰撞安全层后的动作变化。

## 最新对比视频

两段视频使用同一份 MediaPipe 关键点录制 `mediapipe-02.jsonl`。每组产生 807 帧有效仿真输出，并同步回放

视频录制视角从左到右：Baseline → Retarget 03 → Retarget 04

### 安全层 OFF

三组均关闭附加 Safety Layer；MuJoCo 原有碰撞和物理关节限位仍然启用。

[查看或下载对比视频](videos/retarget-comparison/comparison_Safety_OFF.mov)

https://github.com/user-attachments/assets/20ff53f4-c8f9-4d7f-b451-67a239d0fad6

### 安全层 ON

三组均开启相同的 v2 CBF-QP Safety Layer，过滤 Retarget 输出后再执行仿真。

[查看或下载对比视频](videos/retarget-comparison/comparison_Safety_ON.mov)

https://github.com/user-attachments/assets/95f271ae-0a19-46e6-8f49-8c624a4a2b04

## 修改内容

官方参考配置统一命名为 `baseline.yaml`，本目录的后续配置统一采用 `retarget_01.yaml` 至 `retarget_04.yaml`；旧文件保留，兼容前期文档。编号表示改动顺序，不代表后一个方案在所有动作上都更好

### Baseline: 官方参考配置

文件：[baseline.yaml](configs/baseline.yaml)

保留原有额外指尖偏移；`w_pos: 1.0`、`w_dir: 10.0`、`w_full_hand: 1.0`

这里是在当前代码中运行官方参考配置

### Retarget 01: 移除额外指尖偏移

文件：[retarget_01.yaml](configs/retarget_01.yaml)

在 Baseline 基础上，将五指 `fingertip_offsets_m` 设为 `[0, 0, 0]`，直接使用 URDF 指尖 frame 原点。其余配置参数保持不变。

这改变的是重定向参考点，不是电机关节零位，也不修改碰撞体。指尖参考点变化同时可能影响自动尺度校准。

### Retarget 02: 调整位置与整手形状权重

文件：[retarget_02.yaml](configs/retarget_02.yaml)

在 Retarget 01 基础上，将 `w_pos` 从 1.0 调为 2.0，将 `w_full_hand` 从 1.0 调为 0.5。


### Retarget 03: 增加相对指尖位置目标

文件：[retarget_03.yaml](configs/retarget_03.yaml)

在 Retarget 02 基础上新增：

```yaml
retarget:
  w_relative_tip: 1.0
```

配套 Python 代码增加相对指尖位置损失及解析梯度，追踪“拇指指尖到其他四个指尖”的目标向量，并沿用原有捏合激活权重。它强调两指的协同位置，但没有直接要求两个指腹面对面。

本次对比还启用了**仿真限位坐标修正**：

原 retarget 使用的关节活动范围与 MuJoCo 控制角度采用的参考坐标不一致，可能把仿真可执行的姿态排除在求解范围之外，也可能产生随后被仿真裁剪的目标角度。修正后，从 MuJoCo 的关节范围和执行器控制范围取交集，作为 retarget 求解时的有效上下界，让求解与执行使用一致的角度范围。


### Retarget 04: 增加捏合目标距离映射

文件：[retarget_04.yaml](configs/retarget_04.yaml)

保留 Retarget 03 的相对目标项，并增加：

```yaml
retarget:
  relative_tip_remap:
    enabled: true
    close_cm: 1.0
    start_cm: 5.0
    min_target_cm: 0.5
```

配套函数保持目标向量方向，调整其长度。距离使用校准、对齐后的机器人尺度厘米：大于等于 5 cm 不变，1～5 cm 线性映射至 0.5～5 cm，小于等于 1 cm 通常使用 0.5 cm；极接近零时连续过渡至零，避免方向不确定

该改动可能增强接近捏合时的闭合倾向，也可能压缩小间距表达或增加碰撞阻挡。本次对比同样启用仿真限位修正。

## Retarget 与 Teleoperation 相关代码修改

新增 YAML 参数需要对应代码支持，单独替换配置文件不能启用所有改进。以下路径相对于本地 `orca_teleop/` 项目。

### `src/orca_teleop/retargeting/adaptive_analytical.py`

- 支持读取 `fingertip_offsets_m`，允许使用零指尖偏移。
- 新增 `set_sim_joint_limits()`，让优化器使用与 MuJoCo 一致的有效关节范围。
- 新增相对指尖位置损失及梯度，由 `w_relative_tip` 控制，用于 Retarget 03。
- 新增 `remap_relative_vectors()`，根据 `relative_tip_remap` 调整目标距离，用于 Retarget 04。

Retarget 02 只调整已有的 `w_pos`、`w_full_hand` 参数，不需要新增损失项。相对指尖项和距离映射默认关闭。

### 实时 Teleoperation 接入

- `src/orca_teleop/sim_diagnostics.py`：其中的 `simulation_joint_limits()` 提取 MuJoCo 关节范围与执行器控制范围的交集。
- `scripts/teleop_sim.py`、`src/orca_teleop/sim.py`、`src/orca_teleop/pipeline.py`：通过 `--sim-joint-limits` 将上述范围传给重定向器，在开始处理人手输入前启用限位修正。
- 若使用 `--collision-safety`，仿真端调用独立的 `orca_safety_v2` 安全层，先过滤 retarget 输出，再交给 MuJoCo 执行。

选择 YAML、启用限位修正、启用安全层是三个独立设置；更换 YAML 不会自动打开后两项。

## 前期实验

前期 A／B／C 配置和 S01～S03 视频保留原始设置，不能追溯标记为已经使用本次限位修正或新增优化项。

前期配置说明、运行方法及 S01～S03 视频见[前期实验记录](README.history.zh-CN.md)。

## 当前发布范围

本次提供中文说明、配置 YAML 和两段原始 MOV 对比视频。页面提供 H.264 MP4 内嵌预览，并保留原始 MOV 下载链接。新配置依赖上述配套代码；本次更新不包含这些代码的完整发布，不能仅下载 YAML 就在未修改的上游项目中复现全部功能。

配置命名对应：`retarget_01.yaml` 对应旧 `baseline_zero_offsets.yaml`，`retarget_02.yaml` 对应旧 `last.yaml`，`retarget_03.yaml` 对应本地 `relative_tip_exp1.yaml`，`retarget_04.yaml` 对应本地 `relative_tip_exp2.yaml`。本地运行脚本仍可使用旧文件名；使用新文件名时需相应调整路径。

视频观察记录和配套代码发布信息后续补充。英文、韩文版本将在中文内容审核后同步。
