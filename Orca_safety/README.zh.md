# OrcaHand - Safety Layer

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

Safety Layer 用于减少 OrcaHand 遥操作和控制中的手指自碰撞。在原有 retargeter 与 MuJoCo 之间加入独立Safety Layer，在满足碰撞约束和关节限制的前提下，尽量保留原始动作。v2 表示安全模块版本，当前使用的机器人模型仍是 v1 右手

## Safety Layer的效果验证实验

- [4组实际采集动作 OFF/ON 视频对比](output/video/demo_video/README.zh.md)：使用Retarget记录数据后，通过 MuJoCo 进行对比，直观展示关闭与开启 Safety Layer 时的表现
- [7组合成动作 OFF/ON 数据对比](output/comparison/README.zh.md)：直接合成运动轨迹，比较相同运动轨迹下，开启与关闭 Safety Layer 的碰撞情况、间隙违规、动作干预情况及运行耗时

## 原始 Pipeline

原 Adaptive Analytical Retargeter 根据手部关键点生成关节目标，直接交给仿真执行

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



## 加入 Safety Layer 后的 Pipeline

Retargeter 保持不变。安全层结合原始目标 `q_nominal` 和仿真测得的当前状态 `q_current`，生成最终发送的目标 `q_safe`

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




## 核心组件与功能

- **Distance Checking**：通过 Pinocchio 与 FCL 查询config中指定的关节对。只对接近的关节对计算距离梯度，判断关节运动会使它们靠近还是远离

- **CBF Safety Constraint**：定义安全裕量 $h(q)=d(q)-d_{safe}$，并对接近碰撞的关节对施加约束：

$$
\nabla d(q)^T\Delta q \geq -\eta h(q)
$$

  它限制每一步距离减小的幅度；在安全边界上，按局部线性模型，不允许继续朝会减小间隙的方向运动。默认安全距离为 5 mm，距离低于 15 mm 时激活约束，这两个距离可以在config中修改

- **QP Safety Filter**：以 $\Delta q_{nom}=q_{nominal}-q_{current}$ 为期望运动，用 OSQP 求解满足 CBF、关节角度和每步运动限制的关节增量，同时尽量减小对原动作的修改。输出为 $q_{safe}=q_{current}+\Delta q$。约束不影响动作时，输出接近原始目标

- **Nonlinear Command Validation and Fallback**：使用几何模型重新计算 QP 候选目标的碰撞距离。如果候选目标不满足裕量，先将这一步运动减半，最多缩小 8 次；每次仍检查 CBF、关节/步长限制及所有配置碰撞对的距离。找到合格的小步就执行；求解失败或缩小后仍不合格时，继续使用上一条通过验证的目标，而不直接发送原始目标


## 阅读入口

- [output/README.md](output/README.zh.md)：实验数据与视频的目录索引、文件用途和基本阅读说明
- [baseline.md](docs/baseline.zh.md)：动作链路、安全层接口、关节映射、单位与模型限制来源
- [implementation.md](docs/implementation.zh.md)：当前实现与后续改动，包括启动修复、录制与回放功能，以及验证结果和适用边界
- [config](src/orca_safety_v2/configs/v1_right.yaml)：108 对碰撞检测 link 及安全间隙、CBF、QP 和缩步复核参数


## Directory

- **`src/orca_safety_v2/`**：碰撞距离计算、CBF-QP 过滤、仿真接入、参数配置及回放状态重建
- **`scripts/`**：几何验证、动作数据生成、OFF/ON 数据实验、视频导出和 MuJoCo 交互回放
- **`tests/`**：几何、安全过滤、启动同步、仿真接入与回放行为的自动化测试
- **`docs/`**：原系统接口、当前实现、后续改动及验证边界
- **`output/`**：实验数据、动作输入、遥操作日志、摄像头原视频、OFF/ON Demo 和环境记录；详见 [目录索引](output/README.zh.md)


## Install

在 Orcahand 工作区根目录运行；

```bash
uv pip install --python orca_teleop/.venv/bin/python -e './Orca_safety[geometry,safety,test]'
```

## Teleoperation

从 `Orca_safety` 根目录切换到相邻的 `orca_teleop` 目录运行

```bash
cd ../orca_teleop
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right --version v1 --hand right --local --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config ../orca_adaptive_test/configs/baseline.yaml \
  --collision-safety
```

- Safety OFF：去掉 `--collision-safety`
- 被动记录 OFF baseline：改成 `--collision-monitor`，记录距离，但不修改指令
- Safety ON：`--collision-safety`
- 可加 `--collision-safety-config 路径`、`--collision-safety-log 新文件路径`
- 可选 `--record-video 新文件.mp4`（需要 `--local`）：录制摄像头画面及时间，不传则不录制。配合视频时间剪辑和正面 OFF/ON 回放的方法见 [scripts/README.md](scripts/README.zh.md#可选录制摄像头原视频)
- 录制后可用 `replay_viewer.py --mode OFF` 或 `--mode ON` 在 MuJoCo 窗口旋转视角、暂停和逐步查看，见 [交互式回放](scripts/README.zh.md#mujoco-交互式回放)

默认日志在 `Orca_safety/output/teleop/`，每帧 JSONL，包括实际状态、nominal/safe 目标、最小距离、活动约束、干预量、求解状态、耗时及执行后测量


### 终端提示说明

下面是Safety Layer开启时，终端实时检测会出现的提示，以下的距离示例按默认安全间隙 **5 mm** 解释；若修改配置，应以实际阈值为准

**1. 候选目标被拒绝并回退**

```text
COLLISION SAFETY | FALLBACK: nonlinear_command_margin_violation | min=5.77 mm | right_middle_mp__right_ring_mp
```

QP 候选经过几何复核后，发现碰撞或间隙不足，因此本次使用上一条通过验证的目标。`min=5.77 mm` 是执行回退目标后的实际最小距离，不是被拒绝候选的距离，所以大于 5 mm 也可能出现 FALLBACK。后面的名称表示实际最近的是中指 `mp` 链接与无名指 `mp` 链接，`__` 分隔两个名称

被拒绝候选的最小距离可在 JSONL 的 `rejected_min_distance_after` 中查看。QP 的 `solver_status=solved` 只表示求解成功，不代表候选通过后续几何复核

**2. 实际状态低于安全间隙**

```text
COLLISION SAFETY | MEASURED MARGIN VIOLATION | min=4.98 mm | right_ring_ip__right_pinky_ip
```

执行一个控制步后，无名指与小指的两个 `ip` 链接之间实际最小间隙为 4.98 mm，低于默认 5 mm。这是实际安全裕量违规，不一定已发生几何碰撞；是否碰撞应查看 JSONL 的 `measured_collision`

如果同时发生回退和实际裕量违规，终端优先显示 `FALLBACK`。因此，`FALLBACK ... | min=4.93 mm` 表示候选被拒绝，并且执行后实际状态也低于默认安全间隙


**其他提示**

- `3.COLLISION SAFETY READY`：安全过滤已启用，后面显示日志路径。`COLLISION SAFETY MONITOR` 表示仅监测，不修改目标
- `4.COLLISION SAFETY FALLBACK #100: ...`：自安全过滤器初始化或重置以来，累计回退 100 次；冒号后是原因。首次及每累计 100 次输出该计数提示
- `5.FALLBACK: QP primal infeasible`：QP 约束不可行，使用上一条通过验证的目标。其他求解异常或非法输入也会触发回退，具体原因保存在日志的 `error` 字段
- `6.Retargeter | 11.9 fps | retarget 84.22 ms`：统计时间段内 retargeter 平均处理约 11.9 帧/秒，每次调用平均耗时约 84.22 ms。这不是仿真渲染帧率，也不是完整遥操作延迟




## 复现Safety Layer测试实验

自动化测试见 [tests/README.md](tests/README.zh.md)

姿态验证、回放生成和 OFF/ON 对照命令见 [scripts/README.md](scripts/README.zh.md)
