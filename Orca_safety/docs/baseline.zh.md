# 原系统接口与安全层接入

[English](baseline.md) | [中文](baseline.zh.md) | [한국어](baseline.ko.md)

当前使用 **v1 右手机器人模型**；v2 是安全软件版本。实现与验证见 [implementation.md](implementation.zh.md)。

## 动作链路

```text
手部关键点 → Retargeter → OrcaJointPositions → actions_q
→ 仿真执行器 → Safety Layer（Optional）→ MuJoCo env.step
```

Retargeter 输出 17 个关节的绝对角度，单位为度。仿真执行器按关节名称排列，并转换为 rad 后交给安全层。

- `q_nominal`：原始目标
- `q_current`：从 MuJoCo 实际状态读取的关节角度
- `q_safe`：安全层返回的目标，最终交给仿真执行

安全层未接入实体手。OFF/ON 对照使用相同模型、初始状态和 nominal 输入

## 关节映射

MuJoCo 执行器顺序：

```text
wrist → thumb(mcp, abd, pip, dip)
→ index、middle、ring、pinky（abd, mcp, pip）
```

MuJoCo 与 Pinocchio 按关节名称映射；实际状态通过 `jnt_qposadr` 读取。URDF 几何计算需处理参考偏置：`q_URDF = q_MuJoCo − model.qpos0`。不能直接按数组位置复制角度

## 几何与限制来源

- **碰撞几何**：v1 URDF 的 collision mesh，由 Pinocchio/FCL 查询
- **关节范围**：MuJoCo `jnt_range` 与 `actuator_ctrlrange` 的交集
- **每步运动限制**：URDF 速度限制 × 控制周期；当前为 100 rad/s × 0.01 s，即 1 rad/步，可由 `max_step_rad` 收紧
