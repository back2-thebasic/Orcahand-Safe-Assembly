# 安装与运行

前提：已有可运行的 ORCA v1 右手 MediaPipe / MuJoCo 环境。

## 1. 替换文件

将 [runtime/orca_teleop/](runtime/orca_teleop/) 中的 **5 个 Python 文件**，按相同路径复制到本地 `orca_teleop/`，覆盖原文件。覆盖前自行备份。

将 [configs/](configs/) 中的 YAML 复制到本地 `orca_adaptive_test/configs/`。

## 2. 使用本地代码

在本地 `orca_teleop/` 目录执行（已使用 editable 安装则跳过）：

```bash
uv pip install --python .venv/bin/python --no-deps -e .
```

## 3. 启动 Teleoperation

在 `orca_teleop/` 目录运行：

```bash
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right --version v1 --hand right \
  --local --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config ../orca_adaptive_test/configs/retarget_04.yaml \
  --sim-joint-limits
```

- **Retarget 03**：将配置改为 `retarget_03.yaml`，保留 `--sim-joint-limits`。
- **官方 Baseline**：将配置改为 `baseline.yaml`，移除 `--sim-joint-limits`。
- **Retarget 01 / 02**：更换对应 YAML；限位修正由 `--sim-joint-limits` 单独控制。

上述命令不启用附加安全层。本次提供的 5 个文件不包含 safety layer 实现和录制模块，请勿添加 `--record-video` 或 `--landmark-log`。
