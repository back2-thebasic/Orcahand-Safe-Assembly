# Retarget / Teleoperation 代码替换

本目录提供的是现有 `orca_teleop` 项目的 **5 个完整替换文件**，不是独立项目。只需按目录结构复制，不需要手动粘贴代码片段。

## 文件范围

文件位于 [runtime/orca_teleop/](runtime/orca_teleop/)：

- `src/orca_teleop/retargeting/adaptive_analytical.py`
- `src/orca_teleop/sim_diagnostics.py`
- `scripts/teleop_sim.py`
- `src/orca_teleop/sim.py`
- `src/orca_teleop/pipeline.py`

运行基础是已有可用的 ORCA v1 右手 MediaPipe / MuJoCo 环境，包括 `orca_core`、`orca_sim`、`orcahand_description` 和 `orca_teleop` 的 mediapipe、adaptive、sim 依赖。Python 建议使用 3.11。不要把文件放进虚拟环境的 site-packages。

本次文件基于 `orca_teleop` 提交 `062ddeb8c2b182d5c9cd8bd406cb07a532c97fc8`。其余本地项目版本和文件校验值记录在 [manifest.json](runtime/manifest.json)。使用不同版本时，完整文件替换可能覆盖其他修改，应先保留备份。

## 复制文件

假设项目结构为：

```text
Orcahand/
├── Orcahand-Safe-Assembly/
├── orca_teleop/
├── orca_sim/
├── orca_core/
├── orcahand_description/
└── orca_adaptive_test/
```

在 `Orcahand/` 目录执行以下命令，自动备份已有文件并复制五个替换文件和配置：

```bash
python3 - <<'PY'
from pathlib import Path
from datetime import datetime
import json
import shutil

source = Path('Orcahand-Safe-Assembly/Orca_retarget')
target = Path('orca_teleop')
if not (target/'pyproject.toml').is_file():
    raise SystemExit('请在已有 Orcahand 工作区根目录运行')
files = json.loads((source/'runtime/manifest.json').read_text())['files']
backup = Path('retarget-backup-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
backup.mkdir()
for relative in files:
    destination = target/relative
    if destination.exists():
        saved = backup/'orca_teleop'/relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, saved)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source/'runtime/orca_teleop'/relative, destination)
for config in (source/'configs').glob('*.yaml'):
    destination = Path('orca_adaptive_test/configs')/config.name
    if destination.exists():
        saved = backup/'configs'/config.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(destination, saved)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, destination)
print('安装完成，备份目录：', backup.resolve())
PY
```

确保环境以 editable 模式导入本地 `orca_teleop`，否则复制后的代码可能不会生效。例如已有 uv 环境可执行：

```bash
cd orca_teleop
uv pip install --python .venv/bin/python --no-deps -e .
```

这条命令只更新本地代码安装，不会自动补齐基础依赖。

## 实时运行 Retarget 04

在 `orca_teleop/` 目录运行（macOS 使用 `mjpython`）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/mjpython scripts/teleop_sim.py \
  --env right --version v1 --hand right \
  --local --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config ../orca_adaptive_test/configs/retarget_04.yaml \
  --sim-joint-limits
```

Retarget 03：将配置路径改为 `retarget_03.yaml`，保留 `--sim-joint-limits`。

官方 Baseline：将配置路径改为 `baseline.yaml`，移除 `--sim-joint-limits`。Retarget 01 / 02 的配置本身不启用限位修正；需要比较修正效果时单独添加该选项。

## 发布副本的兼容处理

为保证只替换这 5 个文件也能使用原版 MediaPipe publisher，发布副本的 `pipeline.py` 仅向 publisher 传递其支持的可选参数。`teleop_sim.py` 对 `--record-video`、`--landmark-log` 明确报错，因为本次没有提供相关录制模块；正常实时 teleoperation 不需要这些选项。`--diagnostic-log` 使用已包含的 `sim_diagnostics.py`。

本次不安装或修改 safety layer。上述命令保持附加安全层 OFF；MuJoCo 的原有碰撞与物理关节限位仍启用。代码保留原安全层接入选项，但如需使用，应自行使用已有且兼容的安全层环境。
