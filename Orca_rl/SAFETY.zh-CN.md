# 让 RL 转方块策略经过安全层运行

[English](SAFETY.md) | [中文](SAFETY.zh-CN.md) | [한국어](SAFETY.ko.md)

这里有在你自己的机器上把训练好的手内转方块策略（v1 右手，MuJoCo）接到 `Orca_safety` 后面所需的全部内容：
代码、六个训练好的模型和命令。不需要训练。

## 它做了什么

每个 10 ms 控制步，策略输出 17 个关节目标（`q_nominal`）。安全层开时，它们和实测关节角一起进入
`orca_safety_v2.filter.CBFSafetyFilter`，返回的 `q_safe` 才发给舵机，同时作为策略的「上一步目标」反馈回去。
两种模式下，每步之后都会用滤波器使用的同一组 108 对连杆测量手的自间距。

滤波器是从你安装的 `orca_safety_v2` 导入的。按下面的方式以可编辑模式安装 `Orca_safety` 后，
**你在 `Orca_safety/src/` 下做的任何修改就是被测试的版本**，没有任何拷贝。

## 安装（约 5 分钟）

Python 3.11。在包含 `Orcahand-Safe-Assembly/` 的目录下：

```bash
# 1. v1 的 URDF 和碰撞网格（已有的话跳过，之后用 --urdf 指定）
git clone https://github.com/orcahand/orcahand_description.git

# 2. 单独的环境，不动你的遥操作环境
uv venv safety_venv --python 3.11 --seed          # 或者：python3.11 -m venv safety_venv
safety_venv/bin/pip install "mujoco==3.12.0" \
    -r Orcahand-Safe-Assembly/Orca_rl/requirements.txt \
    -e "./Orcahand-Safe-Assembly/Orca_safety[geometry,safety]"

# 3. 检查
cd Orcahand-Safe-Assembly/Orca_rl
../../safety_venv/bin/python -m orca_rl.evaluate_safety --model models/run23/final_model.zip --episodes 4 --workers 4
```

- 请用环境自带的 `pip`，不要用 `uv pip`：`requirements.txt` 里 orca_sim 是从 git 以可编辑方式安装的，`uv pip` 不接受。
- **我们的数字是用 `mujoco==3.12.0` 测的**（`run22_ft_*` 两行除外，它们在另一台机器上用 3.13.0 评估）。
  用 3.12.0 时，上面的检查对这 4 个回合应当正好输出 `57%`（关）和 `11%`（开，回退 25.94%）。
  其他 MuJoCo 版本也能跑，但接触动力学略有差别，单个回合的结果会不同：3.15.0 在同样 4 个回合上是 38% 和 20%。
  我们没有测过 100 回合的数字在不同版本间会差多少，所以关/开请在同一个版本内比较。
- 以上步骤在 macOS（Apple 芯片）、Python 3.11 的空目录里从头跑通过。

URDF 会在 `Orcahand-Safe-Assembly/orcahand_description/` 和 `Orcahand-Safe-Assembly/` 的同级目录里找；
放在别处时用 `--urdf /path/to/orcahand_description/v1/models/urdf/orcahand_right.urdf`。

## 命令

在 `Orcahand-Safe-Assembly/Orca_rl/` 下，用该环境的 Python 运行（`PY=../../safety_venv/bin/python`）。

```bash
# 关 / 开对比：同一策略、同一批 100 个复位 seed、45° 目标。M4 Mac 上 8 进程：关约 1.5 分钟，开约 8 分钟。
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --goal-angle 45 --episodes 100 \
    --out my_results/run23.json

# 用你自己的安全参数：复制 Orca_safety 的 v1_right.yaml，修改后指过去
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --modes on \
    --safety-config /path/to/my_v1_right.yaml --out my_results/run23_mycfg.json

# 每个控制步一行 JSON（q_current、q_nominal、q_safe、间距、激活的连杆对、回退原因、各阶段耗时）
# -> my_logs/run23.off.jsonl 和 my_logs/run23.on.jsonl
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --episodes 8 --log my_logs/run23

# 并排对比视频：左关右开，下方的条是手指间最小间距
$PY -m orca_rl.record_safety --model models/run23/final_model.zip --seeds 0 1 --out off_on.mp4
```

常用选项：`--modes off` / `--modes on` 只跑一边，`--workers N`，`--seed N`（第一个复位 seed），
`--goal-angle`（角度越大越难；我们报告的是 30 / 45 / 60）。

## 模型（`models/`）

| 目录 | 说明 | 45° 目标成功率，安全层关 / 开 |
|---|---|---|
| `run23` | v1 策略，seed 1，3000 万步，训练时没有安全层 | 43% / 34% |
| `run22` | 同上，seed 0 | 39% / 36% |
| `run23_ft_ctrl` | run23 再训练 500 万步，仍然不带安全层 | 50% / 31% |
| `run23_ft_safety` | run23 再训练 500 万步，训练循环里带着滤波器 | 40% / 42% |
| `run22_ft_ctrl` | run22 再训练 500 万步，不带 | 45% / 35% |
| `run22_ft_safety` | run22 再训练 500 万步，带 | 39% / 36% |

每个目录里有 `final_model.zip`、`vecnormalize.pkl`（观测统计量，自动加载）和
`env_kwargs.json`（手的版本、动作模式和尺度，自动读取）。

## 我们测到了什么，哪些可能和安全层有关

详见 [`results/2026-10-12-safety-layer.md`](results/2026-10-12-safety-layer.md) 和
[`results/2026-10-13-safety-finetune.md`](results/2026-10-13-safety-finetune.md)。

- 不开滤波器时，策略有 57–74% 的步数里不同手指的连杆相互接触（MuJoCo 自己的接触检测结果一致：59% 对 56%）。开了之后是 0–0.12%。
- 滤波器在约 97% 的步数上改了目标，18–29% 的步数回退。在 4 回合的检查里，415 次回退全部是
  `nonlinear_command_margin_violation`，没有一次是 QP 失败（汇总里会打印原因）。`Orca_safety` 基准里的遥操作录制
  回退率是 0.2–0.7%，全部来自 QP 残差检查，所以这个策略是苛刻得多的输入。
- 复位姿势下 `right_middle_mp` 和 `right_ring_mp` 相距 6.2 mm，和安全层基准 `open` 场景里是同一对、同一距离，
  所以在 15 mm 的激活距离下总有约束处于激活状态。
- 滤波器不知道方块。方块会把手指推到一起，所以开着时仍有 19–30% 的步数间距小于 5 mm。
- 在这个负载上滤波器每步平均 65–80 ms（M4 Mac），约一半是有限差分梯度。`orca_rl/safety_fast.py` 用最近点
  直接算同样的梯度（23 ms → 3 ms，1857 行激活约束上最大相差 7e-7 m/rad）。它只在 `train.py --safety` 时使用；
  上面的评估运行的是 `Orca_safety/src/` 里原样的滤波器。

## 带着滤波器训练（可选，很慢）

```bash
$PY -m orca_rl.train --name my_ft_safety --version v1 --obs-rotvec --obs-fingertips --seed 1 \
    --n-envs 10 --n-steps 409 --timesteps 5000000 --curriculum-start-deg 45 \
    --resume models/run23/final_model.zip --safety
```

M4 Mac 上约 175 步/秒（500 万步约 8 小时）。输出在 `runs/my_ft_safety/`（不进 git）。
