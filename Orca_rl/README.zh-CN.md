# Orca_rl：ORCA 手内转方块（PPO）

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

ORCA 手掌心朝上、方块放在掌心，策略要把方块的红色面转到指定朝向，并在 15° 以内保持 10 步。
每次成功后换一个新目标，episode 继续。训练用的是 Stable-Baselines3 的 PPO，仿真用上游的 [orca_sim](https://github.com/orcahand/orca_sim)（未修改）。

## 目前的结果

目标成功率（解开的目标数 / 目标尝试数）。难度固定，每个难度 100 个 episode，两个训练 seed。

| 手 | 配置 | 30° | 45° | 60° | 训练中课程最终难度 |
|---|---|---|---|---|---|
| v2 | 基线（run10 / run12） | 49% / 53% | 39% / 37% | 28% / 25% | 42° / 43° |
| v2 | + 旋转向量 + 指尖与接触（run20 / run21） | 74% / 77% | 65% / 62% | 54% / 52% | 82° / 79° |
| **v1** | 基线（run24，只有 seed 0） | 29% | 22% | 15% | 27° |
| **v1** | + 旋转向量 + 指尖与接触（run22 / run23） | 51% / 55% | 39% / 43% | 29% / 31% | 47° / 50° |

- 在 v2 上，加入「该往哪转、还差多少」的旋转向量和指尖观测后，每个难度的成功率都提高了约 25–30 个百分点。
- **实体手是 v1**。同样的配置在 v1 上明显更差，最可能的原因是 v1 仿真场景里方块离手指太远（见 `results/2026-09-27-run22-24-v1.md`），需要按实物摆放调整场景。
- 每组实验的完整记录在 `results/`，按时间记下的问题复盘在 `orca_rl/README.md`。

## 目录

```
Orca_rl/
├── orca_rl/
│   ├── task.py           # 环境：奖励、成功判定、课程、观测、动作映射
│   ├── train.py          # PPO 训练
│   ├── evaluate.py       # 固定难度评估（看这个，不要看训练曲线）
│   ├── diagnose.py       # 把失败拆成「够不到」和「守不住」
│   ├── checks.py         # 奖励有没有漏洞可钻；改奖励后先跑它
│   ├── record.py         # 录单个策略的视频
│   ├── compare_video.py  # 多个策略在同样的 seed 下并排对比
│   └── README.md         # 复盘（run1 至今踩过的坑）
├── results/              # 每组实验的记录
└── requirements.txt
```

## 安装

Python 3.11。在 `Orca_rl/` 目录下：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m orca_rl.checks          # 应该显示 all checks passed
```

## 常用命令

所有命令都在 `Orca_rl/` 目录下运行。输出在 `runs/<name>/`（不进 git）。

```bash
# 训练：v1 手，目前最好的配置
python -m orca_rl.train --name v1_s0 --version v1 --seed 0 --obs-rotvec --obs-fingertips \
    --timesteps 30_000_000 --n-envs 14 --n-steps 292

# 评估：固定难度，100 个 episode
python -m orca_rl.evaluate --model runs/v1_s0/final_model.zip --goal-angle 45 --episodes 100

# 诊断：失败是够不到还是守不住
python -m orca_rl.diagnose --model runs/v1_s0/final_model.zip --goal-angle 45 --episodes 60

# 视频
python -m orca_rl.record --model runs/v1_s0/final_model.zip --goal-angle 45 --out v1_s0.mp4
python -m orca_rl.compare_video --goal-angle 60 --out videos/compare.mp4 \
    --run runs/a "config A" --run runs/b "config B"
```

- M4 Mac 上约 6000 fps，3000 万步约 1.5 小时；i9-11900F（WSL）上约 2500 fps，约 3.5 小时。用 `--device cpu`，网络很小，GPU 反而更慢。
- 训练时的环境设置会写进 `runs/<name>/env_kwargs.json`，评估、诊断和录视频会自动读取，手的版本和动作尺度不会用错。
- 训练中途用 Ctrl-C 或 `kill` 停止，会先保存模型；从 checkpoint 续训用 `--resume runs/<name>/checkpoints/ppo_<N>_steps.zip`，`--timesteps` 表示再训练多少步。

## 动作与控制约定

- 控制周期 **10 ms**：MuJoCo 步长 2 ms，每个动作执行 5 步（orca_sim 默认值）。
- 动作 17 维，范围 [−1, 1]，**relative** 模式：每步在上一步的伺服目标上加 `0.15 × a × 半程`（半程 = 该关节活动范围的一半），再裁剪到关节范围内。
  在 v1 上，伺服目标每 10 ms 最多移动 4.7–9.9°。action_scale 试过 0.06 和 0.3，都比 0.15 差。
- **v1 和 v2 的执行器顺序不同**（v1：手腕、拇指 ×4、食指、中指、无名指、小指；v2：手腕、小指、无名指、中指、食指、拇指），策略不能混用。

## 上实体手之前还没解决的

- 观测里用到了仿真才有的信息：方块位姿和角速度、红面朝向、每根手指的接触标志。实物上需要相机或动捕，以及触觉或估计。
- 还没有做域随机化（`--randomize-physics` 已实现，没训练过）。
- v1 仿真场景（方块尺寸、质量、摆放位置、手的姿态）需要和实物对齐。
