# OrcaHand Pen Spin

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

OrcaHand v1 转笔任务：模型、官方示范、示范修复、策略和部署。

```text
pen_spin/
├── models/
│   ├── pen/              官方笔模型和纹理
│   └── hand/             OrcaHand 网格
├── hardware/             笔的 STEP、3MF 打印文件和视觉标记
├── data/
│   ├── official/         40 条官方示范（5 组）及索引、配方
│   └── demo_repair/      示范修复流程
│       ├── scripts/      从初次转换到导出训练数据的脚本
│       ├── assets/       重定向配置
│       ├── base_scene.xml  对齐用的基础场景
│       └── results/      clip_002 和 clip_004 的各步结果
├── policies/             训练好的策略
├── deploy/               部署包导出（export_bundle.py）和运行时（penspin_deploy/）
├── rl/                   基于示范 003 的残差跟踪（SB3 PPO，CPU），加入延迟和噪声训练（见 rl/README.md）
├── view.py               回放工具
└── requirements.txt      view.py 的依赖
```

- 修复后的 clip_002、clip_004 在 `data/demo_repair/results/trajectory/`，已导出到 `reorient/wuji-mjlab/src/wuji_mjlab/tasks/pen_spin/data/`。
- 笔模型 `models/pen/assets/pen_tracking.xml` 是官方原始 XML，质量 30 g 由加载代码设置，不在 XML 中。

## 策略

三个策略都在 clip_002 + clip_004 上训练。每个文件夹含 `best.pt` 和 `log/params/`（`env.yaml`、`agent.yaml`）。

| 文件夹                           | 训练方式                                     | checkpoint       |
| ----------------------------- | ---------------------------------------- | ---------------- |
| `clip002_004/`                | 无随机化、无扰动，5000 轮                          | `model_4000.pt`  |
| `clip002_004_dr_finetune/`    | 从 `clip002_004` 第 4000 轮接着训6000轮，加随机化和扰动 | `model_10000.pt` |
| `clip002_004_dr_scratch/`（推荐） | 带随机化、扰动和课程，从头训 10000 轮                   | `model_9999.pt`  |

dr = domain randomization（域随机化），dr 策略的训练设置：

- 随机化：关节阻尼 0.6～1.6 倍、转动惯量 0.5～2 倍、关节摩擦 0～0.03 N·m、电机最大力 0.6～1.1 倍、手指质量和惯量、编码器偏差 ±0.01 rad、笔长 205～265 mm、笔径 14～16 mm。
- 扰动：推笔（笔重的 1～2 倍）、碰指尖（指尖连杆重量的最多 4 倍）。
- 课程：扰动强度从零逐渐加到满强度。

### 评估

训练仿真中完整转完的次数（每段示范 64 次，共 128 次），以及标准条件下的跟踪误差：

| 策略 | 标准条件 | 物理随机化 | 随机化 + 满强度扰动 | 位置误差 | 方向误差 |
|---|---|---|---|---|---|
| `clip002_004` | 127/128 | 37/128 | 0/128 | 0.8～1.1 cm | 3.5～4° |
| `clip002_004_dr_finetune` | 128/128 | 123/128 | 158/256 | 1.2～1.5 cm | 5～6° |
| `clip002_004_dr_scratch` | 128/128 | 127/128 | 179/256 | 1.3～1.5 cm | 5～6° |

满强度扰动一栏是两轮合计 256 次，并加入了观测噪声。

## 部署

`policies/clip002_004_dr_scratch/deploy/` 是推荐策略的部署包（ONNX 策略、仿真场景、示范片段、对拍数据），由 `deploy/` 下的程序使用。在独立的 MuJoCo 仿真手上加入相机噪声运行：两段示范均 20/20 转完，位置误差 1.3～1.4 cm；执行延迟 20 ms 以内基本无影响，40 ms 时大多掉笔。
