# Orca_rl：用 ORCA v1 手转铅笔（PPO）

[English](PENCIL.md) | [中文](PENCIL.zh-CN.md) | [한국어](PENCIL.ko.md)

ORCA v1 右手（实体手的版本）掌心朝上，一支铅笔横放在手掌和手指上。
任务是让铅笔绕竖直轴一直转下去（从上往下看逆时针），而且不能掉。没有目标，也没有成功奖励：
奖励就是铅笔实际转过的偏航角，所以一个 episode 的回报大约等于它转了多少弧度。

本文档介绍场景、**奖励函数的详细设计**、观测和动作、所有环境参数和训练参数、每个函数的作用，以及目前的结果。

**当前状态（2026-10-15）：任务能学，但策略还不会连续转。** 训练好的模型 `pencil1` 在每个 episode 开头 2.5 秒
把铅笔推转约 90°，然后手握成拳把铅笔夹死，之后不再动。修正方案（`--pose-penalty`）已经实现并通过检查，还没有训练。

演示：[`demo_videos/7_pencil_v1.mp4`](demo_videos/7_pencil_v1.mp4)（pencil1，复位 seed 0–2，未挑选，半速）。

## 目录

1. [结果](#1-结果)
2. [快速上手](#2-快速上手)
3. [文件](#3-文件)
4. [场景和 episode](#4-场景和-episode)
5. [奖励函数](#5-奖励函数)
6. [观测](#6-观测)
7. [动作和控制](#7-动作和控制)
8. [环境参数](#8-环境参数)
9. [训练：PPO 和记录的指标](#9-训练ppo-和记录的指标)
10. [函数说明](#10-函数说明)
11. [已知问题和下一步](#11-已知问题和下一步)

## 1. 结果

确定性策略，每个 episode 10 秒，评估 seed 1234，MuJoCo 3.12.0。

| 策略 | 每个 episode 的圈数 | 平均转速 | 掉落率 |
|---|---|---|---|
| 什么都不做（30 个 episode） | 0.00 | 0.1°/s | 10% |
| `pencil_smoke`，300 万步（50 个 episode） | 0.13 | 4.7°/s | 4% |
| `pencil1` checkpoint 500 万（50 个 episode） | 0.10 | 3.9°/s | 8% |
| `pencil1` checkpoint 1000 万 | 0.16 | 6.0°/s | 12% |
| `pencil1` checkpoint 1500 万 | 0.25 | 9.9°/s | 28% |
| `pencil1` checkpoint 2000 万 | 0.30 | 11.5°/s | 8% |
| `pencil1` checkpoint 2500 万 | 0.33 | 13.3°/s | 28% |
| **`pencil1` 最终模型，3000 万步（100 个 episode）** | **0.29**（最多 0.55） | 11.9°/s | 14% |
| 参考：脚本 oracle，按 1 rad/s 上限转 | 1.59 | 57°/s | 0% |

「圈数」是铅笔轴线在指定方向上转过的净偏航角，只在铅笔被握住时计入（5.2 节）。平均转速的分母是全部 episode 时间，掉落的 episode 也算在内。

**只有每个 episode 的开头在转。** 把 pencil1 的 episode 按 2.5 秒分成四段（只统计没掉笔的 episode）：

| 评估 seed | 0–2.5 s | 2.5–5 s | 5–7.5 s | 7.5–10 s | 后 5 秒仍在转（>10°） |
|---|---|---|---|---|---|
| 1234（60 个里 48 个） | 82° | 19° | 4° | 1° | 10% |
| 0（60 个里 54 个） | 93° | 18° | 3° | 0° | 9% |

**停下来时手在干什么**（30 个 episode 的平均值）：

| 时刻 | 四指平均弯曲 | 手腕 | 伺服目标顶在关节限位上 | 碰到铅笔的手指数 | 铅笔速度 |
|---|---|---|---|---|---|
| 1.0 s | 0.00 rad | −0.10 | 28% | 1.6 | 128 mm/s |
| 2.5 s | 0.39 rad | −0.40 | 41% | 1.8 | 55 mm/s |
| 10 s | 1.04 rad | −0.60 | 75% | 2.6 | 12 mm/s |

手一开始是平的（弯曲 0，手腕 −0.30）。推完一下之后，它握成拳把铅笔夹住。要继续转就得松开重新抓，
这有掉笔的风险（扣 2 分，并且失去剩下的时间）；夹死虽然 0 分，但不会掉。方块任务也遇到过同样的问题
（「夹住不动」，`orca_rl/README.md` 复盘一的错误 2）。完整记录：`results/2026-10-14-pencil1.md` 和
`results/2026-10-14-pencil-spin-smoke.md`。

pencil1 在 M4 Mac 上训练了 1 小时 41 分钟（10 个环境，约 5000 步/秒）。

## 2. 快速上手

Python 3.11，在 `Orca_rl/` 目录下，使用**训练环境**（`env/requirements-train.lock.txt`，见 `env/README.md`）。
不需要修改 orca_sim：铅笔场景随 `orca_rl` 一起提供（10.1 节）。

```bash
# 1. 检查奖励（约 1 分钟），最后必须输出 "all checks passed"
python -m orca_rl.pencil_checks

# 2. 评估训练好的模型（100 个 episode 应复现 0.29 圈 / 14% 掉落 / 回报 1.27 +- 1.23）
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 100

# 3. 基线
python -m orca_rl.pencil_eval --policy zero --episodes 30
python -m orca_rl.pencil_eval --policy random --episodes 10

# 4. 视频：带文字（640x480），或者不带文字的演示画面（1280x720）
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --video pencil1.mp4
python -m orca_rl.pencil_eval --model models/pencil1/final_model.zip --episodes 3 --seed 0 \
    --video 7_pencil_v1.mp4 --clean

# 5. 训练（pencil1 的设置；M4 Mac 上约 1.5–2.5 小时）
python -m orca_rl.pencil_train --name pencil1 --timesteps 30_000_000 --n-envs 10

# 6. 下一个实验：同上，加上姿态项（5.6 节）
python -m orca_rl.pencil_train --name pencil2 --pose-penalty 5e-4 --timesteps 30_000_000 --n-envs 10

tensorboard --logdir runs/
```

- 产物在 `runs/<name>/`：`final_model.zip`、`vecnormalize.pkl`、`env_kwargs.json`、`checkpoints/`（每 100 万步一个）、`tb/`。
  评估脚本会读取 `env_kwargs.json`，所以模型的奖励和动作设置不会弄混。
- 用 Ctrl-C 或 `kill <pid>` 停止训练时会先保存模型。`--resume runs/<name>/checkpoints/ppo_<N>_steps.zip` 可以接着训练，此时 `--timesteps` 是追加的步数。
- 训练请用训练环境。在安全层环境（`safety_venv`，numpy 2.3.5）里，5 次多进程启动有 2 次在子进程导入 numpy 时崩溃
  （`RuntimeError: Could not parse python long as longdouble`），这时任务代码还没运行；训练环境（numpy 2.4.6）从来没出过这个问题。
  评估和检查在两个环境里都能正常运行。

## 3. 文件

```
Orca_rl/
├── PENCIL.md / PENCIL.zh-CN.md / PENCIL.ko.md   # 本文档
├── orca_rl/
│   ├── pencil.py          # 任务：PencilSpin（场景加载、奖励、观测、复位）
│   ├── pencil_checks.py   # 用脚本策略证明奖励没有漏洞
│   ├── pencil_train.py    # PPO 训练（复用 train.py 的回调和向量化）
│   ├── pencil_eval.py     # 评估和录视频
│   ├── scenes/
│   │   └── scene_right_pencil_spin_v1.xml   # v1 手 + 铅笔（模板，见 10.1 节）
│   └── train.py           # build_vec_env / make_env_fn 多了一个 env_cls 参数
├── models/pencil1/        # final_model.zip、vecnormalize.pkl、env_kwargs.json
├── results/2026-10-14-pencil-spin-smoke.md、2026-10-14-pencil1.md
└── demo_videos/7_pencil_v1.mp4
```

任务复用了方块任务（`orca_rl/task.py`）的手、动作映射、指尖识别和物理随机化，以及方块任务调好的 PPO 超参数。

## 4. 场景和 episode

### 4.1 铅笔

| 属性 | 数值 | 说明 |
|---|---|---|
| 长度 | 170 mm | 真铅笔 175–190 mm；稍微缩短，让两端避开拇指 |
| 直径 | 8 mm | 真铅笔 7 mm；稍微加粗，让它和手的网格之间的接触更稳定 |
| 质量 | 8 g | |
| 转动惯量（对角） | 6.4e-8、1.93e-5、1.93e-5 kg·m² | 实心杆：绕长轴 `m r²/2`，绕横轴 `m(3r² + L²)/12` |
| 碰撞体 | 沿 body 局部 x 轴的一个胶囊体 | +x 是笔尖，−x 是橡皮；笔尖、金属箍和橡皮只是外观 |
| 摩擦 | 滑动 1.0，扭转 0.005，滚动 0.0001 | 和手的 geom 相同 |
| 生成位置 | (0.17, 0.045, 0.17) m | 在静止位置（z ≈ 0.162）上方 8 mm，位于指根上方 |

中心再往后（x = 0.15）的话，橡皮那端会落在掌后的塔座上，铅笔翘起 20°。

### 4.2 手的姿态（`PENCIL_HAND_POSE`）

在方块任务的掌心朝上姿态基础上，把四指并拢、手腕后仰。没列出的关节保持方块任务的参考值
（thumb_mcp 0、thumb_abd −0.733、thumb_pip −0.585、thumb_dip −0.505，四指的 mcp/pip 都是 0）。

| 关节 | 数值（rad） | 方块任务中的值 |
|---|---|---|
| `right_wrist` | −0.30 | −0.15 |
| `right_index_abd` | −0.15 | −0.40 |
| `right_middle_abd` | 0.00 | 0.00 |
| `right_ring_abd` | −0.05 | 0.17 |
| `right_pinky_abd` | 0.08 | 0.52 |

- 方块任务的姿态下指尖张开约 3 cm。试了 6 个初始偏航角（铅笔中心在 x = 0.15），60°、90°、120° 三个都从指缝掉下去；
  换成这个姿态后，12 个初始角度（各落稳 3 秒）全部留在手上，静止时倾斜 3–4°。
- 手腕伺服在手的自重下会下垂，−0.30 → −0.21，掌面基本水平。
- 把手指并得更紧会让手指之间互相碰撞，而这正是小组安全层要防止的情况。

### 4.3 复位

1. 如果开启了物理随机化，先做随机化（第 8 节）。
2. 手摆到 `PENCIL_HAND_POSE`；铅笔放到生成位置，xy 方向随机偏移 ±5 mm，偏航角在 [0°, 360°) 均匀随机。
3. **保持伺服目标不变，落稳 100 个控制步（1 秒）。** 落稳 100 步后，接下来 2 秒的偏航漂移平均 0.6°（最多 2.5°）；
   只落稳 60 步的话，铅笔可能还在以 26 rad/s 的速度绕自身长轴滚动。
4. **拒绝采样：** 落稳后如果铅笔不在手上，或者倾斜超过 15°，就重新抽（最多 10 次，只有第一次使用 seed）。
   400 次随机复位里有 5.8% 需要重抽，没有一次需要重抽两次以上。查过的一例是铅笔在 45°、往拇指方向偏 5 mm：
   它生成在拇指里，落稳过程中被弹到了地上。
5. 第一个动作接着落稳时保持的伺服目标往下走（如果按下垂后的关节角重新算目标，第一步每个关节都会松掉），
   同时记下落稳后的关节角 `q_rest`，供可选的姿态项使用。

### 4.4 episode 结束条件

| 条件 | 结果 |
|---|---|
| 铅笔中心低于 0.12 m（`drop_height`） | `terminated`，这一步奖励 −2 |
| 满 1000 步（10 秒） | `truncated` |
| MuJoCo 状态出现 NaN/Inf | `truncated`，奖励 0，`info["sim_unstable"] = True`（第 11 节） |

## 5. 奖励函数

### 5.1 公式

每个控制步（10 ms）：

```
r_t =   spin_coef           * paid_t                                   # 获得的偏航角，有速率上限（5.3–5.5）
      - action_rate_penalty * sum_j ((target_t,j - target_t-1,j) / halfspan_j)^2
      - pose_penalty        * sum_j (q_t,j - q_rest,j)^2                # 可选，默认关闭
      - drop_penalty        * [这一步铅笔掉落]
```

| 项 | 默认值 | 每步最大值 | 每个 10 秒 episode |
|---|---|---|---|
| `spin_coef * paid` | 1.0 | +0.01（1 rad/s × 10 ms） | 最多 +10 |
| 动作变化 | 0.002 | −0.00077（17 个关节都走满 0.15 个半程） | — |
| 姿态 | 0（建议试 5e-4） | pencil1 最后的拳头（≈9 rad²）在 5e-4 下：−0.0045 | — |
| 掉落 | 2.0 | −2，只扣一次，episode 结束 | — |

`j` 遍历 17 个驱动关节；`halfspan_j` 是关节 `j` 控制范围的一半；`q` 是测得的关节角。
所以一个完美的 10 秒 episode 回报约为 10（5.8 节的 oracle 得 9.98）。

### 5.2 什么时候计入偏航

只有同时满足以下两个条件的步才计入偏航：

- **铅笔被握住**：铅笔中心高于 `in_hand_height` = 0.13 m，**并且**碰到手掌或任意一根手指的任意一节（`_contacts`）。
  掌后的塔座也是手模型的一部分，但铅笔搭在塔座上不算握住。
- **铅笔接近水平**：轴线与水平面的夹角不超过 `max_tilt_deg` = 45°。

其他步既不付分，也暂时不扣分：参考朝向被冻结（5.5 节）。

### 5.3 怎么量偏航：铅笔轴线的朝向

设 `a` 为铅笔长轴（局部 x 轴，从橡皮指向笔尖）在世界坐标系中的方向。朝向是它在水平面上投影的方向：

```
psi = atan2(a_y, a_x)
spin_t = sigma * wrap(psi_t - psi_ref)        # "ccw" 时 sigma = +1，"cw" 时 -1；wrap() 映射到 [-pi, pi)
```

`psi_ref` 是上一个计入步的朝向。为什么用这个，而不用角速度 `ω_z`：

- **圆铅笔绕自身长轴滚动**是在手掌上最省力的动作。它的角速度沿着 `a`，`a` 不变，所以对 `psi` 没有贡献；
  但铅笔有倾斜时，它会对 `ω_z` 有贡献。
- **把铅笔立起来自转**时水平投影接近零；而且 45° 的倾斜限制本来就把它排除了。
- 对一步前后的朝向做差，量的是这一步实际发生的转动，而不是某个瞬间的速度采样。

### 5.4 速率上限和结转

如果直接按 `spin_t` 付分，把铅笔甩起来自由旋转会比稳定地转更划算。所以正向偏航设了上限 `max_spin_rate` = 1 rad/s。
上限**只作用于正向**，超过上限的正向偏航**结转**（最多 `spin_carry_rad` = 0.2 rad）到之后的步：

```
计入的步：
    k       = 距上一个计入步的步数                      # 通常为 1；冻结之后会更大
    gained  = spin_t + carry
    paid_t  = min(gained, max_spin_rate * dt * k)        # dt = 0.01 s
    carry   = min(gained - paid_t, spin_carry_rad)       # 永远不为负
    psi_ref = psi_t
其他步：
    paid_t  = 0                                          # psi_ref 和 carry 保持不变
```

**不变式。** 每个计入步都满足 `carry_旧 + spin_t = paid_t + carry_新 + 丢弃量`，且 `carry ≥ 0`、`丢弃量 ≥ 0`。对整个 episode 求和：

```
sum(paid) = sum(spin) - carry_最终 - sum(丢弃量)  <=  sum(spin) = 计入的净转角
```

**不管策略做什么，回报都不可能超过净转角。** 而且每步最多付 `max_spin_rate * dt`，所以一个 episode 最多付 1 rad/s × 10 s = 10。

每一部分为什么存在：

| 设计 | 没有它会出什么问题 | 实测（`pencil_checks.py`） |
|---|---|---|
| 正向设上限 | 把铅笔甩起来快速自由旋转，比稳定地转拿分更多 | 以 5 倍上限的速度转得 10.00，和按上限转（9.98）一样 |
| **只给**正向设上限 | 两个方向都设上限的话，慢慢正转（全额付分）、快速倒回（只按上限扣分），净转角为零，奖励却是正的 | 棘轮动作：这里得 0.05；双向上限会付 **3.41** |
| 结转超出的部分 | 被伺服手指压着的铅笔会抖。每次抖动的正向那半被上限截断，反向那半全额扣，抖动就变成了隐性惩罚 | 以 0.5 rad/s 转、同时 ±2° 抖动：这里得 4.90（转了 5 rad）；没有结转 **−10.63** |
| 结转最多 0.2 rad | 不限额的结转会把一次甩动存起来，之后慢慢付出去 | — |

结转是在第一次冒烟训练之后加的。没有结转时，随机动作下净偏航是 −0.15 rad，奖励却付了 −3.54；
训练中 `task/paid_frac`（奖励付的偏航 ÷ 净偏航）在 30 万–40 万步时只有 0.13–0.29。加上结转后，同一阶段是 0.66–0.83，训练结束时 0.93–0.97。

### 5.5 铅笔不计入时冻结参考朝向

铅笔没被握住或倾斜超过 45° 时，`psi_ref` 保持不变。等铅笔重新被计入时，`spin` 就是冻结以来的全部变化，上限也按 `k` 步给。
没有这一条的话，策略可以在手上把铅笔往前转（付分），抛起来在空中转回去（不扣分），接住，再重复。
实测：每个循环在手上正转 0.5 rad、在空中倒转 1.0 rad，一个 episode 得 **−4.85**；没有冻结的话会是正分。

### 5.6 惩罚项

- **动作变化**（`action_rate_penalty` = 0.002）：`Σ_j (Δtarget_j / halfspan_j)²`，和方块任务的平滑项相同。
  每步每个关节最多变化 0.15 个半程，所以这一项每步最多 0.00077：它让动作平滑，但不会和任务本身抢。
- **掉落**（`drop_penalty` = 2.0）：铅笔中心低于 0.12 m 时扣一次，episode 结束，所以掉落还会失去剩下的时间。
  它被设得远小于一个完整 episode 能拿到的 10 分：方块任务里掉落惩罚大于成功奖励时，策略学会了把物体夹死不动。
- **姿态**（`pose_penalty`，默认 0）：17 个关节的 `Σ_j (q_j − q_rest_j)²`，参照的是复位时手落稳后的关节角
  （不是指令姿态，所以重力造成的 0.09 rad 手腕下垂不扣分，手不动时这一项正好是 0）。它是因为 pencil1 握拳的问题而加的（第 1 节）。取 5e-4 时：

  | 手的状态 | Σ(q − q_rest)² | 每步扣分 | 占每步最大转动奖励的比例 |
  |---|---|---|---|
  | 不动 | 0 | 0 | 0% |
  | 四指 8 个 mcp/pip 关节都弯到 0.3–0.4 rad | 0.72–1.28 rad² | 0.0004–0.0006 | 4–6% |
  | pencil1 最后的拳头（按平均弯曲估算） | ≈ 9 rad² | ≈ 0.0045 | ≈ 45% |
  | `pencil_checks` 里的脚本拳头，手指关节推到限位 | — | ≈ 0.0105 | ≈ 105%（每个 episode −10.53） |

### 5.7 奖励里刻意没有的东西

| 没有的项 | 原因 |
|---|---|
| 成功奖励 | 没有离散的目标；给「转满一圈」发奖励，只是把奖励已经连续支付的同一个偏航角变成阶跃函数 |
| 存活 / 握住奖励 | 等于为什么都不做付钱——原方块任务的「拖延」漏洞（`orca_rl/README.md`） |
| 每步固定扣分 | 会让提前结束 episode（故意掉笔）变得划算 |
| 按 `ω_z` 给奖励 | 会为倾斜的铅笔绕自身长轴滚动付钱（5.3 节） |
| 铅笔位置项 | 目前没有看到铅笔到处乱跑；掉落本身已经会结束 episode |

### 5.8 脚本检查（`python -m orca_rl.pencil_checks`）

大多数探针是直接摆动铅笔，并不是手能做出来的策略，而是用来测试奖励到底为什么付钱。每个 10 个 episode，每个 10 秒：

```
policy (10 episodes of 10 s)             return    revs   drop
--------------------------------------------------------------
do nothing                                -0.20    0.00    10%
random                                    -2.12   -0.00   100%
oracle: ccw at the cap, 1 rad/s            9.98    1.59     0%
oracle: ccw at 5x the cap                 10.00    7.96     0%
oracle: wrong way (cw), 1 rad/s          -10.01   -1.59     0%
ratchet: creep fwd, snap back              0.05    0.01     0%
0.5 rad/s with a +-2 deg shiver            4.90    0.78     0%
roll about its own axis, 20 rad/s          0.02    0.00     0%
stand it 60 deg up and turn                0.00    0.00     0%
toss: fwd in hand, back in air            -4.85   -0.77     0%

with --pose-penalty 0.0005:
  do nothing                              -0.20    0.00    10%
  oracle: ccw at the cap, 1 rad/s          9.98    1.59     0%
  close a fist and hold                  -10.73    0.03    10%
```

断言：什么都不做约得 0；按上限转 10 秒约得 10；转快 5 倍也不会多得；反方向全额扣分；棘轮动作得不到分；抖动不扣分；
原地滚动和转立着的铅笔都得不到分；在空中丢掉的偏航角落回手上时会被扣；加上姿态项后，不动和 oracle 的得分不变，握拳每个 episode 至少多扣 2。
random 那一行每次运行会略有不同（它的动作没有固定种子）。「什么都不做」有 10% 的掉落，是第 11 节说的慢慢滑落造成的。

## 6. 观测

90 个值，float64，训练时由 `VecNormalize` 归一化（滑动均值 / 标准差，裁剪到 ±10）。

| 下标 | 维数 | 内容 | 坐标系 |
|---|---|---|---|
| 0–16 | 17 | 手的关节角（驱动器顺序，第 7 节） | — |
| 17–19 | 3 | 铅笔中心位置 | 世界 |
| 20–23 | 4 | 铅笔姿态四元数（w, x, y, z） | 世界 |
| 24–40 | 17 | 手的关节角速度 | — |
| 41–43 | 3 | 铅笔线速度 | 世界 |
| 44–46 | 3 | 铅笔角速度 | 铅笔 body |
| 47–49 | 3 | 铅笔轴线 `a`（橡皮 → 笔尖），单位向量 | 世界 |
| 50–52 | 3 | 铅笔角速度 | 世界 |
| 53–69 | 17 | 当前伺服目标，映射到 [−1, 1] | — |
| 70–84 | 15 | 指尖位置减去铅笔中心：拇指、食指、中指、无名指、小指（各 x, y, z） | 世界 |
| 85–89 | 5 | 每根手指的任意一节是否碰到铅笔：拇指、食指、中指、无名指、小指（0/1） | — |

- 世界坐标系也就是手的坐标系：手是固定不动的。
- 伺服目标（53–69）让任务对动作是马尔可夫的：相对模式下动作是对它的增量，`kp · (target − q)` 就是握力。
  方块任务里在它被加进观测之前，策略有 77% 的时间把目标顶在关节限位上做 bang-bang 控制（复盘四）。
- 指尖相对位置和接触标志让方块任务的掉落率减半（复盘十一），所以这里一直开着。
- 开启 `--randomize-physics` 时，17 个手关节角会加上高斯噪声（σ = 0.005 rad）。
- 下标 17–23、41–52、70–89 是只有仿真器才有的信息（第 11 节）。

## 7. 动作和控制

- **时间：** MuJoCo 时间步 2 ms，每个动作保持 5 步 → 控制周期 10 ms（100 Hz）。
- **驱动器：** 17 个位置伺服，`kp` = 2.0 N·m/rad，力矩上限 ±1 N·m（orca_sim 的 v1 模型）。
- **动作：** 17 个 [−1, 1] 之间的值。默认 `relative` 模式：
  `target_t = clip(target_t−1 + 0.15 · a · halfspan, low, high)`——每步每个伺服目标最多移动其范围一半的 15%。
  `absolute` 模式把 `a` 直接映射到关节范围。
- **驱动器顺序（v1）：** 手腕；拇指 mcp、abd、pip、dip；食指 abd、mcp、pip；中指 abd、mcp、pip；
  无名指 abd、mcp、pip；小指 abd、mcp、pip。v2 的顺序不同，所以两种手的策略不能互换。

| 驱动器 | 范围（rad） | | 驱动器 | 范围（rad） |
|---|---|---|---|---|
| wrist | −0.674 … 0.897 | | middle_abd | −0.646 … 0.646 |
| thumb_mcp | −0.873 … 0.873 | | middle_mcp | −0.349 … 1.588 |
| thumb_abd | −1.082 … 0.000 | | middle_pip | −0.349 … 1.868 |
| thumb_pip | −0.794 … 1.230 | | ring_abd | −0.476 … 0.806 |
| thumb_dip | −0.854 … 1.450 | | ring_mcp | −0.349 … 1.588 |
| index_abd | −1.046 … 0.246 | | ring_pip | −0.349 … 1.868 |
| index_mcp | −0.349 … 1.658 | | pinky_abd | −0.122 … 1.169 |
| index_pip | −0.349 … 1.885 | | pinky_mcp | −0.349 … 1.710 |
| | | | pinky_pip | −0.349 … 1.885 |

## 8. 环境参数

`PencilSpin(render_mode=None, version="v1", **kwargs)`。训练时用到的每个值都会写进 `runs/<name>/env_kwargs.json`。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `version` | `"v1"` | 只有 v1 有铅笔场景；其他值会报错 |
| `max_episode_steps` | 1000 | episode 长度（10 秒） |
| `spin_direction` | `"ccw"` | `"ccw"`（+z，从上往下看逆时针）或 `"cw"` |
| `max_spin_rate` | 1.0 | rad/s；超过这个速率的正向偏航不付分（5.4 节） |
| `spin_carry_rad` | 0.2 | 超出上限的正向偏航最多结转多少 rad 到之后的步（5.4 节） |
| `spin_coef` | 1.0 | 转动项的权重 |
| `drop_penalty` | 2.0 | 掉落时扣一次（5.6 节） |
| `pose_penalty` | 0.0 | 每 rad² 偏离落稳姿态的扣分（5.6 节）；建议值 5e-4 |
| `action_rate_penalty` | 0.002 | 平滑项（5.6 节） |
| `max_tilt_deg` | 45.0 | 只有倾斜在这个角度以内时才计入偏航（5.2 节） |
| `in_hand_height` | 0.13 | m；握住要求铅笔中心高于此值（5.2 节） |
| `drop_height` | 0.12 | m；低于此值 episode 以掉落结束 |
| `action_mode` | `"relative"` | `"relative"` 或 `"absolute"`（第 7 节） |
| `action_scale` | 0.15 | 相对动作每步最多移动半程的比例 |
| `randomize_physics` | False | 每次复位随机：铅笔质量 ±30%、铅笔摩擦 ±30%、关节阻尼 ±30%、驱动器增益 ±20%、关节角噪声 σ 0.005 rad |
| `randomization` | None | 覆盖上述范围的字典（键 `cube_mass`、`cube_friction`、`joint_damping`、`actuator_gain`、`obs_noise_rad`；名字和方块任务共用） |
| `reset_settle_steps` | 100 | episode 开始前铅笔落稳的控制步数（4.3 节） |
| `initial_yaw` | `"random"` | `"random"`（0–360° 均匀）或 `"nominal"`（0°） |
| `pencil_pos_xy_jitter` | 0.005 | m，生成位置在 xy 上的均匀随机偏移 |

代码里写死的：复位最多尝试 10 次，落稳后最大倾斜 15°（4.3 节）；`PENCIL_HAND_POSE`（4.2 节）。

`reset(seed=None, options=None)` 接受 `options={"cube_quat": q, "cube_pos": p}` 来固定铅笔的生成位姿（这时不做拒绝采样），
也接受父类的其他选项（`hand_pose_by_joint`、`hand_qpos` 等）。

`reset` 和 `step` 返回的 `info`：

| 键 | 含义 |
|---|---|
| `yaw_deg`、`tilt_deg` | 铅笔轴线的朝向，以及它和水平面的夹角 |
| `pencil_pos`、`pencil_height` | 铅笔中心（m） |
| `in_hand`、`dropped` | 是否握住（5.2 节）；是否低于 `drop_height` |
| `elapsed_steps` | 本 episode 已走的步数 |
| `turned_rad`、`turned_revs` | 到目前为止在指定方向上计入的净偏航 |
| `paid_rad` | 其中已经由奖励付出的部分 |
| `in_hand_frac` | 到目前为止握住的步数比例 |
| `spin_this_step` | 这一步的 `spin_t`（不计入时为 0）；只有 `step` 返回 |
| `sim_unstable` | 这一步因 MuJoCo 数值爆掉而被截断时出现，值为 True |

## 9. 训练：PPO 和记录的指标

Stable-Baselines3 2.9.0 的 PPO。这些值是方块任务调好的设置；每一项为什么和 SB3 默认值不同，见 `orca_rl/README.md`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--n-envs` | 10 | 并行环境数（`SubprocVecEnv`） |
| `--n-steps` | 512 | 每个环境每次 rollout 的步数 → 每次更新 5120 条样本 |
| `--batch-size` | 1024 | 每个 epoch 5 个 minibatch |
| `--n-epochs` | 10 | 上限；通常会被 `target_kl` 提前截停（pencil1 最后每次更新 2 个 epoch） |
| `--lr` | 1.5e-4 | 常数 |
| `--gamma` | 0.995 | 100 Hz 下约 2 秒的视野 |
| `--gae-lambda` | 0.95 | |
| `--clip-range` | 0.2 | |
| `--ent-coef` | 0.0 | 方块任务里熵奖励把动作标准差推到了 11 |
| `--vf-coef` | 0.5 | |
| `--max-grad-norm` | 0.5 | |
| `--target-kl` | 0.02 | 提前结束 epoch 循环 |
| `--log-std-init` | −0.5 | 初始动作标准差 0.61（标准差为 1 时在 [−1, 1] 里会饱和） |
| `--max-log-std` | 0.0 | 硬上限：标准差 ≤ 1，每次 rollout 后施加 |
| `--device` | `cpu` | 网络很小，用 GPU 反而更慢 |
| `--checkpoint-every` | 1,000,000 | 保存模型和归一化统计量 |
| `--seed` | 0 | 第 `i` 个环境用 `seed + i` 复位 |

任务参数：`--max-episode-steps`、`--spin-direction`、`--max-spin-rate`、`--spin-carry`、`--drop-penalty`、
`--pose-penalty`、`--max-tilt-deg`、`--action-mode`、`--action-scale`、`--randomize-physics`（含义同第 8 节）。
`--resume MODEL.zip` 接着训练；超参数会从命令行重新读取。

**网络：** `MlpPolicy`，actor 和 critic 分开，各自 90 → 256 → 256、tanh 激活；线性输出层分别给出 17 个动作均值（actor）和 1 个价值（critic）；
每个动作维度一个与状态无关、可学习的 log-std。共 182,819 个参数。
**归一化：** `VecNormalize` 同时归一化观测和奖励（奖励归一化使用同一个 γ）。

**TensorBoard**（`task/` 每 40 个结束的 episode 写一次；这是带探索噪声的训练数据，正式数字请用 `pencil_eval.py`）：

| 指标 | 含义 |
|---|---|
| `task/revs_per_episode` | **进度指标。** 每个 episode 平均计入的圈数（oracle：1.59） |
| `task/spin_rate_deg_s` | 每秒 episode 时间计入的偏航，掉落的 episode 也算在内 |
| `task/paid_frac` | 奖励付的偏航 ÷ 计入的偏航。接近 1：稳定地转；远小于 1：超过上限或在抖 |
| `task/drop_rate` | 以掉落结束的 episode 比例 |
| `task/episode_length` | 每个 episode 的平均步数 |
| `task/in_hand_frac` | 铅笔被握住的步数比例 |
| `task/unstable_rate` | 因 MuJoCo 数值爆掉而截断的 episode 比例；应该接近 0 |
| `train/epochs_used` | 被 `target_kl` 截停前实际跑了几个 PPO epoch |
| `train/std` | 动作标准差；pencil1 从 0.61 降到 0.30 |

## 10. 函数说明

### 10.1 `orca_rl/pencil.py`

| 名称 | 作用 |
|---|---|
| `PENCIL_HAND_POSE` | 复位姿态中被覆盖的关节角（4.2 节） |
| `SCENE_TEMPLATE` | `scenes/scene_right_pencil_spin_v1.xml` 的路径 |
| `pencil_scene_path()` | 把模板里的 `{orca_sim}` 占位符替换成已安装的 orca_sim 包的路径，结果写到 `<tmp>/orca_rl/scene_right_pencil_spin_v1_<hash>.xml`（用原子重命名写入，所以并行的子进程不会读到写了一半的文件）。这样 orca_sim 可以保持为未修改的上游包：MuJoCo 按包含它的文件来解析 include 路径，所以模板用绝对路径引用 orca_sim 的地面场景和 v1 手模型。编译出的模型已经逐字段核对过，和之前放在包里的场景完全一致 |
| `_wrap(angle)` | 把角度映射到 [−π, π) |
| `saved_pencil_kwargs(model_path)` | 读取模型旁边（或上一级目录，用于 checkpoint）的 `env_kwargs.json`，去掉 `task` 标签 |

`class PencilSpin(OrcaHandRightCubeOrientation)`——铅笔在这里扮演父类中「方块」的角色（`_cube_body_id`、`_cube_qpos_adr` 等）。

| 方法 | 作用 |
|---|---|
| `__init__(...)` | 检查参数，通过 `pencil_scene_path()` 加载场景，建立 [−1, 1] 的动作空间，收集铅笔的碰撞 geom，识别手指（`_setup_fingers`），保存物理参数的标称值供随机化使用，建立观测空间 |
| `_axis()` | 铅笔局部 x 轴在世界坐标系中的方向（橡皮 → 笔尖） |
| `_yaw()` | `atan2(a_y, a_x)`——轴线水平投影的朝向 |
| `_tilt_rad()` | `asin(|a_z|)`——轴线和水平面的夹角 |
| `_angvel_world()` | 把自由关节在 body 系下的角速度转到世界系 |
| `_contacts()` | 遍历一次 MuJoCo 接触：（铅笔是否碰到手掌或手指，5 个手指的接触标志） |
| `_in_hand()` | 铅笔中心高于 `in_hand_height`，并且碰到手掌或手指 |
| `_resolve_initial_cube_quat(options)` | 生成时的姿态：`options["cube_quat"]`，否则绕世界 z 轴随机（或为零）的偏航 |
| `_get_obs()` | 90 维观测（第 6 节） |
| `_get_info()` | `info` 字典（第 8 节） |
| `_reset_tracking()` | 清零本 episode 的偏航记账：`psi_ref`、`k`、结转、计入和已付的偏航、握住步数 |
| `reset(seed, options)` | 随机化 → 摆放 → 落稳 → 必要时拒绝并重抽 → 接着落稳时的伺服目标，记下 `q_rest`（4.3 节） |
| `step(action)` | 动作映射 → MuJoCo 走 5 步 → 检测数值爆掉 → 是否握住、是否掉落 → 计算带上限和结转的计入偏航 → 奖励（5.1 节） → terminated / truncated → 返回观测、奖励、标志、info |

原样借用 `CubeReorientContinuous`（`task.py`）的方法：`_setup_fingers`（从运动学树找出每根手指的连杆和指尖碰撞 geom）、
`_apply_randomization`（每次复位的物理随机化）、`_settle_cube`（保持伺服目标 `reset_settle_steps` 步）、
`_target_from_action`（相对 / 绝对动作映射和裁剪）。继承自 orca_sim 的：`_cube_pos`、`_cube_qvel`、`_cube_dropped`、
`_compose_ctrl_from_qpos`、`_quat_from_axis_angle`、`render`、`close`。

### 10.2 `orca_rl/pencil_checks.py`

| 名称 | 作用 |
|---|---|
| `run(policy, seeds, steps, **env_kwargs)` | 用 10 个 seed 跑一个脚本策略；返回平均回报、圈数、掉落率，以及双向上限和无结转上限分别会付多少 |
| `place(env, pos, quat, angvel_body)` | 直接设置铅笔的位姿（然后 `mj_forward`），返回零动作 |
| `yaw_quat(yaw, tilt)` | 朝向为 `yaw`、从水平抬起 `tilt` 的铅笔的四元数 |
| `zero`、`random` | 基线 |
| `turning(rate)` | oracle：让铅笔平躺在手掌上以 `rate` rad/s 转动 |
| `ratchet` | 以 0.5 rad/s 慢慢正转 0.2 rad，再以 4 rad/s 快速倒回 0.2 rad |
| `shiver` | 以 0.5 rad/s 转动，同时每 2 步做一次 ±2° 的抖动 |
| `roll` | 让铅笔以 20 rad/s 绕自身长轴转 |
| `standing` | 铅笔立起 60° 时让它转动 |
| `toss` | 在手上正转 0.5 rad，抬到 5 cm 高处在空中倒转 1.0 rad，再放回去 |
| `fist` | 把四根手指弯到限位并保持 |
| `main()` | 跑全部探针，打印表格，逐条断言，有任何一条失败就以退出码 1 结束 |

### 10.3 `orca_rl/pencil_train.py`

| 名称 | 作用 |
|---|---|
| `PencilMetricsCallback(window=40)` | 收集每个结束 episode 的最后一个 `info`，每 40 个 episode 记录一次 `task/` 指标 |
| `parse_args()` | 命令行参数（第 9 节） |
| `main()` | 写出 `env_kwargs.json`（带 `"task": "pencil_spin"`），建立向量化环境，新建或接着训练 PPO，带回调训练，保存模型和归一化统计量（Ctrl-C / SIGTERM 时也会保存） |

复用 `train.py` 的：`build_vec_env(..., env_cls=PencilSpin)`、`EpochsUsedCallback`、`ClampLogStdCallback`、`resolve_tensorboard_dir`。
对 `train.py` 唯一的改动是 `make_env_fn` 和 `build_vec_env` 新增的 `env_cls` 参数（默认 `CubeReorientContinuous`，所以方块任务不受影响）。

### 10.4 `orca_rl/pencil_eval.py`

| 名称 | 作用 |
|---|---|
| `build_actor(args)` | `zero`、`random`，或者一个 PPO 模型，连同它的 `VecNormalize` 统计量和保存的环境设置 |
| `make_camera(env)` | 对准手掌的固定相机（方位角 200°，仰角 −55°，距离 0.34 m） |
| `main()` | 跑完所有 episode，打印圈数、转速、掉落率、episode 长度、握住比例、回报；可选写出视频 |

参数：`--model`、`--policy {model,random,zero}`、`--episodes`（50）、`--max-episode-steps`（1000）、`--seed`
（1234；第 `i` 个 episode 用 `seed + i`）、`--stochastic`、`--randomize-physics`、`--video OUT.mp4`、`--fps`（50 = 半速）、
`--clean`（1280×720，无文字）。

## 11. 已知问题和下一步

- **策略夹住铅笔而不是转它**（第 1 节）。下一步：`pencil2`，只比 pencil1 多加 `--pose-penalty 5e-4`。
  要看的数字是后 5 秒仍在转的 episode 比例（pencil1 约 10%）。之后要下结论还需要第二个 seed，和方块任务的规则一样。
- **铅笔顺着手指方向静置时会滑进指缝。** 朝向在 15° 和 165° 附近、手不动时，铅笔 4–8 秒后滑进指缝，然后竖着掉下去；
  这就是「什么都不做」有 10% 掉落率的原因。这是 v1 手真实的几何特性，策略需要学会处理（并拢或屈指，或者不在这个角度停下）。
- **MuJoCo 偶尔会数值爆掉。** 300 万步的冒烟训练里一次，pencil1 的 3000 万步里两次（`MUJOCO_LOG.TXT`：QACC 在铅笔的一个转动自由度上出现 NaN/Inf）。
  这时 MuJoCo 会把整个状态重置为 XML 里的姿态；`PencilSpin.step` 通过仿真时钟倒退检测到这种情况，截断 episode 且不计分。
  两次训练里的第一次事件完全相同（仿真时间 1.4260，DOF 21），所以在 seed 0 下可以复现。
  如果 `task/unstable_rate` 上升，可以加大铅笔绕长轴的转动惯量（6.4e-8 kg·m²）或者减小时间步。
- **只有仿真器才有的观测。** 铅笔的位姿、速度和每根手指的接触在实体手上都拿不到；上真机之前需要非对称 actor-critic 或状态估计，和方块任务一样。
- **圆铅笔。** 碰撞胶囊体比真正的六棱铅笔更容易滚。滚动不得分（5.3 节），但更容易从手掌上滚下去。
- **还没训练过的：** 物理随机化；第二个 seed；安全层（`--safety` 目前只支持方块任务）。
- **只有 v1。**
