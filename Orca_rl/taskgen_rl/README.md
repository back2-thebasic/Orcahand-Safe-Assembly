# taskgen_rl: PPO on the teammates' generated tasks

[中文说明见下方](#中文)

PPO training on the three example bundles of
[niujiazhen/orca-agent-taskgen](https://github.com/niujiazhen/orca-agent-taskgen) @ `d2809a5`
(v1 hand, kinematic 6-DoF wrist). The environment changes needed for PPO to solve them are in
`patches/` as 8 commits; they are not pushed to the teammate's repo.

| Task | Success (100 episodes) | Median steps | Scripted controller |
|---|---|---|---|
| Pick up red cube | 100% (seed 0) | 28 | 100%, 426 steps |
| Open → half-close → fist → open | 100% (seeds 0 and 1) | 39 / 40 | 100%, 37 steps |
| Pick up blue cylinder and place it | 100% (seed 0), 0% (seed 1) | 63 | 100%, 964 steps |

The successful policies satisfy the tasks' success tests but move unrealistically: they fling the
cube, toss the cylinder into the target, and twist the free wrist during the gesture. Details,
failed attempts and open questions: `../results/2026-09-29-taskgen.md` (Chinese).

## Contents

- `patches/0001`–`0008`: reward/success fixes, `rl/train_bundle.py`, `rl/evaluate_bundle.py`,
  `rl/wrappers.py` (optional relative finger actions) and `tests/test_rl_exploits.py`.
- `render_compare.py`: side-by-side video of a trained policy and the scripted controller.
  Copy it into the taskgen repo and run it from there.

## Usage

```bash
cd orca-agent-taskgen
git checkout -b fix/rl-reward-exploits d2809a5
git am ../Orcahand-Safe-Assembly/Orca_rl/taskgen_rl/patches/*.patch
.venv/bin/python -m pytest tests/test_rl_exploits.py -q     # 10 passed

# gesture: relative finger actions are needed
.venv/bin/python rl/train_bundle.py --bundle examples/generated/handfist_v0 \
    --name fist_prec_s0 --seed 0 --relative-hand 0.15 --timesteps 15_000_000
# pick-up and place: absolute actions
.venv/bin/python rl/train_bundle.py --bundle examples/generated/bluecylinderplace_v0 \
    --name place_rel_s0 --seed 0 --timesteps 30_000_000

.venv/bin/python rl/evaluate_bundle.py --model runs_taskgen/place_rel_s0/final_model.zip --episodes 100
.venv/bin/python rl/evaluate_bundle.py --policy scripted --bundle examples/generated/bluecylinderplace_v0
```

---

## 中文

在组员 [orca-agent-taskgen](https://github.com/niujiazhen/orca-agent-taskgen)（`d2809a5`）的三个示例任务上训练 PPO。
让 PPO 能解出这些任务所需的环境修改放在 `patches/`（8 个 commit），没有推到组员仓库。

| 任务 | 成功率（100 回合） | 完成步数（中位） | 脚本控制器 |
|---|---|---|---|
| 抓起红方块 | 100%（seed 0） | 28 | 100%，426 步 |
| 张开 → 半握 → 握拳 → 张开 | 100%（seed 0 和 1） | 39 / 40 | 100%，37 步 |
| 抓起蓝色圆柱并放到目标区 | 100%（seed 0），0%（seed 1） | 63 | 100%，964 步 |

成功的策略满足现有成功条件，但动作不真实：抓起时把方块往上甩，放置时把圆柱抛进目标区，做手势时随意转动手腕。
每一步的修改原因、失败的尝试和待讨论问题见 `../results/2026-09-29-taskgen.md`。

用法见上方 Usage：先 `git am` 补丁，再用 `rl/train_bundle.py` 训练、`rl/evaluate_bundle.py` 评估。
手势任务需要 `--relative-hand 0.15`（相对手指动作），抓起和放置不需要。
