"""Generate v1 right-hand open -> index/middle crossing -> open commands."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from orca_sim import OrcaHandRight
from orca_safety_v2.adapter import build_sim_model


def make_trajectory(open_q, cross_q, dt, durations=(1., 1.5, 1., 1.5, 1.)):
    """Cubic transitions with flat endpoints; all durations are simulation seconds."""
    counts = [int(round(seconds/dt)) for seconds in durations]
    if any(count < 1 for count in counts):
        raise ValueError('each phase must contain at least one control step')
    trajectory, phases = [], []
    for name, count in zip(['open', 'approach', 'cross_hold', 'return', 'open_end'], counts):
        t = np.arange(1, count+1)/count
        smooth = t*t*(3-2*t)
        alpha = smooth if name == 'approach' else (1-smooth if name == 'return' else
                np.ones(count) if name == 'cross_hold' else np.zeros(count))
        trajectory.extend(open_q+alpha[:, None]*(cross_q-open_q))
        phases.extend([name]*count)
    return np.asarray(trajectory), phases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('Orca_safety/output/synthetic/index-middle-crossing'))
    args = p.parse_args()
    if args.output.exists():
        p.error('output directory exists; choose a new name')
    env = OrcaHandRight(version='v1'); env.reset()
    try:
        model, config, low, high, max_step, addresses = build_sim_model(env)
        dt = float(env.model.opt.timestep*env.frame_skip)
        open_q = env.data.qpos[addresses].copy()
        cross_q = open_q.copy()
        index = {n.removeprefix('right_'):i for i,n in enumerate(model.joint_names)}
        # Only lateral joints of index and middle change. Other fingers and all
        # flexion joints retain the initial open pose, making the X easy to see.
        cross_q[index['index_abd']] = .23
        cross_q[index['middle_abd']] = -.40
        pairs = [n for n in model.pair_indices if n.startswith('right_index_') and '__right_middle_' in n]
        if np.any(cross_q < low) or np.any(cross_q > high):
            raise ValueError('crossing target violates joint bounds')
        if not any(d.collision for d in model.distances(cross_q, pairs)):
            raise ValueError('target does not produce index-middle overlap in geometry')
        trajectory, phases = make_trajectory(open_q, cross_q, dt)
        header = dict(event='start', software='orca_safety_v2', robot='v1_right',
                      source='synthetic_index_middle_crossing', mode='synthetic_nominal',
                      joint_names=list(model.joint_names), units={'angle':'rad','distance':'m','time':'s'},
                      config=vars(config), low=low.tolist(), high=high.tolist(), max_step=max_step.tolist(),
                      reference_offsets=model.offsets.tolist(), pairs=list(model.pair_indices),
                      open_target=open_q.tolist(), crossing_target=cross_q.tolist(),
                      timing_note='Artificial timestamps for replay; not webcam or live retarget data.')
        args.output.mkdir(parents=True)
        # Stable synthetic epoch keeps existing clip readers usable, without
        # pretending these commands were recorded from a person or camera.
        epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
        with (args.output/'motion.jsonl').open('x') as file:
            file.write(json.dumps(header)+'\n')
            for i, (q, phase) in enumerate(zip(trajectory, phases)):
                row = dict(event='frame', frame=i, timestamp=(epoch+timedelta(seconds=i*dt)).isoformat(),
                           sim_time=(i+1)*dt, q_nominal=q.tolist(), phase=phase)
                if i == 0:
                    row['q_current'] = open_q.tolist()
                file.write(json.dumps(row)+'\n')
        (args.output/'README.md').write_text('''# 食指与中指交叉：合成目标动作

`motion.jsonl` 是 v1 右手的合成 nominal 输入，不是人手 retarget 录制，不包含实际执行结果。兼容 `replay_viewer.py` 和 `replay_video.py`，同一文件用于 OFF 和 ON。

共 600 个控制步，每步 0.01 秒，共 6 秒仿真时间：

- 0～1 秒：保持五指张开的模型 reset 姿态。
- 1～2.5 秒：食指、中指逐渐靠近并交叉。
- 2.5～3.5 秒：保持交叉目标。
- 3.5～5 秒：逐渐恢复张手。
- 5～6 秒：保持张手，给实际运动留出稳定时间。

只修改食指与中指的外展关节，其他关节目标保持不变。交叉目标是食指 abd=0.23 rad（约 13.18°）、中指 abd=−0.40 rad（约 −22.92°），采用平滑的三次插值。这些是模型中的绝对角度，已包含该模型的坐标约定，不是从零姿态计算的增量。

目标在关节范围内，但配置的食指—中指碰撞几何会相交，用于观察自碰撞安全层的干预。OFF 仍保留 MuJoCo 自身的控制范围和接触动力学，接触力可能让手指停在挤压姿态，无法完全实现交叉目标；ON 应提前阻止手指继续靠近，未完成交叉是预期行为。两指近距离观察时可用鼠标缩放和旋转。实际运动以回放结果为准。

从工作区根目录运行：

```bash
orca_teleop/.venv/bin/mjpython Orca_safety/scripts/replay_viewer.py \\
  Orca_safety/output/synthetic/index-middle-crossing/motion.jsonl \\
  --mode OFF --view front --speed 0.5 --paused
```

把 OFF 改为 ON 即可观察安全层效果。macOS 必须使用 mjpython 启动交互窗口。空格播放/暂停，N/B 前进/后退一步，R 回到开头；鼠标自由调整视角。无需 `--video-timing`，因为本数据没有摄像头视频；`--clip` 如需使用，按本合成轨迹的时间筛选。

Safety ON 是对合成目标重新运行当前版本的安全层，不是预先手工编排的一条避碰动作。安全层只检查配置范围和采样状态，不保证任何连续轨迹或实体手安全。本数据只用于仿真展示。
''')
        print(f'Saved {len(trajectory)} steps ({len(trajectory)*dt:.2f}s): {args.output}/motion.jsonl')
    finally:
        env.close()


if __name__ == '__main__':
    main()
