"""Replay a live JSONL nominal sequence with Safety OFF/ON and export videos."""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path

import cv2
import mujoco
import numpy as np

from orca_sim import OrcaHandRight
from orca_safety_v2.adapter import build_sim_model
from orca_safety_v2.config import FilterConfig
from orca_safety_v2.filter import CBFSafetyFilter


def clip_range(value):
    try:
        start, end = map(float, value.split(':'))
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError
        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError('clip must be START:END in recorded wall seconds')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--output', type=Path, required=True, help='new output directory')
    p.add_argument('--clip', type=clip_range, action='append', help='wall seconds from first frame; repeat to concatenate clips')
    p.add_argument('--video-timing', type=Path, help='webcam .timing.json: interpret --clip as webcam video seconds')
    p.add_argument('--speed', type=float, default=1., help='simulation playback speed; 0.5 is half speed')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--view', choices=['oblique', 'front'], default='oblique')
    p.add_argument('--azimuth', type=float, default=None)
    p.add_argument('--elevation', type=float, default=None)
    p.add_argument('--camera-distance', type=float, default=None)
    p.add_argument('--config', type=Path, help='pair YAML matching recording; parameters come from log')
    args = p.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0 or args.fps <= 0:
        p.error('speed and fps must be positive')
    if args.width < 320 or args.height < 240 or args.width % 2 or args.height % 2:
        p.error('use even dimensions of at least 320x240')
    rows = [json.loads(l) for l in args.log.read_text().splitlines() if l.strip()]
    headers = [r for r in rows if r.get('event') == 'start']
    frames = [r for r in rows if r.get('event') == 'frame']
    if len(headers) != 1 or len(frames) < 2:
        p.error('need one start record and at least two frames')
    header = headers[0]
    if header.get('robot') != 'v1_right' or header.get('units', {}).get('angle') != 'rad':
        p.error('expected v1_right absolute-radian log')
    q = np.asarray([r['q_nominal'] for r in frames], dtype=float)
    times = [datetime.fromisoformat(r['timestamp']) for r in frames]
    wall = np.asarray([(t-times[0]).total_seconds() for t in times])
    sim = np.asarray([r['sim_time'] for r in frames])
    if q.shape != (len(frames), len(header['joint_names'])) or not np.isfinite(q).all():
        p.error('invalid nominal sequence')
    if np.any(np.diff(wall) < 0) or np.any(np.diff(sim) <= 0):
        p.error('unordered timestamps')
    selected = np.ones(len(frames), dtype=bool) if not args.clip else np.zeros(len(frames), dtype=bool)
    clip_offset = 0.
    if args.video_timing:
        video_info = json.loads(args.video_timing.read_text())
        clip_offset = (datetime.fromisoformat(video_info['start_timestamp'])-times[0]).total_seconds()
    for start, end in args.clip or []:
        selected |= (wall >= start+clip_offset) & (wall < end+clip_offset)
    if not selected.any():
        p.error('clips contain no control steps')
    if args.output.exists():
        p.error('output directory already exists; choose a new name')
    envs, renderers, writers = [], [], []
    logging.basicConfig(level=logging.ERROR)
    try:
        for _ in range(2):
            env = OrcaHandRight(version='v1'); envs.append(env); env.reset()
        model, _, low, high, max_step, addresses = build_sim_model(envs[1], args.config)
        if list(model.joint_names) != header['joint_names'] or list(model.pair_indices) != header['pairs']:
            raise ValueError('joint order or collision pairs differ from recording; use matching --config')
        for field, actual in [('low', low), ('high', high), ('reference_offsets', model.offsets)]:
            if not np.allclose(actual, header[field], atol=1e-8, rtol=0):
                raise ValueError(f'model {field} differs from recording')
        dt = envs[0].model.opt.timestep*envs[0].frame_skip
        if not np.allclose(np.diff(sim), dt, atol=1e-8, rtol=0):
            raise ValueError('log control-step timing differs from current model')
        # This recorder starts immediately after reset. Refuse to invent missing
        # velocities/actuator state for a log starting mid-run.
        if not np.allclose(frames[0]['q_current'], envs[0].data.qpos[addresses], atol=1e-8, rtol=0):
            raise ValueError('log does not start at model reset; full initial dynamic state is unavailable')
        config = FilterConfig(**header['config'])
        model.broadphase_distance = config.activation_distance
        if config.max_step_rad is not None:
            max_step = np.minimum(max_step, config.max_step_rad)
        if not np.allclose(max_step, header['max_step'], atol=1e-8, rtol=0):
            raise ValueError('step limits differ from recording')
        safety = CBFSafetyFilter(model, low, high, max_step, config)
        safety.reset(envs[1].data.qpos[addresses])
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(envs[0].model, camera)
        azimuth, elevation = (90., 0.) if args.view == 'front' else (135., -25.)
        camera.azimuth = azimuth if args.azimuth is None else args.azimuth
        camera.elevation = elevation if args.elevation is None else args.elevation
        if args.camera_distance is not None:
            if args.camera_distance <= 0:
                raise ValueError('camera distance must be positive')
            camera.distance = args.camera_distance
        for env in envs:
            renderers.append(mujoco.Renderer(env.model, args.height, args.width))
        args.output.mkdir(parents=True)
        for name, width in [('OFF', args.width), ('ON', args.width), ('OFF-ON', args.width*2)]:
            writer = cv2.VideoWriter(str(args.output/f'{name}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                     args.fps, (width, args.height))
            writers.append(writer)
            if not writer.isOpened():
                raise RuntimeError('OpenCV MP4 encoder unavailable')
        elapsed, written = 0., 0
        last = int(np.flatnonzero(selected)[-1])
        for i in range(last+1):
            # Replay hidden preceding steps too, preserving each branch history.
            safe, info = safety.filter(envs[1].data.qpos[addresses], q[i])
            envs[0].step(q[i]); envs[1].step(safe)
            if selected[i]:
                elapsed += dt/args.speed
                target_count = int(np.ceil(elapsed*args.fps-1e-9))
                if target_count > written:
                    images = []
                    for mode, env, renderer in zip(['OFF', 'ON'], envs, renderers):
                        renderer.update_scene(env.data, camera=camera)
                        image = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                        cv2.rectangle(image, (0, 0), (args.width, 64), (25, 25, 25), -1)
                        cv2.putText(image, f'Safety {mode} | frame {frames[i]["frame"]}', (12, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1, cv2.LINE_AA)
                        label = f'recorded {wall[i]:.2f}s | sim {sim[i]:.2f}s'
                        if mode == 'ON' and info.get('fallback'):
                            label += ' | HOLD'
                        cv2.putText(image, label, (12, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    .48, (255, 255, 255), 1, cv2.LINE_AA)
                        images.append(image)
                    for _ in range(target_count-written):
                        for writer, image in zip(writers, images+[np.concatenate(images, axis=1)]):
                            writer.write(image)
                    written = target_count
            if (i+1) % 100 == 0:
                print(f'Replayed {i+1}/{last+1} steps', flush=True)
        (args.output/'README.md').write_text(f'''# 真人动作 OFF/ON 视频

- `OFF.mp4`：原始 nominal 直接交给仿真（MuJoCo 执行器仍保留模型自身的控制范围裁剪）。
- `ON.mp4`：相同 nominal 经 CBF-QP 安全层后执行；`HOLD` 表示该步 fallback。
- `OFF-ON.mp4`：左右并排，左 OFF，右 ON。

两边使用相同模型 reset 状态、目标序列、仿真步长和视角。Safety ON 重新计算，不播放采集日志中的 q_safe。保留显示前的历史回放，仅剪掉画面，不跳过仿真步骤。

保留 {int(selected.sum())} 步；播放速度为仿真时间的 {args.speed:g} 倍，视频 {written/args.fps:.2f} 秒，{args.fps} fps。片段筛选范围（{'摄像头视频秒数' if args.video_timing else '相对首条帧日志的现实秒数'}）：{args.clip or '全部'}。视频编码对时长的取整误差最多一帧。共同视角：azimuth={camera.azimuth:g}°、elevation={camera.elevation:g}°。

`recorded` 是执行后日志时间；`sim` 是仿真累计时间。省略现实计算等待，按仿真时间播放。多个片段按原采集顺序拼接，切换时可能出现画面跳转。复现的是执行端收到的动作，不含人手视频，也不等同于两次真人闭环遥操作。

使用 --video-timing 时，通过 UTC 起点换算摄像头视频与动作日志时间，属于近似对齐，包含识别、队列与执行延迟，不是逐帧手势同步。
''')
        print(f'Videos saved: {args.output} ({written/args.fps:.2f}s)', flush=True)
    finally:
        for writer in writers: writer.release()
        for renderer in renderers: renderer.close()
        for env in envs: env.close()


if __name__ == '__main__':
    main()
