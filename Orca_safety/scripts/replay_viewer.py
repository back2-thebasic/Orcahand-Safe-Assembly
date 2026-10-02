"""Open recorded nominal commands in the interactive MuJoCo viewer, OFF or ON."""
import argparse
import logging
from pathlib import Path
import queue
import time

import mujoco
import mujoco.viewer
import numpy as np

from orca_safety_v2.replay import load_recording, prepare_replay


def clip_range(value):
    try:
        start, end = map(float, value.split(':'))
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError
        return start, end
    except ValueError:
        raise argparse.ArgumentTypeError('expected START:END seconds')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log', type=Path)
    p.add_argument('--mode', choices=['OFF', 'ON'], required=True)
    p.add_argument('--clip', type=clip_range, action='append')
    p.add_argument('--video-timing', type=Path)
    p.add_argument('--speed', type=float, default=1., help='simulation playback speed')
    p.add_argument('--view', choices=['front', 'oblique'], default='front')
    p.add_argument('--azimuth', type=float)
    p.add_argument('--elevation', type=float)
    p.add_argument('--camera-distance', type=float)
    p.add_argument('--config', type=Path)
    p.add_argument('--paused', action='store_true', help='start paused')
    p.add_argument('--check-only', action='store_true', help='validate/reconstruct without opening a window')
    p.add_argument('--exit-after-playback', action='store_true', help='close at end instead of holding last frame')
    args = p.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        p.error('speed must be positive and finite')
    if args.camera_distance is not None and (not np.isfinite(args.camera_distance) or args.camera_distance <= 0):
        p.error('camera distance must be positive and finite')
    for angle in (args.azimuth, args.elevation):
        if angle is not None and not np.isfinite(angle):
            p.error('camera angles must be finite')
    logging.basicConfig(level=logging.ERROR)
    try:
        recording = load_recording(args.log, args.clip, args.video_timing)
        print(f'Preparing Safety {args.mode}; hidden preceding steps are replayed too.', flush=True)
        env, states, indices, fallbacks, dt = prepare_replay(
            recording, args.mode, args.config,
            progress=lambda done, total: print(f'Prepared {done}/{total} steps', flush=True))
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    try:
        print(f'Ready: {len(states)} displayed steps, {len(states)*dt:.2f}s simulation time.', flush=True)
        if args.check_only:
            return
        commands = queue.SimpleQueue()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        def restore(index):
            mujoco.mj_setState(env.model, env.data, states[index], spec)
            mujoco.mj_forward(env.model, env.data)
        restore(0)
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=commands.put) as viewer:
            with viewer.lock():
                mujoco.mjv_defaultFreeCamera(env.model, viewer.cam)
                az, el = (90., 0.) if args.view == 'front' else (135., -25.)
                viewer.cam.azimuth = az if args.azimuth is None else args.azimuth
                viewer.cam.elevation = el if args.elevation is None else args.elevation
                if args.camera_distance is not None:
                    viewer.cam.distance = args.camera_distance
            print('Space: pause/play | N/right: next | B/left: previous | R: restart | close window: exit', flush=True)
            index, paused, ended = 0, args.paused, False
            deadline = time.monotonic()+dt/args.speed
            while viewer.is_running():
                while not commands.empty():
                    key = commands.get()
                    if key == 32:
                        if ended:
                            index, ended = 0, False
                        paused = not paused
                    elif key in (78, 262):
                        index = min(index+1, len(states)-1); paused = True; ended = index == len(states)-1
                    elif key in (66, 263):
                        index = max(index-1, 0); paused = True; ended = False
                    elif key == 82:
                        index, ended = 0, False
                    deadline = time.monotonic()+dt/args.speed
                now = time.monotonic()
                if not paused and now >= deadline:
                    if index+1 < len(states):
                        index += 1
                    else:
                        ended, paused = True, True
                        if args.exit_after_playback:
                            break
                    # Avoid large frame jumps if the viewer temporarily stalls.
                    deadline = now+dt/args.speed
                i = indices[index]
                with viewer.lock():
                    restore(index)
                label = 'END' if ended else ('PAUSED' if paused else 'PLAYING')
                if fallbacks[index]:
                    label += ' | HOLD'
                viewer.set_texts((None, None,
                    f'Safety {args.mode} | {label}\nframe {recording.frames[i]["frame"]} | '
                    f'{"trajectory" if recording.header.get("source", "").startswith("synthetic") else "recorded"} '
                    f'{recording.wall_seconds[i]:.2f}s | sim {recording.sim_seconds[i]:.2f}s\n'
                    'Space play/pause | N/B step | R restart', ''))
                viewer.sync()
                time.sleep(.002)
    finally:
        env.close()


if __name__ == '__main__':
    main()
