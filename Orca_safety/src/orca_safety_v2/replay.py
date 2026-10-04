"""Validate recorded commands and reconstruct a branch for interactive viewing."""
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from orca_sim import OrcaHandRight

from .adapter import build_sim_model
from .config import FilterConfig
from .filter import CBFSafetyFilter


@dataclass
class Recording:
    header: dict
    frames: list
    nominal: np.ndarray
    wall_seconds: np.ndarray
    sim_seconds: np.ndarray
    selected: np.ndarray


def load_recording(path, clips=None, video_timing=None):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    headers = [r for r in rows if r.get('event') == 'start']
    frames = [r for r in rows if r.get('event') == 'frame']
    if len(headers) != 1 or len(frames) < 2:
        raise ValueError('need one start record and at least two frames')
    header = headers[0]
    if header.get('robot') != 'v1_right' or header.get('units', {}).get('angle') != 'rad':
        raise ValueError('expected v1_right absolute-radian recording')
    nominal = np.asarray([r['q_nominal'] for r in frames], dtype=float)
    if nominal.shape != (len(frames), len(header['joint_names'])) or not np.isfinite(nominal).all():
        raise ValueError('invalid nominal commands')
    times = [datetime.fromisoformat(r['timestamp']) for r in frames]
    wall = np.asarray([(t-times[0]).total_seconds() for t in times])
    sim = np.asarray([r['sim_time'] for r in frames])
    if not np.isfinite(sim).all() or np.any(np.diff(wall) < 0) or np.any(np.diff(sim) <= 0):
        raise ValueError('unordered or invalid timestamps')
    offset = 0.
    if video_timing:
        info = json.loads(Path(video_timing).read_text())
        offset = (datetime.fromisoformat(info['start_timestamp'])-times[0]).total_seconds()
    selected = np.ones(len(frames), dtype=bool) if not clips else np.zeros(len(frames), dtype=bool)
    for start, end in clips or []:
        if not np.isfinite([start, end]).all() or not 0 <= start < end:
            raise ValueError('invalid clip range')
        selected |= (wall >= start+offset) & (wall < end+offset)
    if not selected.any():
        raise ValueError('clips contain no control steps')
    return Recording(header, frames, nominal, wall, sim, selected)


def validate_recording_model(recording, env, model, low, high, max_step, addresses):
    """Reuse replay compatibility checks without reconstructing or stepping a branch."""
    header = recording.header
    if list(model.joint_names) != header['joint_names'] or list(model.pair_indices) != header['pairs']:
        raise ValueError('joint order or collision pairs differ; use matching --config')
    for field, actual in [('low', low), ('high', high), ('reference_offsets', model.offsets)]:
        if not np.allclose(actual, header[field], atol=1e-8, rtol=0):
            raise ValueError(f'model {field} differs from recording')
    dt = float(env.model.opt.timestep*env.frame_skip)
    if not np.allclose(np.diff(recording.sim_seconds), dt, atol=1e-8, rtol=0):
        raise ValueError('recorded control-step timing differs from model')
    if not np.allclose(recording.frames[0]['q_current'], env.data.qpos[addresses], atol=1e-8, rtol=0):
        raise ValueError('recording does not start at reset; full initial dynamic state unavailable')
    config = FilterConfig(**header['config'])
    model.broadphase_distance = config.activation_distance
    if config.max_step_rad is not None:
        max_step = np.minimum(max_step, config.max_step_rad)
    if not np.allclose(max_step, header['max_step'], atol=1e-8, rtol=0):
        raise ValueError('step limits differ from recording')
    return config, max_step


def prepare_replay(recording, mode, config_path=None, progress=None):
    """Cache physical states, simulating hidden history before selected frames.

    The returned environment stays open for viewing; the caller must close it.
    Playback restores cached states rather than changing the safety timestep.
    """
    if mode not in ('OFF', 'ON'):
        raise ValueError('mode must be OFF or ON')
    env = OrcaHandRight(version='v1')
    try:
        env.reset()
        model, _, low, high, max_step, addresses = build_sim_model(env, config_path)
        config, max_step = validate_recording_model(
            recording, env, model, low, high, max_step, addresses)
        dt = float(env.model.opt.timestep*env.frame_skip)
        safety = CBFSafetyFilter(model, low, high, max_step, config) if mode == 'ON' else None
        if safety:
            safety.reset(env.data.qpos[addresses])
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        states, indices, fallbacks = [], [], []
        last = int(np.flatnonzero(recording.selected)[-1])
        for i in range(last+1):
            target = recording.nominal[i]
            fallback = False
            if safety:
                target, info = safety.filter(env.data.qpos[addresses], target)
                fallback = info['fallback']
            env.step(target)
            if recording.selected[i]:
                state = np.empty(mujoco.mj_stateSize(env.model, spec))
                mujoco.mj_getState(env.model, env.data, state, spec)
                states.append(state); indices.append(i); fallbacks.append(fallback)
            if progress and ((i+1) % 100 == 0 or i == last):
                progress(i+1, last+1)
        return env, np.asarray(states), np.asarray(indices), np.asarray(fallbacks), dt
    except BaseException:
        env.close()
        raise
