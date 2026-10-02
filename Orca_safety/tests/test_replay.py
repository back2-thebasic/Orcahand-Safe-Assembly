"""Replay preserves branch history when only a later clip is displayed."""
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest
from orca_sim import OrcaHandRight
from orca_safety_v2.adapter import build_sim_model
from orca_safety_v2.replay import load_recording, prepare_replay


@pytest.fixture
def recording_path(tmp_path):
    env = OrcaHandRight(version='v1'); env.reset()
    try:
        model, config, low, high, step, addresses = build_sim_model(env)
        q0 = env.data.qpos[addresses].copy()
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = [dict(event='start', robot='v1_right', units={'angle': 'rad'},
                     joint_names=list(model.joint_names), pairs=list(model.pair_indices),
                     config=vars(config), low=low.tolist(), high=high.tolist(),
                     max_step=step.tolist(), reference_offsets=model.offsets.tolist())]
        for i in range(6):
            q = q0.copy(); q[0] += .01*i
            rows.append(dict(event='frame', frame=i, q_nominal=q.tolist(),
                             q_current=q0.tolist(), q_safe=[999.]*len(q0),
                             timestamp=(start+timedelta(seconds=i)).isoformat(), sim_time=.01*(i+1)))
        path = tmp_path/'recording.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path
    finally:
        env.close()


@pytest.mark.parametrize('mode', ['OFF', 'ON'])
def test_clip_preserves_hidden_history(recording_path, mode):
    whole = load_recording(recording_path)
    clip = load_recording(recording_path, clips=[(3, 5)])
    full_env, full_states, _, _, _ = prepare_replay(whole, mode)
    try:
        clip_env, clip_states, indices, _, _ = prepare_replay(clip, mode)
        try:
            np.testing.assert_array_equal(indices, [3, 4])
            np.testing.assert_allclose(clip_states, full_states[indices], atol=1e-12, rtol=0)
            # q_safe=999 in the log is intentionally ignored in both branches.
            assert np.max(np.abs(clip_env.data.qpos)) < 2
        finally:
            clip_env.close()
    finally:
        full_env.close()


def test_webcam_clip_origin(recording_path, tmp_path):
    info = tmp_path/'camera.timing.json'
    info.write_text(json.dumps({'start_timestamp': '2026-01-01T00:00:02+00:00'}))
    clip = load_recording(recording_path, clips=[(1, 3)], video_timing=info)
    np.testing.assert_array_equal(np.flatnonzero(clip.selected), [3, 4])


def test_wrong_joint_order_rejected(recording_path):
    recording = load_recording(recording_path)
    recording.header['joint_names'] = list(reversed(recording.header['joint_names']))
    with pytest.raises(ValueError, match='joint order'):
        prepare_replay(recording, 'OFF')
