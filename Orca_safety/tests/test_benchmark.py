"""Benchmark contracts against the actual production executor and MuJoCo model."""
from datetime import datetime, timedelta, timezone
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from orca_sim import OrcaHandRight

from orca_safety_v2.adapter import build_sim_model
from orca_safety_v2.benchmark import load_scenarios, run_benchmark
from orca_safety_v2.benchmark_metrics import CORE_METRICS, collect_metrics
from orca_safety_v2.config import FilterConfig
from orca_safety_v2.teleop import SimulationCollisionSafety


@pytest.fixture
def inputs(tmp_path):
    env = OrcaHandRight(version='v1')
    try:
        env.reset()
        model, _, low, high, step, adr = build_sim_model(env)
        q0 = env.data.qpos[adr].copy()
        q = np.tile(q0, (6, 1))
        q[:, 0] += np.arange(6) * .002
        np.savez(tmp_path/'motion.npz', q_nominal=q, joint_names=model.joint_names)
        config = FilterConfig(safe_distance=.004, max_step_rad=.03, nonlinear_backtracking_steps=2)
        rows = [dict(event='start', robot='v1_right', units={'angle': 'rad'},
                     joint_names=list(model.joint_names), pairs=list(model.pair_indices),
                     config=vars(config), low=low.tolist(), high=high.tolist(),
                     max_step=np.minimum(step, .03).tolist(), reference_offsets=model.offsets.tolist())]
        epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i, nominal in enumerate(q):
            rows.append(dict(event='frame', frame=i, q_nominal=nominal.tolist(),
                             q_current=q0.tolist(), q_safe=[999.]*17,
                             timestamp=(epoch+timedelta(seconds=i)).isoformat(), sim_time=.01*(i+1)))
        (tmp_path/'recording.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest = dict(version=1, scenarios=[
            dict(name='npz_motion', type='npz', path='motion.npz', units='rad', control_dt_s=.01),
            dict(name='recorded', type='jsonl', path='recording.jsonl'),
            dict(name='clip', type='jsonl', path='recording.jsonl', clips=[[2, 4]]),
        ])
        config_path = tmp_path/'benchmark.yaml'
        config_path.write_text(yaml.safe_dump(manifest))
        return config_path, q
    finally:
        env.close()


def test_six_metrics_have_exact_definitions():
    rows = [dict(min_distance_measured=d, q_nominal=[0., 0.], q_safe=q,
                 measured_collision=c, measured_margin_violation=v, fallback=f,
                 total_safety_time_ms=t)
            for d, q, c, v, f, t in [(.006, [0., 0.], False, False, False, 1.),
                                     (-.001, [3., 4.], True, True, True, 9.)]]
    m = collect_metrics(rows, 'on')
    assert set(CORE_METRICS) <= m.keys() and len(CORE_METRICS) == 6
    assert m['minimum_distance_mm'] == -1
    assert m['collision_frames'] == m['margin_violation_frames'] == m['fallback_frames'] == 1
    assert m['collision_rate'] == m['margin_violation_rate'] == m['fallback_rate'] == .5
    assert m['mean_modification_rad'] == 2.5
    assert m['safety_runtime_p95_ms'] == pytest.approx(8.6)
    assert collect_metrics(rows, 'off')['safety_runtime_p95_ms'] is None
    with pytest.raises(ValueError):
        collect_metrics([], 'on')


def test_both_uses_production_dispatch_and_exact_same_inputs(inputs, tmp_path, monkeypatch):
    path, nominal = inputs
    calls = {'off': [], 'on': []}
    dispatch = SimulationCollisionSafety.dispatch
    def observed(self, q):
        calls['off' if self.monitor_only else 'on'].append(q.copy())
        return dispatch(self, q)
    monkeypatch.setattr(SimulationCollisionSafety, 'dispatch', observed)
    output = tmp_path/'results'
    result = run_benchmark(path, output, ['npz_motion'], progress=None)
    assert result['status'] == 'complete'
    scenario = result['scenarios'][0]
    assert scenario['comparison']['fair'] is True
    for mode in ('off', 'on'):
        np.testing.assert_array_equal(calls[mode], nominal)
        rows = [json.loads(line) for line in (output/f'runs/npz_motion_{mode}.jsonl').read_text().splitlines()]
        np.testing.assert_array_equal([r['q_nominal'] for r in rows[1:]], nominal)
        with (output/f'runs/npz_motion_{mode}.csv').open() as stream:
            frames = list(csv.DictReader(stream))
        assert len(frames) == 6 and all(r['scored'] == 'True' for r in frames)
        assert scenario['runs'][mode]['metrics']['scored_frames'] == 6
    assert scenario['runs']['off']['metrics']['mean_modification_rad'] == 0
    assert scenario['runs']['off']['metrics']['safety_runtime_p95_ms'] is None
    assert json.loads((output/'benchmark_summary.json').read_text())['status'] == 'complete'
    with (output/'benchmark_results.csv').open() as stream:
        summary_rows = list(csv.DictReader(stream))
    assert len(summary_rows) == 2 and summary_rows[0]['safety_runtime_p95_ms'] == ''
    assert 'N/A' in (output/'benchmark_report.md').read_text()
    with pytest.raises(FileExistsError):
        run_benchmark(path, output, ['npz_motion'], progress=None)


def test_recorded_parameters_and_clip_history_are_preserved(inputs, tmp_path):
    path, _ = inputs
    result = run_benchmark(path, tmp_path/'results', ['recorded', 'clip'], progress=None)
    assert result['status'] == 'complete'
    full, clip = result['scenarios']
    assert full['effective_config']['safe_distance'] == .004
    assert full['effective_config']['max_step_rad'] == .03
    assert full['effective_config']['nonlinear_backtracking_steps'] == 2
    for mode in ('off', 'on'):
        assert clip['runs'][mode]['executed_frames'] == 4
        assert clip['runs'][mode]['metrics']['scored_frames'] == 2
        def frames(name):
            rows = (tmp_path/f'results/runs/{name}_{mode}.jsonl').read_text().splitlines()
            return [json.loads(row) for row in rows[1:]]
        all_rows, clip_rows = frames('recorded'), frames('clip')
        for i in range(4):
            for key in ('q_safe', 'q_measured', 'q_current'):
                np.testing.assert_array_equal(clip_rows[i][key], all_rows[i][key])
        assert max(abs(v) for r in clip_rows for v in r['q_safe']) < 2
        header = json.loads((tmp_path/f'results/runs/clip_{mode}.jsonl').read_text().splitlines()[0])
        assert header['config'] == full['effective_config']
        np.testing.assert_allclose(header['max_step'], .03)


@pytest.mark.parametrize('mode', ['off', 'on'])
def test_single_mode_and_discovery(inputs, tmp_path, mode):
    path, _ = inputs
    assert run_benchmark(path, tmp_path/'unused', list_only=True) == ['npz_motion', 'recorded', 'clip']
    assert not (tmp_path/'unused').exists()
    result = run_benchmark(path, tmp_path/mode, ['npz_motion'], mode, progress=None)
    assert result['status'] == 'complete'
    scenario = result['scenarios'][0]
    assert list(scenario['runs']) == [mode]
    assert scenario['comparison']['fair'] is None


def test_bad_input_rejected_before_creating_output(inputs, tmp_path):
    path, _ = inputs
    with pytest.raises(ValueError, match='Unknown scenario'):
        run_benchmark(path, tmp_path/'unknown', ['missing'], progress=None)
    assert not (tmp_path/'unknown').exists()
    rows = [json.loads(line) for line in (tmp_path/'recording.jsonl').read_text().splitlines()]
    rows[0]['joint_names'].reverse()
    (tmp_path/'recording.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='joint order'):
        run_benchmark(path, tmp_path/'invalid', progress=None)
    assert not (tmp_path/'invalid').exists()


def test_failed_branch_is_reported_and_other_scenarios_continue(inputs, tmp_path, monkeypatch):
    path, _ = inputs
    dispatch = SimulationCollisionSafety.dispatch
    def fail_once(self, q):
        if self.filter is not None and not getattr(fail_once, 'failed', False):
            fail_once.failed = True
            raise RuntimeError('injected benchmark failure')
        return dispatch(self, q)
    monkeypatch.setattr(SimulationCollisionSafety, 'dispatch', fail_once)
    result = run_benchmark(path, tmp_path/'results', ['npz_motion', 'recorded'], progress=None)
    assert result['status'] == 'error'
    bad, good = result['scenarios']
    assert bad['runs']['on']['status'] == 'error'
    assert bad['comparison']['fair'] is False
    assert good['comparison']['fair'] is True
    assert 'injected benchmark failure' in (tmp_path/'results/benchmark_report.md').read_text()


def test_default_manifest_registers_all_existing_inputs():
    path = Path(__file__).resolve().parents[1]/'configs/benchmark.yaml'
    scenarios, _ = load_scenarios(path)
    assert len(scenarios) == 12
    assert {'intentional_collision', 'adaptive_analytical_replay', 'index_middle_crossing',
            'retarget-session-01', 'retarget-session-02', 'retarget-session-03'} <= {s.name for s in scenarios}
    assert all(not s.nominal.flags.writeable for s in scenarios)


def test_report_images_and_sources_resolve_from_custom_output(tmp_path):
    import re
    from orca_safety_v2.benchmark_metrics import scenario_introductions
    source = tmp_path/'fixtures/poses.json'
    image = tmp_path/'fixtures/open.png'
    output = tmp_path/'custom/deep/result'
    summary = {'scenarios': [
        {'name': 'open', 'control_dt_ms': 10,
         'source': {'type': 'pose_suite', 'path': str(source),
                    'presentation': {'description': '保持张手姿态。', 'image': {'path': str(image)}}},
         'runs': {'on': {'metrics': {'scored_frames': 120}}}},
        {'name': 'recorded', 'control_dt_ms': 10,
         'source': {'type': 'jsonl', 'path': str(tmp_path/'recording.jsonl')}, 'runs': {}},
    ]}
    report = '\n'.join(scenario_introductions(summary, output))
    assert '保持张手姿态。' in report and '1.20 秒仿真时间' in report
    assert '不是本次 OFF/ON 动态执行截图' in report
    links = re.findall(r'\]\(<([^>]+)>\)', report)
    assert {(output/path).resolve() for path in links} == {source, image, tmp_path/'recording.jsonl'}
    assert report.count('![') == 1
    assert '### recorded' in report
