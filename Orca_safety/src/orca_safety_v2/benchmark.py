"""Thin deterministic orchestration around SimulationCollisionSafety.dispatch."""
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import re
import sys

import mujoco
import numpy as np
import yaml
from orca_sim import OrcaHandRight

from .adapter import build_sim_model
from .benchmark_metrics import CORE_METRICS, FrameCSV, collect_metrics, export_results, fallback_reasons
from .config import FilterConfig
from .replay import load_recording, validate_recording_model
from .teleop import SimulationCollisionSafety


@dataclass
class Scenario:
    name: str
    nominal: np.ndarray
    selected: np.ndarray
    config: FilterConfig
    source: dict


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_hash(q):
    array = np.ascontiguousarray(q, dtype='<f8')
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def pose_trajectory(open_q, target, steps):
    """Original compare_simulation.py open -> target -> open schedule."""
    phase = np.linspace(0, 1, steps)
    alpha = np.minimum(np.minimum(phase / .35, (1 - phase) / .35), 1)
    return open_q[None, :] + alpha[:, None] * (target - open_q)[None, :]


def _path(base, value):
    return (base / value).resolve()


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError(f'Invalid scenario name: {value!r}')
    return value


def load_scenarios(config_path):
    """Resolve only registered sources, validate all inputs before any output."""
    config_path = Path(config_path).resolve()
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict) or document.get('version') != 1:
        raise ValueError('Benchmark configuration requires version: 1')
    specs = document.get('scenarios')
    if not isinstance(specs, list) or not specs:
        raise ValueError('Register at least one scenario')
    base = config_path.parent
    safety_path = _path(base, document['safety_config']) if document.get('safety_config') else None
    env = OrcaHandRight(version='v1')
    scenarios = []
    try:
        env.reset()
        model, config, low, high, step, addresses = build_sim_model(env, safety_path)
        names = list(model.joint_names)
        dt = float(env.model.opt.timestep * env.frame_skip)
        for spec in specs:
            if spec.get('enabled', True) is False:
                continue
            kind = spec['type']
            _name(spec['name'])
            if bool(spec.get('path')) == bool(spec.get('glob')):
                raise ValueError(f"{spec['name']}: specify exactly one path or glob")
            if spec.get('glob'):
                if kind != 'jsonl':
                    raise ValueError('Only jsonl registrations support glob')
                paths = [Path(p).resolve() for p in sorted(glob.glob(str(base / spec['glob'])))]
                if not paths:
                    raise ValueError(f"No recordings matched {spec['glob']}")
            else:
                paths = [_path(base, spec['path'])]
            for path in paths:
                source = {'type': kind, 'path': str(path), 'sha256': file_hash(path)}
                if spec.get('provenance'):
                    provenance = _path(base, spec['provenance'])
                    source['provenance'] = {'path': str(provenance), 'sha256': file_hash(provenance),
                                            'data': json.loads(provenance.read_text())}
                if kind == 'jsonl':
                    timing = _path(base, spec['video_timing']) if spec.get('video_timing') else None
                    recording = load_recording(path, spec.get('clips'), timing)
                    # Use URDF-derived step bounds, then the recorded optional cap.
                    recorded_config, _ = validate_recording_model(
                        recording, env, model, low, high, model.velocity_limits * dt, addresses)
                    source.update(clips=spec.get('clips'), recorded_header=recording.header)
                    if timing:
                        source['video_timing'] = {'path': str(timing), 'sha256': file_hash(timing)}
                    name = path.stem if spec.get('glob') else spec['name']
                    scenarios.append(Scenario(name, recording.nominal, recording.selected,
                                              recorded_config, source))
                elif kind == 'npz':
                    if spec.get('units') != 'rad' or not np.isclose(spec.get('control_dt_s', -1), dt, atol=1e-8, rtol=0):
                        raise ValueError('NPZ must declare units: rad and matching control_dt_s')
                    with np.load(path, allow_pickle=False) as data:
                        if list(data['joint_names']) != names:
                            raise ValueError('Replay joint order mismatch')
                        q = np.array(data['q_nominal'], dtype=float, copy=True)
                    scenarios.append(Scenario(spec['name'], q, np.ones(len(q), dtype=bool), config, source))
                elif kind == 'pose_suite':
                    poses = {name: np.asarray(q, dtype=float) for name, q in json.loads(path.read_text()).items()}
                    q0 = poses['open']
                    # Named target overrides reuse the existing synthetic scenario definitions.
                    index = {n.removeprefix('right_'): i for i, n in enumerate(names)}
                    for name, overrides in spec.get('extra_targets', {}).items():
                        q = q0.copy()
                        for joint, value in overrides.items():
                            q[index[joint]] = value
                        poses[name] = q
                    steps = spec.get('steps', 120)
                    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
                        raise ValueError('pose_suite steps must be an integer >= 2')
                    for name, target in poses.items():
                        if q0.shape != (len(names),) or target.shape != q0.shape:
                            raise ValueError('Pose joint count mismatch')
                        q = pose_trajectory(q0, target, steps)
                        scenarios.append(Scenario(name, q, np.ones(steps, dtype=bool), config,
                                                  {**source, 'pose': name, 'steps': steps,
                                                   'target_overrides': spec.get('extra_targets', {}).get(name)}))
                else:
                    raise ValueError(f'Unknown scenario type: {kind}')
        seen = set()
        for scenario in scenarios:
            _name(scenario.name)
            if scenario.name in seen:
                raise ValueError(f'Duplicate scenario: {scenario.name}')
            seen.add(scenario.name)
            q = np.array(scenario.nominal, dtype=float, copy=True, order='C')
            if q.ndim != 2 or q.shape[1] != len(names) or len(q) < 2 or not np.isfinite(q).all():
                raise ValueError(f'{scenario.name}: expected finite N x {len(names)} nominal trajectory')
            if not scenario.selected.any():
                raise ValueError(f'{scenario.name}: no scored frames')
            # Keep every hidden preceding control step; omit only unused trailing steps.
            last = int(np.flatnonzero(scenario.selected)[-1]) + 1
            scenario.nominal = q[:last].copy()
            scenario.selected = scenario.selected[:last].copy()
            scenario.nominal.setflags(write=False)
            scenario.selected.setflags(write=False)
            note = document.get('scenario_notes', {}).get(scenario.name)
            if note:
                presentation = {'description': str(note['description'])}
                if note.get('image'):
                    image_path = _path(base, note['image'])
                    presentation['image'] = {'path': str(image_path), 'sha256': file_hash(image_path),
                                             'kind': 'static_target_pose'}
                scenario.source = {**scenario.source, 'presentation': presentation}
        if not scenarios:
            raise ValueError('No enabled scenarios')
        metadata = {'joint_names': names, 'control_dt_s': dt, 'safety_config_path': str(safety_path) if safety_path else None}
        return scenarios, metadata
    finally:
        env.close()


def _fingerprint(env, runner):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(env.model, spec))
    mujoco.mj_getState(env.model, env.data, state, spec)
    return {
        'initial_state_sha256': trajectory_hash(state),
        'joint_names': list(runner.model.joint_names), 'pairs': list(runner.model.pair_indices),
        'config': vars(runner.config), 'control_dt_s': float(env.model.opt.timestep * env.frame_skip),
        'low': runner.low.tolist(), 'high': runner.high.tolist(),
        'max_step': runner.max_step.tolist(), 'reference_offsets': runner.model.offsets.tolist(),
    }


def run_branch(scenario, mode, output, safety_path, progress=print):
    env = runner = csv_log = None
    run = {'status': 'running', 'executed_frames': 0,
           'expected_nominal_sha256': trajectory_hash(scenario.nominal),
           'jsonl': f'runs/{scenario.name}_{mode}.jsonl', 'csv': f'runs/{scenario.name}_{mode}.csv'}
    actual_inputs, scored = [], []
    try:
        env = OrcaHandRight(version='v1')
        env.reset()
        runner = SimulationCollisionSafety(
            env, config_path=safety_path, log_path=output / run['jsonl'],
            monitor_only=mode == 'off', filter_config=scenario.config)
        run['fingerprint'] = _fingerprint(env, runner)
        csv_log = FrameCSV(output / run['csv'], runner.model.joint_names)
        for i, nominal in enumerate(scenario.nominal):
            # Dispatch receives a fresh copy; neither branch can mutate shared inputs.
            expected = nominal.copy()
            info = runner.dispatch(expected.copy())
            actual = np.asarray(info['q_nominal'], dtype=float)
            if not np.array_equal(actual, expected):
                raise ValueError(f'Executor nominal differs at frame {i}')
            actual_inputs.append(actual.copy())
            run['executed_frames'] += 1
            csv_log.write(info, bool(scenario.selected[i]))
            if scenario.selected[i]:
                scored.append(info)
            if progress and ((i + 1) % 100 == 0 or i + 1 == len(scenario.nominal)):
                progress(f'{scenario.name} {mode.upper()}: {i+1}/{len(scenario.nominal)} frames')
        run['nominal_sha256'] = trajectory_hash(actual_inputs)
        if run['nominal_sha256'] != run['expected_nominal_sha256']:
            raise ValueError('Executed nominal hash differs from registered input')
        run.update(status='complete', metrics=collect_metrics(scored, mode),
                   fallback_reasons=fallback_reasons(scored))
    except Exception as exc:
        run.update(status='error', error=f'{type(exc).__name__}: {exc}')
    finally:
        try:
            if csv_log is not None:
                csv_log.close()
            if runner is not None:
                runner.close()
        finally:
            if env is not None:
                env.close()
    return run


def compare_runs(runs):
    if not {'off', 'on'} <= runs.keys():
        return {'fair': None, 'reason': '单模式运行，无成对比较。'}
    off, on = runs['off'], runs['on']
    if any(r['status'] != 'complete' for r in (off, on)):
        return {'fair': False, 'reason': '至少一个分支未完成，结果不可比较。'}
    if off['nominal_sha256'] != on['nominal_sha256'] or off['fingerprint'] != on['fingerprint']:
        return {'fair': False, 'reason': '输入或初态/配置/周期不一致，结果不可比较。'}
    return {'fair': True, 'on_minus_off': {
        k: None if off['metrics'][k] is None or on['metrics'][k] is None
        else on['metrics'][k] - off['metrics'][k] for k in CORE_METRICS
    }}


def environment_metadata(safety_path):
    root = Path(__file__).resolve().parents[2]
    workspace = root.parent
    source_files = list((root / 'src/orca_safety_v2').rglob('*.py'))
    source_files += list((workspace / 'orca_sim/src/orca_sim').glob('*.py'))
    # Fingerprint all resources of the actual v1 model (includes referenced meshes).
    resources = [p for p in (workspace / 'orcahand_description/v1').rglob('*')
                 if p.is_file() and p.suffix.lower() in {'.xml', '.mjcf', '.urdf', '.stl'}]
    if safety_path:
        resources.append(Path(safety_path))
    packages = {}
    for name in ('numpy', 'mujoco', 'pin', 'coal', 'osqp', 'scipy', 'PyYAML', 'orca-core', 'orca-sim'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    imports = {name: importlib.util.find_spec(name).origin
               for name in ('orca_core', 'orca_sim', 'orca_safety_v2')}
    return {'python': sys.version, 'executable': sys.executable, 'packages': packages,
            'import_paths': imports,
            'source_sha256': {str(p.relative_to(workspace)): file_hash(p) for p in sorted(source_files)},
            'model_resource_sha256': {str(p.relative_to(workspace)) if p.is_relative_to(workspace) else str(p): file_hash(p)
                                       for p in sorted(set(resources))}}


def run_benchmark(config_path, output, selected_names=None, mode='both', list_only=False, progress=print):
    if mode not in ('off', 'on', 'both'):
        raise ValueError('mode must be off, on or both')
    scenarios, metadata = load_scenarios(config_path)
    if selected_names:
        unknown = set(selected_names) - {s.name for s in scenarios}
        if unknown:
            raise ValueError('Unknown scenario(s): ' + ', '.join(sorted(unknown)))
        scenarios = [s for s in scenarios if s.name in selected_names]
    if list_only:
        return [s.name for s in scenarios]
    output = Path(output).resolve()
    # Never overwrite previous evidence or append to an incomplete run.
    output.mkdir(parents=True, exist_ok=False)
    (output / 'runs').mkdir()
    modes = ['off', 'on'] if mode == 'both' else [mode]
    summary = {'schema_version': 1, 'status': 'running', 'robot': 'v1_right',
               'software': 'orca_safety_v2', 'started_at': datetime.now(timezone.utc).isoformat(),
               'benchmark_config': {'path': str(Path(config_path).resolve()), 'sha256': file_hash(config_path),
                                    'data': yaml.safe_load(Path(config_path).read_text())},
               'environment': environment_metadata(metadata['safety_config_path']),
               'rate_units': 'fraction (0..1); Markdown displays percent',
               'scenarios': []}
    export_results(output, summary)
    for scenario in scenarios:
        result = {'name': scenario.name, 'source': scenario.source,
                  'nominal_sha256': trajectory_hash(scenario.nominal),
                  'selected_frames_sha256': trajectory_hash(scenario.selected.astype(float)),
                  'safe_distance_mm': scenario.config.safe_distance * 1000,
                  'control_dt_ms': metadata['control_dt_s'] * 1000,
                  'effective_config': vars(scenario.config), 'runs': {}}
        summary['scenarios'].append(result)
        for branch in modes:
            result['runs'][branch] = run_branch(scenario, branch, output,
                                               metadata['safety_config_path'], progress)
            export_results(output, summary)
        result['comparison'] = compare_runs(result['runs'])
        export_results(output, summary)
    failed = any(r['status'] != 'complete' for s in summary['scenarios'] for r in s['runs'].values())
    unfair = any(s['comparison']['fair'] is False for s in summary['scenarios'])
    summary.update(status='error' if failed or unfair else 'complete',
                   ended_at=datetime.now(timezone.utc).isoformat())
    export_results(output, summary)
    return summary
