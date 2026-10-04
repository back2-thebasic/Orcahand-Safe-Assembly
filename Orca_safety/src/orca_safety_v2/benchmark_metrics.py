"""Six benchmark metrics and exports; no collision or safety implementation."""
from collections import Counter
import csv
import json
import os
from pathlib import Path

import numpy as np

CORE_METRICS = (
    'minimum_distance_mm', 'collision_rate', 'margin_violation_rate',
    'mean_modification_rad', 'fallback_rate', 'safety_runtime_p95_ms',
)


def collect_metrics(rows, mode):
    if not rows:
        raise ValueError('Cannot summarize zero scored frames')
    distances = np.asarray([r['min_distance_measured'] for r in rows], dtype=float)
    modifications = np.asarray([
        np.linalg.norm(np.asarray(r['q_safe']) - np.asarray(r['q_nominal'])) for r in rows
    ], dtype=float)
    if not np.isfinite(distances).all() or not np.isfinite(modifications).all():
        raise ValueError('Non-finite benchmark measurements')
    n = len(rows)
    collisions = sum(bool(r['measured_collision']) for r in rows)
    violations = sum(bool(r['measured_margin_violation']) for r in rows)
    fallbacks = sum(bool(r['fallback']) for r in rows)
    runtime = None
    if mode == 'on':
        timings = np.asarray([r['total_safety_time_ms'] for r in rows], dtype=float)
        if not np.isfinite(timings).all() or np.any(timings < 0):
            raise ValueError('Invalid safety timing measurements')
        runtime = float(np.percentile(timings, 95))
    return {
        'scored_frames': n,
        'minimum_distance_mm': float(distances.min() * 1000),
        'collision_rate': collisions / n, 'collision_frames': collisions,
        'margin_violation_rate': violations / n, 'margin_violation_frames': violations,
        'mean_modification_rad': float(modifications.mean()),
        'fallback_rate': fallbacks / n, 'fallback_frames': fallbacks,
        'safety_runtime_p95_ms': runtime,
    }


def fallback_reasons(rows):
    return dict(Counter(r.get('error', 'unknown') for r in rows if r['fallback']))


class FrameCSV:
    """All executed frames, including hidden history, with an explicit scored flag."""
    def __init__(self, path, joint_names):
        self.stream = Path(path).open('x', newline='', encoding='utf-8')
        self.names = list(joint_names)
        fields = ['frame', 'scored', 'sim_time', 'min_distance_measured_mm',
                  'measured_collision', 'measured_margin_violation', 'modification_rad',
                  'fallback', 'total_safety_time_ms']
        fields += [f'{key}.{name}' for key in ('q_nominal', 'q_safe', 'q_measured') for name in self.names]
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()

    def write(self, info, scored):
        row = {key: info[key] for key in ('frame', 'sim_time', 'measured_collision',
                                         'measured_margin_violation', 'fallback')}
        row.update(scored=scored,
                   min_distance_measured_mm=info['min_distance_measured'] * 1000,
                   modification_rad=float(np.linalg.norm(np.asarray(info['q_safe']) - info['q_nominal'])),
                   total_safety_time_ms=info.get('total_safety_time_ms'))
        for key in ('q_nominal', 'q_safe', 'q_measured'):
            row.update({f'{key}.{name}': value for name, value in zip(self.names, info[key], strict=True)})
        self.writer.writerow(row)

    def close(self):
        self.stream.close()


def export_results(output, summary):
    output = Path(output)
    # Replace complete snapshots, so readers never see half-written JSON.
    temp = output / 'benchmark_summary.json.tmp'
    temp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(output / 'benchmark_summary.json')
    fields = ['scenario', 'mode', 'status', 'executed_frames', 'scored_frames',
              'safe_distance_mm', 'control_dt_ms', *CORE_METRICS,
              'collision_frames', 'margin_violation_frames', 'fallback_frames',
              'nominal_sha256', 'error']
    with (output / 'benchmark_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for scenario in summary['scenarios']:
            for mode, run in scenario['runs'].items():
                writer.writerow({
                    'scenario': scenario['name'], 'mode': mode, 'status': run['status'],
                    'executed_frames': run['executed_frames'],
                    'safe_distance_mm': scenario['safe_distance_mm'],
                    'control_dt_ms': scenario['control_dt_ms'],
                    'nominal_sha256': run.get('nominal_sha256'), 'error': run.get('error'),
                    **run.get('metrics', {}),
                })
    (output / 'benchmark_report.md').write_text(render_report(summary, output), encoding='utf-8')


def scenario_introductions(summary, output=None):
    """Render only selected scenarios and existing, explicitly labelled pictures."""
    if not summary['scenarios']:
        return []
    lines = ['', '## 动作介绍与已有仿真图片', '',
             '以下图片是既有几何验证的静态目标姿态渲染，不是本次 OFF/ON 动态执行截图。'
             '默认 benchmark 不生成截图。未配置图片的场景仅展示介绍。', '']
    for scenario in summary['scenarios']:
        source = scenario['source']
        presentation = source.get('presentation', {})
        description = presentation.get('description')
        if not description:
            description = {'jsonl': '重放已有 JSONL 中的原始 q_nominal。',
                           'npz': '重放已有 NPZ 中的原始 q_nominal。',
                           'pose_suite': '使用已有目标姿态生成张手→目标→张手轨迹。'}.get(source.get('type'), '已注册目标轨迹。')
        counts = [run.get('metrics', {}).get('scored_frames') for run in scenario['runs'].values()]
        counts = [n for n in counts if n is not None]
        lines += [f"### {scenario['name']}", '', description, '']
        path = Path(source['path'])
        target = os.path.relpath(path, output) if output is not None else str(path)
        lines += [f'数据来源：[{path.name}](<{target}>)。', '']
        if counts:
            n = counts[0]
            duration = n * scenario['control_dt_ms'] / 1000
            lines += [f'统计 {n} 个控制帧，合计 {duration:.2f} 秒仿真时间（不是录制墙钟时间）。', '']
        image = presentation.get('image')
        if image:
            path = Path(image['path'])
            target = os.path.relpath(path, output) if output is not None else str(path)
            lines += [f"![{scenario['name']}：已有静态目标姿态渲染](<{target}>)", '']
    return lines


def render_report(summary, output=None):
    lines = [
        '# Retargeting + Safety v2 Benchmark', '',
        f"运行状态：**{summary['status']}**；机器人：v1 右手；安全软件：orca_safety_v2。", '',
        '每个场景的 OFF/ON 使用同一份 nominal 轨迹。ON 重新执行生产安全路径；'
        'OFF 仅监测，保留 MuJoCo 目标限幅和接触动力学。', '',
        '## 六个核心指标', '',
        '| 场景 | 模式 | 最小实际距离 mm | 碰撞率（帧数） | 裕量违规率（帧数） | 平均修改量 rad | Fallback 率（帧数） | 安全耗时 p95 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    def rate(m, key, count):
        return f"{m[key]*100:.2f}%（{m[count]}/{m['scored_frames']}）"
    for scenario in summary['scenarios']:
        for mode, run in scenario['runs'].items():
            m = run.get('metrics')
            if not m:
                lines.append(f"| {scenario['name']} | {mode.upper()}：失败 | — | — | — | — | — | — |")
                continue
            timing = 'N/A' if m['safety_runtime_p95_ms'] is None else f"{m['safety_runtime_p95_ms']:.3f}"
            lines.append(f"| {scenario['name']} | {mode.upper()} | {m['minimum_distance_mm']:.3f} | "
                         f"{rate(m, 'collision_rate', 'collision_frames')} | "
                         f"{rate(m, 'margin_violation_rate', 'margin_violation_frames')} | "
                         f"{m['mean_modification_rad']:.6f} | {rate(m, 'fallback_rate', 'fallback_frames')} | {timing} |")
    lines += ['', '## OFF/ON 比较与有效配置', '']
    for scenario in summary['scenarios']:
        pair = scenario.get('comparison', {})
        prefix = (f"- **{scenario['name']}**：安全距离 {scenario['safe_distance_mm']:.3f} mm；"
                  f"控制周期 {scenario['control_dt_ms']:.3f} ms。")
        if pair.get('fair'):
            delta = pair['on_minus_off']
            lines.append(prefix + '输入、初态、关节顺序、配置和周期一致；'
                         f"最小距离变化 {delta['minimum_distance_mm']:+.3f} mm，"
                         f"碰撞率变化 {delta['collision_rate']*100:+.2f} 个百分点，"
                         f"裕量违规率变化 {delta['margin_violation_rate']*100:+.2f} 个百分点。")
        else:
            lines.append(prefix + pair.get('reason', '单模式运行，无成对比较。'))
    abnormal = [(s['name'], mode, r) for s in summary['scenarios'] for mode, r in s['runs'].items()
                if r.get('error') or r.get('fallback_reasons')]
    if abnormal:
        lines += ['', '## 异常与回退原因', '']
        for name, mode, run in abnormal:
            details = run.get('error') or json.dumps(run['fallback_reasons'], ensure_ascii=False)
            lines.append(f'- {name} / {mode.upper()}：{details.replace(chr(10), " ")}')
    lines += scenario_introductions(summary, output)
    lines += [
        '', '## 指标口径', '',
        '- 最小距离来自执行后的实际状态；裕量违规使用实际距离严格小于配置安全距离。',
        '- 碰撞率、裕量违规率和 fallback 率以纳入统计的控制帧为分母；原始帧数同时展示。',
        '- 平均修改量为 mean(||q_safe − q_nominal||₂)，使用弧度和全部 17 个关节。',
        '- p95 使用生产 total_safety_time_ms；不包含仿真推进、执行后测量和日志 I/O。OFF 为 N/A。',
        '- clip 前置历史仍执行并记录；每帧 CSV 的 scored 列标明是否纳入统计。',
        '- QP、缩步、非线性复核等完整诊断保留在 runs/*.jsonl，不作为额外主指标。',
        '- 当前测量仅覆盖配置碰撞对和控制步采样，不测连续时间碰撞、可信穿透深度或任务成功率。',
        '- 状态 complete 表示实验执行和公平性检查完成，不表示所有场景无碰撞或已通过安全认证。',
        '',
    ]
    return '\n'.join(lines)
