"""Headless OFF/ON benchmark using the production v2 safety executor."""
import argparse
import logging
from pathlib import Path

from orca_safety_v2.benchmark import run_benchmark


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root/'configs/benchmark.yaml')
    parser.add_argument('--output', type=Path, default=root/'results/safety_benchmark')
    parser.add_argument('--scenario', action='append', help='Exact registered name; repeat to select several')
    parser.add_argument('--mode', choices=['off', 'on', 'both'], default='both')
    parser.add_argument('--list-scenarios', action='store_true', help='Validate inputs and list names without running')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    try:
        result = run_benchmark(args.config, args.output, args.scenario, args.mode,
                               list_only=args.list_scenarios,
                               progress=lambda message: print(message, flush=True))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    if args.list_scenarios:
        for name in result:
            print(name)
        return 0
    print(f'Report: {args.output.resolve() / "benchmark_report.md"}', flush=True)
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
