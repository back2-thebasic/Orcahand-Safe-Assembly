# Automated Tests

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

This directory checks safety-module code behavior. See [scripts/README.md](../scripts/README.md) for OFF/ON data experiments and replay.

- `test_geometry.py`: distances, joint mapping, finite-difference gradients, and distant-pair screening.
- `test_filter.py`: CBF-QP, joint/step limits, backtracking validation, and failure fallback.
- `test_integration.py`: simulation integration, startup pose synchronization, passive monitoring, and logs.
- `test_replay.py`: preserved simulation history when clipping, webcam time conversion, and joint-order validation.

Install dependencies following the [main README](../README.md), then run from the workspace root:

```bash
orca_teleop/.venv/bin/python -m pytest Orca_safety/tests -q
```

Passing tests only means the covered code behavior matches expectations. It does not guarantee continuous-motion or physical-hand safety.
