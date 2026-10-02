# 自动化测试

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

本目录检查安全模块的代码行为；OFF/ON 数据实验和回放见 [scripts/README.md](../scripts/README.zh.md)。

- `test_geometry.py`：距离、关节映射、有限差分梯度与远距离筛查。
- `test_filter.py`：CBF-QP、关节/步长限制、缩步复核及失败回退。
- `test_integration.py`：仿真接入、启动姿态同步、被动监测与日志。
- `test_replay.py`：剪辑保留仿真历史、摄像头时间换算与关节顺序校验。

按[主 README](../README.zh.md)安装依赖后，从工作区根目录运行：

```bash
orca_teleop/.venv/bin/python -m pytest Orca_safety/tests -q
```

测试通过只表示所覆盖的代码行为符合预期，不代表连续运动或实体手的安全保证。
