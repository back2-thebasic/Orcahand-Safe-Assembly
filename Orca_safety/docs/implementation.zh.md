# 当前实现与验证

[English](implementation.md) | [中文](implementation.zh.md) | [한국어](implementation.ko.md)

安全层适用于 v1 右手 MuJoCo 仿真。接口约定见 [baseline.md](baseline.zh.md)，安装与运行见 [主 README](../README.zh.md) 和 [scripts/README.md](../scripts/README.zh.md)

## 安全层实现

- **碰撞距离与 CBF**：Pinocchio/FCL 检查 108 个配置 link 对；默认安全间隙 5 mm，距离低于 15 mm 时激活约束。远距离使用保守 AABB 下界筛查，接近时计算距离梯度
- **QP 过滤**：OSQP 在满足 CBF、关节范围和每步运动限制时，尽量保留原始目标
- **非线性复核与回退**：重新检查候选目标的距离；不合格时最多将运动减半 8 次，每次仍检查全部约束。没有合格候选或求解失败时，复用上一条通过验证的目标
- **可选模式与日志**：`--collision-safety` 开启过滤；`--collision-monitor` 只监测。JSONL 保存输入、输出、实际状态、距离及干预信息

## 后续改动

- **启动同步修复**：等待 retarget 标定时保持 MuJoCo 实际 reset 姿态，避免 core neutral 与仿真姿态不一致。候选运动逐次缩小，改善原先连续复核失败、一直保持旧目标的问题；参数 `nonlinear_backtracking_steps=0` 可关闭缩步尝试
- **可选摄像头录制**：`--record-video` 保存人手视频和时间说明文件，可按原视频时间选择回放片段
- **视频对比**：`replay_video.py` 用相同 nominal 输入生成 OFF、ON 和并排视频，支持剪辑、视角调整与慢速播放
- **交互回放**：`replay_viewer.py` 可在 MuJoCo 窗口分别观察 OFF/ON，支持自由视角、暂停及逐步查看。先计算轨迹再播放，显示速度不改变安全控制步长



## 验证与展示

- **自动化测试**：最近一次运行 26 项通过，覆盖几何、QP、失败回退、启动同步和回放历史。运行方法见 [tests/README.md](../tests/README.zh.md)
- **静态几何检查**：[passive/](../output/passive/) 保存姿态、距离和梯度核验
- **历史数据实验**：[7组合成动作对照](../output/comparison/README.zh.md)及 [retarget 回放汇总](../output/retarget-comparison/summary.json)保存 OFF/ON 结果
- **直观展示**：[4组 Demo](../output/video/demo_video/README.zh.md)包括真人采集动作与合成动作（食指-中指交叉）
## 当前边界

- 只覆盖配置的碰撞对，不覆盖同指内部及部分结构性接触；5 mm 间隙也会限制指尖接触
- 目标复核和控制步采样不保证完整连续轨迹安全，实际动态运动仍可能超调
- 网格距离查询与梯度计算仍有延迟，缩步尝试也会增加开销，尚未证明满足稳定实时遥操作要求
- OFF 仍保留 MuJoCo 自身的控制限制与接触力，交叉目标可能表现为接触、挤压；ON 未完成危险目标是预期干预
- 结果仅针对仿真；尚未验证实体手安全。当前安全模块测试通过也不表示其他项目的全部测试通过
