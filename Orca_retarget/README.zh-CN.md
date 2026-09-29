# ORCA Hand Retarget 测试对比

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

ORCA v1 右手 · MediaPipe · Adaptive Analytical · MuJoCo 仿真

本仓库用于分享 YAML 参数与测试视频。包括配置改动，和3个场景比较视频。README 中使用兼容性更好的 H.264 预览，原始 MOV 可通过每段视频下方的链接下载

## 配置与改动说明

### A：原始官方配置

**文件：[baseline.yaml](configs/baseline.yaml)**

- 用途：作为baseline，保留额外指尖偏移
- `w_pos: 1.0`，`w_dir: 10.0`，`w_full_hand: 1.0`
- `norm_delta: 0.04`，`lp_alpha: 1.0`（无输出低通平滑）

### B：零偏移配置

**文件：[baseline_zero_offsets.yaml](configs/baseline_zero_offsets.yaml)**

- 相对 A：五指 `fingertip_offsets_m` 均为 `[0, 0, 0]`，直接使用 URDF 指尖 frame 原点，消除指尖偏移
- 其他 YAML 参数与 A 一致
- 修改目的：避免在已定义的指尖位置上额外叠加指尖长度
- 注意：该修改也影响自动尺度标定，不是固定尺度下的纯几何对照

### C：调整参数后配置

**文件：[last.yaml](configs/last.yaml)**

- 当前相对 B：`w_pos: 1.0 → 2.0`，`w_full_hand: 1.0 → 0.5`
- 修改目的：提高位置约束并降低全手形状约束的权重，观察是否改善捏合，同时检查姿态是否退化

> B/C 需要 retarget 代码支持 `fingertip_offsets_m`；未修改的上游代码不一定读取此字段

## 指尖偏移：修改原因与影响

### 为什么默认配置有偏移量？

Orca 的 retarget 实现使用经验偏移，补偿 URDF 坐标系原点与实际指腹位置之间的差异。它是几何位置补偿，不是电机零位或关节角度偏置。

上游代码将这些数值标注为手工调整的经验值，并计划使用碰撞几何定义替代，因此它们不一定适合所有模型版本。默认拇指、食指、中指、无名指、小指分别沿局部 z 轴增加 **30.5、43.3、45.3、45.3、38.3 mm**。

来源：[Orca 上游偏移定义](https://github.com/orcahand/orca_teleop/blob/main/src/orca_teleop/retargeting/constants.py)、[相关几何工具说明](https://github.com/orcahand/orca_teleop/blob/main/src/orca_teleop/retargeting/utils.py)。

### 为什么设置零偏移后可以捏合？

优化器使用的指尖位置为：

- **默认偏移：** URDF 指尖 frame 的位置，加上经该 frame 旋转后的局部偏移。
- **零偏移：** 直接使用 URDF 指尖 frame 的位置。

当前 v1 URDF 已通过固定关节定义指尖位置。例如，食指指尖 frame 相对于末节 link 的位置为 `[-9, 0, 35] mm`；该固定关节无旋转，叠加默认偏移后，计算点变为 `[-9, 0, 78.3] mm`。这些是相对于末节 link 的坐标，并非整根手指长度。

如果额外偏移使计算点偏离实际接触位置，优化器可能让这些“虚拟指尖”接近目标，但仿真中的真实指尖仍存在间隙。设置零偏移后，计算点回到 URDF 指尖 frame，因此可能改善实际指尖的对齐与捏合。

**本次观察：** 原始配置下，拇指与食指无法完成捏合；设置零偏移后能够完成。

这一结果支持“经验偏移与当前模型可能不匹配”的解释，但不能证明默认配置在所有模型上都错误，也不能证明偏移是唯一原因。URDF 指尖 frame 不一定恰好位于实际接触表面，后续仍可能需要更小、经过几何验证的补偿。


**零偏移只改变 retarget 使用的计算点，不改变模型外形、碰撞体或关节限位，也不直接提供平滑或夹持力控制。** 视觉上闭合不等于已经验证物理接触或稳定抓持。


## 如何下载和运行 YAML

### 1. YAML 放在哪里？

在本仓库点击 **Code → Download ZIP**，解压后将 `configs/` 放到你本机 Orca 项目的 `orca_adaptive_test/` 中，保持如下结构。也可以在单个 YAML 页面点击 Raw 下载对应文件。

```text
Orcahand/
├── orca_teleop/
│   ├── .venv/
│   └── scripts/teleop_sim.py
├── orca_sim/
├── orcahand_description/
│   └── v1/models/urdf/orcahand_right.urdf
└── orca_adaptive_test/
    └── configs/
        ├── baseline.yaml
        ├── baseline_zero_offsets.yaml
        └── last.yaml
```

不要放进 `.venv` 或 Python 的 site-packages。已有同名实验文件时先保留旧版本，避免旧视频失去对应参数。

### 2. 运行前提

B/C 的零偏移字段需要代码支持。打开 `orca_teleop/src/orca_teleop/retargeting/adaptive_analytical.py`，在 `_build_frame_indices()` 中检查是否已经读取 `fingertip_offsets_m`。本次测试使用的代码已有此支持。

如果没有，将函数末尾设置 `_frame_offsets` 的那一小段替换为以下代码（放在函数内部，保持缩进），其他代码不动：

```python
        self._frame_offsets = [np.zeros(3, dtype=np.float64) for _ in self._computed_frame_names]
        tip_offsets = self._retarget_cfg.get("fingertip_offsets_m", {})
        for finger, frame_name in zip(FINGERS, self._frame_map.tip, strict=True):
            frame_idx = self._computed_frame_names.index(frame_name)
            offset = np.asarray(
                tip_offsets.get(finger, FINGERTIP_OFFSETS_M[finger]), dtype=np.float64
            )
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError(f"fingertip_offsets_m.{finger} must contain three finite values")
            self._frame_offsets[frame_idx] = offset
```

没有此支持时，单纯下载 B/C YAML 不能保证零偏移生效。A 未设置该字段，仍使用旧偏移。确保运行环境导入的是你修改的源码。

### 3. 选择 A、B 或 C，然后启动

先进入 `orca_teleop`，替换为自己的实际路径：

```bash
cd /xxx/Orcahand/orca_teleop
```

三选一，在同一个终端设置本次配置：

```bash
# A：原始配置
RETARGET_CONFIG=../orca_adaptive_test/configs/baseline.yaml
```

```bash
# B：五指零偏移
RETARGET_CONFIG=../orca_adaptive_test/configs/baseline_zero_offsets.yaml
```

```bash
# C：当前最新配置
RETARGET_CONFIG=../orca_adaptive_test/configs/last.yaml
```

然后执行统一启动命令。直接使用虚拟环境中的 mjpython，无需先 activate：

```bash
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right \
  --version v1 \
  --hand right \
  --local \
  --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config "$RETARGET_CONFIG"
```

每行末尾的反斜杠后不要加空格。v2 YAML 不适用于这条 v1 命令

踩的一个坑：不要同时开启两个使用同一端口的仿真。若绿色骨架正常但仿真不动，检查 `Publisher connected` 日志，并用 `lsof -nP -iTCP:50051 -sTCP:LISTEN` 查看是否有旧实例残留

## S01：张手、半握、握拳

动作：张手 → 半握 → 握拳 → 张开，重复 3 次

### A：原始配置

https://github.com/user-attachments/assets/c61373b0-8059-412d-8560-4a3ba6c9c63a

[查看或下载原始 MOV](videos/S01/S01-baseline.mov)

### B：零偏移

https://github.com/user-attachments/assets/29e975da-077b-48ba-9e53-38181359c819

[查看或下载原始 MOV](videos/S01/S01-zero%20offsets.mov)

### C：当前最新配置

https://github.com/user-attachments/assets/25fd48fd-ee29-44b5-9cb1-a7689365d1fd

[查看或下载原始 MOV](videos/S01/S01-last.mov)

**本场景结论：** 所有yaml完成较好。B中对大拇指的reatrgert并不是太好

## S02：拇指—食指慢速捏合与释放

动作：慢速靠近 → 捏合 → 释放，重复 3 次

### A：原始配置

https://github.com/user-attachments/assets/4c55b450-9961-4b50-bdd5-de74a9e2b264

[查看或下载原始 MOV](videos/S02/S02-baseline.mov)

### B：零偏移

https://github.com/user-attachments/assets/8d29cfac-a2db-4344-89fa-7863eeb2a6a5

[查看或下载原始 MOV](videos/S02/S02-zero%20offsets.mov)

### C：当前最新配置

https://github.com/user-attachments/assets/c37a71f9-74a2-4a70-8b3a-697967042f29

[查看或下载原始 MOV](videos/S02/S02-last.mov)

**本场景结论：** 官方默认配置无法完成捏合动作，拇指和食指始终有间隙。B和C对该任务完成情况较好，但是B中拇指捏合时方向不太自然

## S03：三指抓持与释放

动作：拇指—食指-中指捏合 → 释放，重复 3 次。

### A：原始配置

https://github.com/user-attachments/assets/52aaa9e6-61f5-478b-94e8-735014e2cfac

[查看或下载原始 MOV](videos/S03/S03-baseline.mov)

### B：零偏移

https://github.com/user-attachments/assets/57ebade0-5c62-416b-878a-cc3ec69e8192

[查看或下载原始 MOV](videos/S03/S03-zero%20offsets.mov)

### C：当前最新配置

https://github.com/user-attachments/assets/505e5a70-ac18-4eff-b557-5c8502d0ffa5

[查看或下载原始 MOV](videos/S03/S03-last.mov)

**本场景结论：** 空手测试仅验证抓持姿态，不代表能抓住物体。官方配置依然无法捏合。B和C完成的较好，但B在松开时，对拇指的retarget不够好，拇指的IP Joint（最靠近指尖的一个关节）的retarget不太自然

## 如何添加视频

1. 原始视频保存在对应的 `videos/S01/`、`S02/`、`S03/` 目录中
2. 普通仓库视频链接不会可靠地生成 README 内嵌播放器。README 中使用上传到 GitHub Markdown 编辑区后生成的 `user-attachments` 地址
3. 预览视频使用 H.264 MP4，以获得更好的浏览器兼容性
4. 每次录像同时显示人手与仿真，固定标定手势。更新 YAML 时也更新本页说明；已有视频仍对应旧参数时，新增配置编号，不覆盖旧参数

GitHub 视频附件的操作见[官方说明](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)

## 文件结构

```text
configs/                 所有分享用 YAML
videos/S01/              S01-baseline.mov、S01-zero offsets.mov、S01-last.mov
videos/S02/              S02-baseline.mov、S02-zero offsets.mov、S02-last.mov
videos/S03/              S03-baseline.mov、S03-zero offsets.mov、S03-last.mov
README.md                英文版（仓库默认）
README.zh-CN.md          中文版
README.ko.md             韩文版
```

