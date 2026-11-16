# OrcaHand v1 转笔部署

[English](README.en.md) | [中文](README.md) | [한국어](README.ko.md)

## 0. 设置终端路径

每个新终端执行一次

`mkdir -p` 创建日志目录，已经存在的日志会保留

```bash
ORCA_ROOT=/Users/su/Code/Orcahand
ORCA_PY="$ORCA_ROOT/.scratch/deploy-venv/bin/python"
ORCA_SCRIPTS="$ORCA_ROOT/orca_core/scripts"
ORCA_CFG="$ORCA_ROOT/orca_core/orca_core/models/v1/orcahand-right/config.yaml"
PEN_TASK="$ORCA_ROOT/tasks/pen_spin"
PEN_BUNDLE="$PEN_TASK/policies/clip002_004_dr_scratch/deploy"
PEN_LOGS="$PEN_TASK/deploy/logs"
PEN_CAMERA=device:0
mkdir -p "$PEN_LOGS"
```

## 1. 验证 ONNX 和仿真

第一条命令对比 ONNX 和导出时的参考结果，预期看到 `PASS`

第二条命令在 MuJoCo 里运行 `clip_002`，看 `completed` 是否为 `true`、`pen_lost_at_s` 是否为 `null`

这两条在接真机前跑

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy.checks parity "$PEN_BUNDLE"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand twin --pen-model training
```

## 2. 检查真手通信、校准与中立姿势

给 OrcaHand 供电、接 USB，让手里空着

下面的命令尝试连接电机总线，打印结果并断开

看到 `Connection successful` 再继续

```bash
"$ORCA_PY" - "$ORCA_CFG" <<'PY'
import sys
from orca_core import OrcaHand

hand = OrcaHand(config_path=sys.argv[1])
ok, message = hand.connect(interactive=False)
print(message)
if ok:
    hand.disconnect()
raise SystemExit(0 if ok else 1)
PY
```

新装手如果还没张紧，运行下面的命令进入张紧模式

脚本会动电机，按装配流程张紧后按 `Ctrl+C` 退出。

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/tension.py" "$ORCA_CFG"
```

这只手如果还没校准，运行下面的命令测量关节行程

结果保存在 `config.yaml` 旁的 `calibration.yaml`

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/calibrate.py" "$ORCA_CFG"
```

校准后，运行下面的命令让手指和腕关节回到中立姿势

观察手指是否平顺，有没有卡住或持续异响

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/neutral.py" "$ORCA_CFG"
```

## 3. 空手回放 `clip_002`

`--replay` 直接发送示范关节目标，用来检查手指方向、起始抓姿和夹具间隙

`--ease-s 5` 让手用五秒到达起始姿势，后续按示范原速回放

`--wrist-deg` 默认是 `0` 度，如果使用其他腕角，回放、腕标测量和后续转笔都用同一个值。

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand real --replay \
  --orca-config "$ORCA_CFG" --ease-s 5 \
  --log "$PEN_LOGS/replay_002_$(date +%Y%m%d_%H%M%S).npz"
```

程序启动时就会使能电机。首次按一次 Enter，等到 `frozen`，看姿势是否正确，然后输入 `q` 并按 Enter 退出

姿势正确后重跑，空手到 `frozen` 后按第二次 Enter 保持首帧，再按第三次 Enter 播放整段

日志在 `hold` 开始后才有逐帧记录。

## 4. 标定相机和腕标

按 100% 打印 [ChArUco 板](calibration/charuco-7x5-35mm.png) 和 [腕标 AprilTag 36h11 ID 0](calibration/apriltag-36h11-id0-50.4mm.png)。把 ChArUco 板贴在平整硬板上，确认格子约 35 mm、格内标记约 26 mm。用正式跟踪时的相机和分辨率执行标定。

```bash
"$ORCA_PY" "$PEN_TASK/perception/tools/calibrate_camera.py" \
  --source "$PEN_CAMERA"
```

固定相机，移动标定板

可以在中央拍 5 张不同倾角、画面四角各 2 张，再拍 2 张不同距离

板子停稳、窗口显示至少 `12 corners` 时按空格，确认 `views kept` 加一

收够至少 15 张后按 Enter，脚本会写入 [camera.yaml](../perception/config/camera.yaml)

把腕标固定在掌心朝上时相机能看见的位置，保留白边。如果装在 xArm 末端夹具上，测量和转笔使用同一个 OrcaHand 腕角

```bash
nano "$PEN_TASK/perception/config/wrist_tag.yaml"
```

`tag_id` 保持 `0`

实测黑框边长填 `size_m`，以训练模型 `right_palm_link` 为基准测量标签中心的三轴偏移，填 `pos_in_palm`，长度都用米

用装配 CAD 或实测确定标签的安装朝向，填 `quat_in_palm`，顺序是 `[w, x, y, z]`。标签印刷面上的右、上、朝外分别是标签的 `+X`、`+Y`、`+Z`

确认尺寸和朝向后设 `measured: true`，用 `Ctrl+O`、Enter 保存，再用 `Ctrl+X` 退出

## 5. 单独验证笔跟踪

按 [笔贴纸的面和方向](../hardware/tags/texture_faces.csv) 贴好笔两端

在第一个终端运行 observer，observer 会打开相机并把笔在手掌坐标系下的位姿发布到本机端口 `5555`

```bash
"$ORCA_PY" "$PEN_TASK/perception/observer.py" --source "$PEN_CAMERA" --preview
```

等到 `palm frame FIXED`，用自己的手拿着笔走过起始抓姿和旋转范围
，看轮廓是否贴着笔、两端是否突然对调、遮挡后能否恢复。如果相机或手掌移动了，停稳后在预览窗口按 `w` 重新采样


在第二个终端先执行第 0 步，再运行下面的命令读位姿。保持 OrcaHand 不动，把笔沿一根已知的手掌轴平移约 20 mm，确认对应坐标变化的大小和正负方向。再用至少一个实测位置核对原点偏移。`A_top` 方向是从笔尾指向笔头的单位向量。

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" - <<'PY'
import time
import zmq
from penspin_deploy.geometry import quat_apply

with zmq.Context() as ctx, ctx.socket(zmq.SUB) as sub:
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVTIMEO, 2000)
    sub.connect("tcp://127.0.0.1:5555")
    try:
        while True:
            try:
                msg = sub.recv_json()
            except zmq.Again:
                print("没收到 observer 消息")
                continue
            pen = msg.get("pen")
            if not msg["palm_fixed"]:
                print("腕标还在采样")
            elif pen is None:
                print("当前帧没有笔位姿")
            else:
                pose = pen["pose"]
                center = [round(v * 1000, 1) for v in pose[:3]]
                axis = quat_apply(pose[3:], [1, 0, 0]).round(2).tolist()
                print(f"笔中心(mm) {center}  A_top 方向 {axis}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
PY
```

按 `Ctrl+C` 退出读取程序。第一个终端的 observer 可以继续运行。读取程序约每秒显示四帧，短暂掉帧可能出现在两次打印之间

## 6. 验证位姿超时退出

真机相机模式默认允许最后一帧有效笔位姿最多使用 150 ms。等到 `frozen` 后，如果没有有效位姿或位姿已过期，按 Enter 会提示 `not starting` 并留在 `frozen`。进入 `hold` 或开始转笔后，如果位姿过期，程序会打印 `stopped: pen pose timeout`，停止发送新关节目标并退出本轮。短暂掉帧时，最后一帧仍可用到门限。observer 仍在发消息但看不到笔时，位姿年龄也会继续增长

带笔前，在电脑上运行下面的检查。它用模拟相机和模拟手验证启动拦截、短暂掉帧恢复以及 `hold` 和转笔阶段的超时退出。看到 `PASS` 表示模拟检查通过

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy.checks pen-timeout "$PEN_BUNDLE"
```

第 7 步的命令会自动启用检查，无需改 YAML。150 ms 是初始值，可以根据第 5 步的实际相机表现用 `--pen-timeout-ms` 调整。如果自动退出，`SUMMARY` 和通过 `--log` 保存的日志会带有 `abort_reason`。退出会尝试解除力矩，笔可能会落下

## 7. 带笔握住首帧，再尝试完整转笔

超时保护验证通过后，保持第一个终端的 observer 运行。在第二个终端执行下面的命令。真实 OrcaHand 会根据相机测得的笔位姿运行部署包里的 `policy.onnx`

如果第 3、4 步用了非零腕角，在这条命令里也加上对应的 `--wrist-deg`

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand real --pen camera --observer localhost:5555 \
  --orca-config "$ORCA_CFG" --ease-s 5 \
  --log "$PEN_LOGS/hold_002_$(date +%Y%m%d_%H%M%S).npz"
```

1. 第一次 Enter，空手移到起始抓姿，等终端显示 `frozen`
2. 把笔放到起始抓姿后按第二次 Enter。看到 `hold` 后，ONNX 会调整关节来保持首帧。第一轮观察几秒，输入 `q` 并按 Enter 退出。退出时电机会尝试解除力矩，可以在下面放软垫接笔
3. 握笔通过后重跑，把日志前缀改成 `spin_002_`。到 `hold` 且握稳后按第三次 Enter，参考轨迹开始推进。`clip_002` 重复成功后，再试 `--clip clip_004`

日志写到 `tasks/pen_spin/deploy/logs/`，时间戳用来区分各次测试。掉笔时，对照录像、`pen_age`（单位为 20 ms 控制周期）、实测关节角 `qpos` 和目标 `target`。终端的 `completed: true` 表示参考轨迹跑到末尾，是否全程握住笔看现场或录像
