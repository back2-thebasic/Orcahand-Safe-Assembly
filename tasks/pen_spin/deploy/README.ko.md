# OrcaHand v1 펜 돌리기 배포

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## 0. 터미널 경로 설정

새 터미널을 열 때마다 아래 블록을 한 번 실행합니다. `mkdir -p`는 로그 디렉터리를 만들며 기존 로그는 그대로 둡니다.

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

## 1. ONNX와 시뮬레이션 검증

첫 번째 명령은 ONNX 출력과 내보낼 때 저장한 기준 결과를 비교합니다. `PASS`가 표시되는지 확인합니다. 두 번째 명령은 MuJoCo에서 `clip_002`를 실행합니다. `completed`가 `true`, `pen_lost_at_s`가 `null`인지 확인합니다. 두 명령 모두 실제 로봇 손을 연결하기 전에 실행합니다.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy.checks parity "$PEN_BUNDLE"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand twin --pen-model training
```

## 2. 실물 통신 확인, 관절 캘리브레이션, 중립 자세 확인

OrcaHand에 전원을 공급하고 USB를 연결합니다. 로봇 손에는 아무것도 쥐여 주지 않습니다. 아래 명령은 모터 버스에 연결을 시도하고 결과를 출력한 뒤 연결을 해제합니다. `Connection successful`이 표시되면 다음으로 넘어갑니다.

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

새로 조립한 손의 텐던 장력을 아직 맞추지 않았다면 아래 명령을 실행합니다. 스크립트를 실행하면 모터가 움직입니다. 조립 절차에 따라 장력을 조정한 뒤 `Ctrl+C`로 종료합니다.

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/tension.py" "$ORCA_CFG"
```

관절 캘리브레이션이 필요하면 아래 명령으로 관절의 가동 범위를 측정합니다. 결과는 `config.yaml`과 같은 디렉터리의 `calibration.yaml`에 저장됩니다.

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/calibrate.py" "$ORCA_CFG"
```

캘리브레이션을 마친 뒤 아래 명령을 실행해 손가락과 손목을 중립 자세로 이동합니다. 움직임이 부드러운지, 걸리는 부분이나 계속되는 이상 소음이 있는지 확인합니다.

```bash
"$ORCA_PY" "$ORCA_SCRIPTS/neutral.py" "$ORCA_CFG"
```

## 3. 펜을 쥐지 않은 상태에서 `clip_002` 재생

`--replay`는 시연에 기록된 관절 목표값을 그대로 보냅니다. 손가락의 움직임 방향, 시작 파지 자세, 장착부와의 간격을 확인할 때 사용합니다. `--ease-s 5`는 시작 자세에 도달하는 데 5초를 사용합니다. 이후에는 시연의 원래 속도로 재생합니다. `--wrist-deg`의 기본값은 `0`도입니다. 다른 각도를 쓴다면 재생, 손목 태그 측정, 펜 돌리기에서 같은 값을 사용합니다.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand real --replay \
  --orca-config "$ORCA_CFG" --ease-s 5 \
  --log "$PEN_LOGS/replay_002_$(date +%Y%m%d_%H%M%S).npz"
```

프로그램이 시작되면 모터가 활성화됩니다. 첫 실행에서는 Enter를 한 번 누르고 `frozen`이 표시될 때까지 기다립니다. 시작 자세를 확인한 뒤 `q`를 입력하고 Enter를 눌러 종료합니다.

시작 자세가 맞으면 펜을 쥐지 않은 상태로 다시 실행합니다. `frozen`에서 Enter를 두 번째로 눌러 첫 프레임 자세를 유지하고, 세 번째로 눌러 전체 클립을 재생합니다. 프레임별 로그는 `hold`부터 기록됩니다.

## 4. 카메라 캘리브레이션과 손목 태그 측정

[ChArUco 보드](calibration/charuco-7x5-35mm.png)와 [손목용 AprilTag 36h11 ID 0](calibration/apriltag-36h11-id0-50.4mm.png)을 100% 배율로 인쇄합니다. ChArUco 인쇄물을 평평하고 단단한 판에 붙이고, 한 칸의 변이 약 35 mm, 칸 안의 마커가 약 26 mm인지 확인합니다. 실제 추적에 사용할 카메라와 해상도로 캘리브레이션합니다.

```bash
"$ORCA_PY" "$PEN_TASK/perception/tools/calibrate_camera.py" \
  --source "$PEN_CAMERA"
```

카메라는 고정하고 보드를 움직입니다. 화면 중앙에서 기울기를 바꿔 5장, 네 모서리에서 각각 2장, 거리를 바꿔 2장을 찍으면 됩니다. 보드가 멈추고 미리보기에 `12 corners` 이상이 표시될 때 스페이스바를 누릅니다. `views kept`가 1만큼 증가하는지 확인합니다. 15장 이상을 모은 뒤 Enter를 누르면 [camera.yaml](../perception/config/camera.yaml)에 저장됩니다.

손바닥이 위를 향할 때 카메라가 볼 수 있는 위치에 손목 태그를 단단히 고정합니다. 태그 주변의 흰 여백은 남겨 둡니다. 태그를 xArm 끝단 장착부에 붙이면 측정할 때와 펜을 돌릴 때 OrcaHand 손목 각도를 같게 유지합니다.

```bash
nano "$PEN_TASK/perception/config/wrist_tag.yaml"
```

`tag_id`는 `0`으로 둡니다. 검은 테두리 바깥쪽 변의 실측 길이를 `size_m`에 입력합니다. 학습 모델의 `right_palm_link`를 기준으로 태그 중심의 세 축 오프셋을 측정해 `pos_in_palm`에 입력합니다. 길이 단위는 모두 미터입니다.

조립 CAD 또는 실측으로 태그의 장착 방향을 구해 `quat_in_palm`에 `[w, x, y, z]` 순서로 입력합니다. 인쇄된 태그에서 오른쪽은 `+X`, 위쪽은 `+Y`, 종이 표면 바깥쪽은 `+Z`입니다. 치수와 방향을 확인한 뒤 `measured: true`로 설정합니다. `Ctrl+O`와 Enter로 저장하고 `Ctrl+X`로 종료합니다.

## 5. 펜 추적 별도 검증

[펜 라벨의 부착 면과 방향](../hardware/tags/texture_faces.csv)에 맞춰 펜 양 끝에 라벨을 붙입니다. 첫 번째 터미널에서 observer를 실행합니다. observer는 카메라를 열고 손바닥 좌표계로 표현한 펜 자세를 로컬 포트 `5555`로 전송합니다.

```bash
"$ORCA_PY" "$PEN_TASK/perception/observer.py" --source "$PEN_CAMERA" --preview
```

`palm frame FIXED`가 표시되면 직접 펜을 들고 시작 파지 자세와 회전 범위를 따라 움직입니다. 미리보기의 윤곽선이 펜을 따라가는지, 앞뒤가 갑자기 뒤바뀌는지, 잠시 가린 뒤 추적이 다시 되는지 확인합니다. 카메라나 손바닥을 움직였다면 다시 고정하고 미리보기 창에서 `w`를 눌러 손목 태그를 다시 샘플링합니다.

두 번째 터미널에서 0단계를 실행한 뒤 아래 명령으로 자세를 읽습니다. OrcaHand를 고정한 채 손바닥 좌표계의 방향을 아는 한 축을 따라 펜을 약 20 mm 옮깁니다. 해당 좌표의 변화량과 부호를 확인합니다. 실측한 위치를 최소 한 곳 사용해 원점 오프셋도 확인합니다. `A_top` 방향은 펜의 뒤쪽에서 앞쪽으로 향하는 단위 벡터입니다.

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
                print("observer 메시지가 수신되지 않음")
                continue
            pen = msg.get("pen")
            if not msg["palm_fixed"]:
                print("손목 태그 샘플링 중")
            elif pen is None:
                print("현재 프레임에 펜 자세 없음")
            else:
                pose = pen["pose"]
                center = [round(v * 1000, 1) for v in pose[:3]]
                axis = quat_apply(pose[3:], [1, 0, 0]).round(2).tolist()
                print(f"펜 중심 (mm) {center}  A_top 방향 {axis}")
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass
PY
```

`Ctrl+C`를 눌러 수신 프로그램을 종료합니다. 첫 번째 터미널의 observer는 계속 실행해도 됩니다. 수신 프로그램은 초당 약 네 프레임을 표시하므로 짧은 추적 끊김은 출력 사이에 지나갈 수 있습니다.

## 6. 펜 자세 갱신이 끊겼을 때 종료되는지 검증

실물의 카메라 모드에서는 마지막 유효한 펜 자세를 기본적으로 최대 150 ms 동안 사용합니다. 유효한 자세가 없거나 이미 유효 시간이 지났다면 `frozen`에서 Enter를 눌러도 `not starting`이 표시되고 `frozen` 상태가 유지됩니다. `hold` 중이거나 펜을 돌리는 중에 유효 시간이 지나면 `stopped: pen pose timeout`이 표시되며 새 관절 목표값 전송을 멈추고 이번 실행을 종료합니다. 잠깐 추적이 끊겨도 제한 시간까지는 마지막 자세를 사용할 수 있습니다. observer가 계속 메시지를 보내더라도 메시지에 펜 자세가 없으면 마지막 유효 자세 이후 경과 시간은 계속 늘어납니다.

실제 펜을 쥐여 주기 전에 아래 검사를 컴퓨터에서 실행합니다. 모의 카메라와 모의 로봇 손으로 오래된 자세가 시작을 차단하는지, 짧게 추적이 끊겨도 복구되는지, `hold`와 회전 중에 제한 시간을 넘기면 종료되는지 확인합니다. `PASS`가 표시되면 모의 검사를 통과한 것입니다.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy.checks pen-timeout "$PEN_BUNDLE"
```

7단계 명령에는 이 검사가 자동으로 적용됩니다. 초기값은 150 ms이며 5단계에서 확인한 카메라 동작을 기준으로 `--pen-timeout-ms`를 조정할 수 있습니다. 자동 종료 시 `SUMMARY`와 `--log`로 저장한 파일에 `abort_reason`이 기록됩니다. 종료할 때 모터 토크 해제를 시도하므로 펜이 떨어질 수 있습니다.

## 7. 첫 프레임에서 펜을 쥔 뒤 전체 회전 시도

시간 초과 검사를 통과한 뒤 첫 번째 터미널의 observer는 계속 실행해 둡니다. 두 번째 터미널에서 아래 명령을 실행합니다. 실제 OrcaHand는 카메라가 측정한 펜 자세를 사용해 배포 번들의 `policy.onnx`를 실행합니다. 3단계와 4단계에서 0도가 아닌 손목 각도를 사용했다면 `--wrist-deg`에 같은 값을 지정합니다.

```bash
cd "$PEN_TASK/deploy"
"$ORCA_PY" -m penspin_deploy "$PEN_BUNDLE" \
  --clip clip_002 --hand real --pen camera --observer localhost:5555 \
  --orca-config "$ORCA_CFG" --ease-s 5 \
  --log "$PEN_LOGS/hold_002_$(date +%Y%m%d_%H%M%S).npz"
```

1. Enter를 한 번 눌러 펜을 쥐지 않은 손을 시작 파지 자세로 이동합니다. `frozen`이 표시될 때까지 기다립니다.
2. 펜을 시작 파지 자세에 놓고 Enter를 두 번째로 누릅니다. `hold` 상태에서 ONNX 정책이 관절을 조정해 첫 프레임 자세를 유지합니다. 첫 실행에서는 몇 초 동안 관찰한 뒤 `q`를 입력하고 Enter를 눌러 종료합니다. 종료 시 모터 토크 해제를 시도하므로 손 아래에 부드러운 패드를 놓아 펜을 받을 수 있습니다.
3. 파지가 안정적이면 로그 파일의 접두사를 `spin_002_`로 바꾸고 다시 실행합니다. `hold`에 들어가 펜을 안정적으로 쥔 것을 확인한 뒤 Enter를 세 번째로 눌러 기준 궤적을 진행합니다. `clip_002`가 반복해서 성공하면 `--clip clip_004`를 시도합니다.

로그는 `tasks/pen_spin/deploy/logs/`에 저장됩니다. 파일명에 찍힌 시간으로 각 실행을 구분합니다. 펜이 떨어지면 녹화 영상과 함께 20 ms 제어 주기 단위의 `pen_age`, 실측 관절 각도 `qpos`, 명령한 목표값 `target`을 비교합니다. `completed: true`는 기준 궤적이 끝까지 진행됐다는 뜻입니다. 펜을 끝까지 쥐고 있었는지는 현장이나 영상에서 확인합니다.
