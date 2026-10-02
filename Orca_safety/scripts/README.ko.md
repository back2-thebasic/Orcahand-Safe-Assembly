# 실험 및 재생 스크립트

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

[메인 README](../README.ko.md)에 따라 의존성을 설치하세요. 웹캠 녹화 예시를 제외한 모든 명령은 `Orca_safety`와 `orca_teleop`이 있는 작업 공간의 최상위 디렉터리에서 실행합니다.

## 스크립트 용도

- `verify_geometry.py`: 정적 자세, 충돌 거리 및 기울기 검증.
- `compare_simulation.py`: 동일한 목표 궤적으로 OFF/ON 데이터 실험.
- `generate_retarget_replay.py`: 합성 특징점을 기존 retargeter에 입력하여 NPZ 재생 입력 생성.
- `generate_crossing_motion.py`: 검지와 중지 교차 목표의 JSONL 입력 생성.
- `replay_video.py`: 실제 기록 또는 합성 JSONL을 읽어 OFF, ON 및 나란히 배치된 MP4 출력.
- `replay_viewer.py`: MuJoCo에서 OFF 또는 ON을 대화형으로 관찰.

## MuJoCo 대화형 재생

macOS에서는 `mjpython`으로 창을 실행합니다.

```bash
orca_teleop/.venv/bin/mjpython Orca_safety/scripts/replay_viewer.py \
  Orca_safety/output/teleop/retarget-session-02.jsonl \
  --mode OFF --view front --speed 0.5 --paused
```

OFF를 ON으로 바꾸어 비교합니다. 스페이스는 재생/일시 정지, N/B는 한 단계 앞으로/뒤로 이동, R은 처음으로 돌아갑니다. 마우스 왼쪽 드래그로 회전, 오른쪽 드래그로 이동, 휠로 확대/축소합니다. 창을 열기 전에 궤적을 계산하므로 ON 준비에 시간이 걸릴 수 있습니다.

[합성 교차 동작](../output/synthetic/index-middle-crossing/README.ko.md)을 재생하려면 입력을 `Orca_safety/output/synthetic/index-middle-crossing/motion.jsonl`로 바꾸세요. 웹캠 시간 파일은 필요하지 않습니다.

## 영상 출력

```bash
orca_teleop/.venv/bin/python Orca_safety/scripts/replay_video.py \
  Orca_safety/output/teleop/retarget-session-02.jsonl \
  --video-timing Orca_safety/output/video/camera/session-02.timing.json \
  --clip 7:14 --view front --speed 0.5 \
  --output Orca_safety/output/video/my-key-actions
```

`OFF.mp4`, `ON.mp4`, `OFF-ON.mp4`가 생성됩니다. 예시 구간은 실제 원본 영상을 보고 조정하세요. 전체 기록을 출력하려면 `--clip`을 제거합니다.

대화형 재생과 영상 재생의 공통 매개변수:

- `--clip START:END`: 여러 번 지정할 수 있습니다. `--video-timing`과 함께 쓰면 웹캠 영상의 초 단위 시간을 사용합니다. 없으면 첫 로그 프레임부터 경과한 현실 시간을 사용하며, 합성 입력은 궤적 시간을 사용합니다. 선택 구간 이전의 시뮬레이션 이력도 계산합니다.
- `--speed 0.5`: 시뮬레이션 시간 기준으로 절반 속도로 재생하며 제어 주기는 변경하지 않습니다. ON은 로그의 `q_safe`를 재생하는 대신 안전 목표를 다시 계산합니다.
- `--view front`: 손바닥 정면 시점입니다. 다른 방향은 `--azimuth`, `--elevation`으로 설정합니다. `--camera-distance`가 클수록 멀어지며, 현재 v1 기본값은 약 0.45 m입니다.

## 선택적 웹캠 원본 영상 녹화

`orca_teleop` 디렉터리에서 원격 조작을 실행할 때 기존 `teleop_sim.py --local` 명령에 다음 인수를 추가합니다.

```bash
--collision-safety-log ../Orca_safety/output/teleop/retarget-session-04.jsonl \
--record-video ../Orca_safety/output/video/camera/session-04.mp4
```

`--record-video`를 지정하지 않으면 녹화하지 않으며, `--show-video`와 독립적입니다. Ctrl+C로 정상 종료하세요. 구간 시간을 맞추기 위한 `.timing.json`도 생성되며 `--video-timing`으로 사용합니다. 영상과 실행 후 로그는 현실 시간으로 근사 정렬되므로 인식 및 큐 지연이 포함됩니다.

## 데이터 실험 및 입력 생성

```bash
orca_teleop/.venv/bin/python Orca_safety/scripts/verify_geometry.py \
  --output Orca_safety/output/my-passive --render
orca_teleop/.venv/bin/python Orca_safety/scripts/compare_simulation.py \
  --steps 120 --poses Orca_safety/output/my-passive/poses.json \
  --output Orca_safety/output/my-comparison
orca_teleop/.venv/bin/python Orca_safety/scripts/generate_retarget_replay.py \
  --steps 120 --output Orca_safety/output/replay/my-replay.npz
orca_teleop/.venv/bin/python Orca_safety/scripts/compare_simulation.py \
  --replay Orca_safety/output/replay/my-replay.npz \
  --output Orca_safety/output/my-retarget-comparison
orca_teleop/.venv/bin/python Orca_safety/scripts/generate_crossing_motion.py \
  --output Orca_safety/output/synthetic/my-crossing
```

120개 제어 단계는 시뮬레이션 시간 1.2초에 해당합니다. 비교, 영상 출력 및 교차 동작 생성에는 새 출력 디렉터리를 사용하세요. 정적 검증은 같은 이름의 파일을 덮어씁니다. macOS 렌더링에는 그래픽 접근 권한이 필요하며, 오프라인 실험에는 웹캠이나 실제 로봇이 필요하지 않습니다.

출력 파일은 [output/README.md](../output/README.ko.md), 코드 테스트는 [tests/README.md](../tests/README.ko.md), 검증 범위는 [implementation.md](../docs/implementation.md) (중국어)를 참고하세요.
