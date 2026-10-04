# OrcaHand - Safety Layer

[English](README.md) | [中文](README.zh.md) | [한국어](README.ko.md)

Safety Layer는 OrcaHand 원격 조작과 제어 중 손가락 자체 충돌을 줄이기 위한 모듈입니다. 기존 retargeter와 MuJoCo 사이에 배치되어 충돌 제약과 관절 제한을 만족하면서 원래 명령을 최대한 유지합니다. **v2는 안전 소프트웨어의 버전이며, 로봇 모델은 v1 오른손입니다.**

## Safety Layer 효과 검증

- [네 가지 동작의 OFF/ON 영상 비교](output/video/demo_video/README.ko.md): retargeting 기록과 합성 동작을 MuJoCo에서 재생하여 안전층 적용 전후를 관찰합니다.
- [일곱 가지 합성 동작의 OFF/ON 데이터 비교](output/comparison/README.ko.md): 동일한 목표 궤적의 충돌, 안전 간격 위반, 동작 개입 및 실행 시간을 비교합니다.
- [기존 동작 데이터 기반 Safety Layer OFF/ON 비교 실험](results/safety_benchmark/benchmark_report.ko.md): 12개 합성 및 실제 사람의 기록 시나리오에 동일한 관절 목표를 적용하여 안전 효과, 동작 수정량, fallback 비율 및 계산 시간을 비교합니다.

## 기존 Pipeline

Adaptive Analytical Retargeter는 손의 특징점으로 관절 목표를 생성하고 시뮬레이션에 직접 전달합니다.

```text
Human Hand
    ↓
Hand Tracking
    ↓
Adaptive Analytical Retargeting
    ↓
q_nominal
    ↓
OrcaHand / MuJoCo
```

## Safety Layer를 추가한 Pipeline

Retargeter는 변경하지 않습니다. 안전층은 원래 목표 `q_nominal`과 시뮬레이션에서 측정한 현재 상태 `q_current`를 이용해 최종 목표 `q_safe`를 생성합니다.

```text
Human Hand
    ↓
Hand Tracking
    ↓
Adaptive Analytical Retargeting
    ↓
q_nominal
    ↓
┌─────────────────────────┐
│ Collision Safety Layer  │
│-------------------------│
│ Distance Checking       │
│      ↓                  │
│ CBF Safety Constraint   │
│      ↓                  │
│ QP Safety Filter        │
└─────────────────────────┘
    ↓
q_safe
    ↓
MuJoCo
    ↓
Real OrcaHand
```

그림의 실제 로봇 경로는 향후 확장 대상이며, 현재 안전층은 MuJoCo 시뮬레이션에만 연결되어 있습니다.

## 핵심 구성 요소

- **Distance Checking**: Pinocchio/FCL로 설정된 충돌 링크 쌍의 거리를 조회합니다. 가까운 쌍에 대해서만 거리 기울기를 계산하여 관절 움직임이 두 링크를 가까워지게 하는지 판단합니다.
- **CBF Safety Constraint**: 안전 여유를 $h(q)=d(q)-d_{safe}$로 정의하고 가까운 링크 쌍에 다음 제약을 적용합니다.

$$
 \nabla d(q)^T\Delta q \geq -\eta h(q)
$$

  각 제어 단계에서 거리가 감소하는 양을 제한합니다. 안전 경계에서는 국소 선형 모델상 간격을 더 줄이는 움직임을 허용하지 않습니다. 기본 안전 거리는 5 mm이며, 15 mm 미만에서 제약이 활성화됩니다. 두 값 모두 설정에서 변경할 수 있습니다.

- **QP Safety Filter**: OSQP가 $\Delta q_{nom}=q_{nominal}-q_{current}$에 가까운 관절 증분을 구하면서 CBF, 관절 각도 및 단계별 움직임 제한을 만족시킵니다. 출력은 $q_{safe}=q_{current}+\Delta q$이며, 제약이 활성화되지 않으면 원래 목표에 가깝습니다.
- **Nonlinear Command Validation and Fallback**: 후보 목표의 충돌 거리를 다시 계산합니다. 안전 간격이 부족하면 움직임을 최대 8회 절반으로 줄이며, 매번 CBF, 관절/단계 제한과 모든 설정된 충돌 쌍을 검사합니다. 유효한 작은 움직임이 있으면 실행하고, 최적화 실패 또는 유효한 후보가 없으면 마지막으로 검증된 목표를 유지합니다.

## 문서 안내

- [output/README.md](output/README.ko.md): 출력 디렉터리, 파일 용도와 읽는 방법.
- [baseline.md](docs/baseline.ko.md): 동작 전달 경로, 안전층 인터페이스, 관절 매핑, 단위 및 모델 제한.
- [implementation.md](docs/implementation.ko.md): 현재 구현, 시작 문제 수정, 녹화/재생 기능, 검증과 적용 범위.
- [설정](src/orca_safety_v2/configs/v1_right.yaml): 108개 충돌 링크 쌍과 안전 간격, CBF, QP 및 단계 축소 검증 매개변수.

## Directory

- **`src/orca_safety_v2/`**: 충돌 거리 계산, CBF-QP 필터, 시뮬레이션 연결, 설정 및 재생 상태 복원.
- **`scripts/`**: 기하 검증, 동작 생성, OFF/ON 데이터 실험, 영상 출력 및 MuJoCo 대화형 재생.
- **`tests/`**: 기하 계산, 필터, 시작 상태 동기화, 연결 및 재생 테스트.
- **`docs/`**: 기존 인터페이스, 현재 구현, 후속 변경과 검증 범위.
- **`output/`**: 실험 데이터, 동작 입력, 원격 조작 로그, 웹캠 영상, OFF/ON 데모 및 환경 기록. [디렉터리 안내](output/README.ko.md)를 참고하세요.

## 설치

Orcahand 작업 공간의 최상위 디렉터리에서 실행합니다.

```bash
uv pip install --python orca_teleop/.venv/bin/python -e './Orca_safety[geometry,safety,test]'
```

## 원격 조작

`Orca_safety`에서 인접한 `orca_teleop` 디렉터리로 이동하여 실행합니다.

```bash
cd ../orca_teleop
.venv/bin/mjpython scripts/teleop_sim.py \
  --env right --version v1 --hand right --local --show-video \
  --retargeter adaptive_analytical \
  --urdf_path ../orcahand_description/v1/models/urdf/orcahand_right.urdf \
  --retarget-config ../orca_adaptive_test/configs/baseline.yaml \
  --collision-safety
```

- Safety OFF: `--collision-safety`를 제거합니다.
- 수동 모니터링 OFF 기준선: `--collision-monitor`로 명령을 수정하지 않고 거리를 기록합니다.
- Safety ON: `--collision-safety`를 사용합니다.
- 선택 사항: `--collision-safety-config PATH`, `--collision-safety-log NEW_PATH`.
- `--record-video NEW_FILE.mp4`는 웹캠 영상과 시간을 녹화합니다 (`--local` 필요). 지정하지 않으면 녹화하지 않습니다. [녹화와 구간 선택](scripts/README.ko.md#선택적-웹캠-원본-영상-녹화)을 참고하세요.
- `replay_viewer.py --mode OFF` 또는 `--mode ON`으로 시점을 회전하고 일시 정지하거나 단계별로 관찰할 수 있습니다. [대화형 재생](scripts/README.ko.md#mujoco-대화형-재생)을 참고하세요.

기본 로그 위치는 `Orca_safety/output/teleop/`입니다. 각 JSONL 프레임에는 실제 상태, nominal/safe 목표, 최소 간격, 활성 제약, 개입량, 최적화 상태, 소요 시간 및 실행 후 측정값이 포함됩니다.

### 터미널 메시지

아래 예시는 기본 안전 간격 **5 mm**를 기준으로 설명합니다. 설정을 변경했다면 해당 임계값을 적용하세요.

**1. 후보 목표 거부 및 fallback**

```text
COLLISION SAFETY | FALLBACK: nonlinear_command_margin_violation | min=5.77 mm | right_middle_mp__right_ring_mp
```

기하 검증에서 충돌이나 부족한 간격이 확인되어 이전에 검증된 목표를 사용합니다. `min=5.77 mm`는 fallback 목표 실행 후의 실제 최소 거리이며, 거부된 후보의 거리가 아닙니다. 따라서 표시된 거리가 5 mm보다 커도 fallback이 발생할 수 있습니다. 링크 쌍은 중지 `mp`와 약지 `mp`이며, `__`가 두 이름을 구분합니다.

거부된 후보의 최소 거리는 JSONL의 `rejected_min_distance_after`에서 확인합니다. `solver_status=solved`는 QP 계산 성공만 뜻하며, 비선형 검증 통과를 의미하지 않습니다.

**2. 실제 상태의 안전 간격 위반**

```text
COLLISION SAFETY | MEASURED MARGIN VIOLATION | min=4.98 mm | right_ring_ip__right_pinky_ip
```

제어 단계 실행 후 약지와 소지의 `ip` 링크 사이 실제 최소 간격이 4.98 mm로, 기본 5 mm보다 작습니다. 이는 안전 여유 위반이며 기하학적 충돌이 반드시 발생했다는 뜻은 아닙니다. 충돌 여부는 JSONL의 `measured_collision`로 확인하세요.

Fallback과 실제 간격 위반이 동시에 발생하면 터미널에는 `FALLBACK`이 우선 표시됩니다. 따라서 `FALLBACK ... | min=4.93 mm`는 후보 거부와 실행 후 간격 부족이 함께 발생했음을 나타냅니다.

**그 밖의 메시지**

- `COLLISION SAFETY READY`: 안전 필터가 활성화되었으며 뒤에 로그 경로가 표시됩니다. `COLLISION SAFETY MONITOR`는 모니터링만 수행합니다.
- `COLLISION SAFETY FALLBACK #100: ...`: 초기화/reset 이후 fallback 누적 100회입니다. 콜론 뒤에 원인이 표시되며, 첫 발생과 매 100회마다 출력됩니다.
- `FALLBACK: QP primal infeasible`: QP 제약을 만족할 수 없어 이전 검증 목표를 사용합니다. 최적화 오류나 잘못된 입력도 fallback을 유발하며 원인은 `error`에 기록됩니다.
- `Retargeter | 11.9 fps | retarget 84.22 ms`: 집계 구간의 평균 처리 속도와 호출 소요 시간입니다. 렌더링 FPS나 전체 원격 조작 지연을 의미하지 않습니다.

## Safety Layer 테스트 재현

자동화 테스트는 [tests/README.md](tests/README.ko.md), 자세 검증, 입력 생성 및 OFF/ON 실험은 [scripts/README.md](scripts/README.ko.md)를 참고하세요.


## 통합 자동 Benchmark

동일한 nominal 입력으로 Safety OFF/ON을 실행하고 6개 핵심 지표를 비교합니다.
이 디렉터리에서 `../orca_teleop/.venv/bin/python scripts/run_safety_benchmark.py`를 실행하세요.
`--scenario NAME`, `--mode off|on|both`(기본값 both), 설정 및 출력 경로를 지원합니다.
프레임별 JSONL/CSV, 집계 CSV/JSON 및 Markdown 보고서를 생성합니다.
시나리오 설명, 지표 정의 및 결과는 [실험 보고서](results/safety_benchmark/benchmark_report.ko.md)를 참고하세요.
