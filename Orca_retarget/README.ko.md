# ORCA Hand 리타게팅 개선 및 시뮬레이션 비교

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

이 문서는 공식 참조 설정을 출발점으로 손끝 기준점, 최적화 가중치, 관절 제한 및 집기 목표를 단계적으로 개선한 과정을 기록합니다. 현재는 엄지와 다른 손가락의 협응을 개선하고, 충돌 안전 계층을 적용했을 때 동작이 어떻게 달라지는지 살펴보는 데 중점을 둡니다.

## 최신 비교 영상

두 영상은 동일한 MediaPipe 랜드마크 기록인 `mediapipe-02.jsonl`을 사용합니다. 각 설정에서 유효한 시뮬레이션 출력 807프레임을 생성하고 동기화하여 재생합니다.

녹화 화면의 왼쪽부터: Baseline → Retarget 03 → Retarget 04.

### 안전 계층 OFF

세 설정 모두 추가 Safety Layer를 끕니다. MuJoCo의 기존 충돌 처리와 물리적 관절 제한은 계속 적용됩니다.

[비교 영상 보기 또는 다운로드](videos/retarget-comparison/comparison_Safety_OFF.mov)

https://github.com/user-attachments/assets/20ff53f4-c8f9-4d7f-b451-67a239d0fad6

### 안전 계층 ON

세 설정 모두 동일한 v2 CBF-QP Safety Layer를 사용하며, 리타게팅 출력을 필터링한 뒤 시뮬레이션에 전달합니다.

[비교 영상 보기 또는 다운로드](videos/retarget-comparison/comparison_Safety_ON.mov)

https://github.com/user-attachments/assets/95f271ae-0a19-46e6-8f49-8c624a4a2b04

## 설정 변경 내용

공식 참조 설정의 이름은 `baseline.yaml`입니다. 이 디렉터리의 후속 설정은 `retarget_01.yaml`부터 `retarget_04.yaml`까지의 이름을 사용합니다. 번호는 변경 순서를 나타내며, 이후 설정이 모든 동작에서 더 우수하다는 뜻은 아닙니다.

### Baseline: 공식 참조 설정

파일: [baseline.yaml](configs/baseline.yaml)

기존의 추가 손끝 오프셋을 유지합니다. 가중치는 `w_pos: 1.0`, `w_dir: 10.0`, `w_full_hand: 1.0`입니다.

현재 코드에서 공식 참조 설정을 실행한 결과입니다.

### Retarget 01: 추가 손끝 오프셋 제거

파일: [retarget_01.yaml](configs/retarget_01.yaml)

Baseline을 바탕으로 다섯 손가락의 `fingertip_offsets_m`을 모두 `[0, 0, 0]`으로 설정하여 URDF 손끝 프레임의 원점을 직접 사용합니다. 나머지 설정값은 변경하지 않습니다.

이는 리타게팅 기준점을 변경하는 것이며, 모터나 관절의 영점 또는 충돌 형상을 변경하는 것은 아닙니다. 손끝 기준점 변경은 자동 스케일 보정에도 영향을 줄 수 있습니다.

### Retarget 02: 위치 및 전체 손 형상 가중치 조정

파일: [retarget_02.yaml](configs/retarget_02.yaml)

Retarget 01을 바탕으로 `w_pos`를 1.0에서 2.0으로, `w_full_hand`를 1.0에서 0.5로 변경합니다.

### Retarget 03: 상대 손끝 위치 목표 추가

파일: [retarget_03.yaml](configs/retarget_03.yaml)

Retarget 02에 다음 설정을 추가합니다.

```yaml
retarget:
  w_relative_tip: 1.0
```

관련 Python 코드에 상대 손끝 위치 손실과 해석적 기울기를 추가했습니다. 엄지 손끝에서 나머지 네 손끝으로 향하는 목표 벡터를 추적하고, 기존 집기 활성화 가중치를 그대로 사용합니다. 두 손끝의 위치 협응을 강조하지만, 손가락 끝의 접촉면이 서로 마주 보도록 직접 요구하지는 않습니다.

이번 비교에서는 **시뮬레이션 관절 제한 좌표 보정**도 활성화합니다.

기존 리타게팅 관절 범위와 MuJoCo 제어 각도의 기준 좌표가 일치하지 않아, 시뮬레이션에서 실행 가능한 자세가 최적화 범위에서 제외되거나 시뮬레이션에서 나중에 잘리는 목표 각도가 생성될 수 있습니다. 이를 보정하기 위해 MuJoCo 관절 범위와 액추에이터 제어 범위의 교집합을 최적화기의 유효 상·하한으로 사용하여, 최적화와 실행이 일치하는 각도 범위를 사용하도록 합니다.

### Retarget 04: 집기 목표 거리 재매핑 추가

파일: [retarget_04.yaml](configs/retarget_04.yaml)

Retarget 03의 상대 위치 목표를 유지하고 다음 설정을 추가합니다.

```yaml
retarget:
  relative_tip_remap:
    enabled: true
    close_cm: 1.0
    start_cm: 5.0
    min_target_cm: 0.5
```

관련 함수는 목표 벡터의 방향을 유지하면서 길이를 조정합니다. 거리는 스케일 보정과 좌표 정렬을 거친 로봇 기준 센티미터 단위입니다. 5 cm 이상은 그대로 유지하고, 1~5 cm는 0.5~5 cm로 선형 매핑하며, 1 cm 이하는 일반적으로 0.5 cm를 사용합니다. 0에 매우 가까운 구간에서는 방향이 정의되지 않는 문제를 처리하기 위해 연속적으로 0으로 줄어듭니다.

이 변경은 집기에 가까워질 때 손끝을 더 모으도록 유도할 수 있지만, 작은 간격의 표현을 압축하거나 충돌로 인한 동작 차단을 늘릴 수도 있습니다. 이번 비교에서는 시뮬레이션 관절 제한 보정도 활성화합니다.

## Retarget 및 Teleoperation 관련 코드 변경

새 YAML 파라미터에는 해당 기능을 지원하는 코드가 필요합니다. 설정 파일만 교체해서 모든 개선 기능을 활성화할 수는 없습니다. 아래 경로는 로컬 `orca_teleop/` 프로젝트 기준입니다.

### `src/orca_teleop/retargeting/adaptive_analytical.py`

- `fingertip_offsets_m`을 읽어 손끝 오프셋을 0으로 설정할 수 있도록 했습니다.
- 최적화기가 MuJoCo와 일치하는 유효 관절 범위를 사용하도록 `set_sim_joint_limits()`를 추가했습니다.
- Retarget 03을 위해 `w_relative_tip`으로 제어하는 상대 손끝 위치 손실과 기울기를 추가했습니다.
- Retarget 04를 위해 `relative_tip_remap`에 따라 목표 거리를 조정하는 `remap_relative_vectors()`를 추가했습니다.

Retarget 02는 기존 `w_pos`, `w_full_hand` 파라미터만 조정하므로 새로운 손실 항이 필요하지 않습니다. 상대 손끝 항과 거리 재매핑은 기본적으로 비활성화되어 있습니다.

### 실시간 Teleoperation 연동

- `src/orca_teleop/sim_diagnostics.py`: `simulation_joint_limits()`가 MuJoCo 관절 범위와 액추에이터 제어 범위의 교집합을 추출합니다.
- `scripts/teleop_sim.py`, `src/orca_teleop/sim.py`, `src/orca_teleop/pipeline.py`: `--sim-joint-limits`를 통해 해당 범위를 리타게팅 모듈에 전달하고, 사람 손 입력 처리를 시작하기 전에 관절 제한 보정을 활성화합니다.
- `--collision-safety`를 사용하면 시뮬레이션 측에서 독립적인 `orca_safety_v2` 안전 계층을 호출하여 리타게팅 출력을 필터링한 뒤 MuJoCo에서 실행합니다.

YAML 선택, 관절 제한 보정 활성화, 안전 계층 활성화는 서로 독립적인 세 가지 설정입니다. YAML을 변경해도 나머지 두 기능이 자동으로 켜지지는 않습니다.

## 이전 실험

이전 A/B/C 설정과 S01~S03 영상은 당시의 설정을 그대로 유지합니다. 현재의 관절 제한 보정이나 새 최적화 항을 사용한 것으로 소급해서 설명해서는 안 됩니다.

이전 설정 설명, 실행 방법 및 S01~S03 영상은 [이전 실험 기록](README.history.ko.md)을 참고하세요.

## 설치 및 실행

원래 경로에 덮어쓸 수 있는 전체 코드 파일 5개를 제공합니다. 코드를 직접 편집할 필요 없이 [설치 안내](INSTALL.md)를 참고하세요. 녹화 모듈과 안전 계층 구현은 포함하지 않습니다. configs에는 baseline.yaml과 retarget_01.yaml~retarget_04.yaml만 유지하며 과거 문서의 경로도 갱신했습니다.
