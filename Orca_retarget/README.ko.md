# ORCA Hand Retargeting 테스트 비교
[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

ORCA v1 오른손 · MediaPipe · Adaptive Analytical · MuJoCo 시뮬레이션

이 저장소는 YAML 파라미터와 테스트 영상을 공유하기 위한 것입니다. 설정 변경 사항과 3개 시나리오의 비교 영상이 포함되어 있습니다. README에서는 호환성이 더 좋은 H.264 미리보기 영상을 사용하며, 원본 MOV 파일은 각 영상 아래의 링크에서 다운로드할 수 있습니다.

## 설정 및 변경 사항

### A: 공식 원본 설정

**파일: [baseline.yaml](configs/baseline.yaml)**

- 용도: baseline으로 사용하며 추가 fingertip offset을 유지
- `w_pos: 1.0`, `w_dir: 10.0`, `w_full_hand: 1.0`
- `norm_delta: 0.04`, `lp_alpha: 1.0`(출력 low-pass smoothing 없음)

### B: Zero-offset 설정

**파일: [baseline_zero_offsets.yaml](configs/baseline_zero_offsets.yaml)**

- A와 비교: 다섯 손가락의 `fingertip_offsets_m`을 모두 `[0, 0, 0]`으로 설정하여 URDF fingertip frame 원점을 직접 사용하고 fingertip offset을 제거
- 그 외 YAML 파라미터는 A와 동일
- 변경 목적: 이미 정의된 fingertip 위치에 손가락 끝 길이가 추가로 중첩되는 것을 방지
- 주의: 이 변경은 자동 scale calibration에도 영향을 주므로, 고정된 scale에서 수행한 순수한 기하학적 비교는 아님

### C: 파라미터 조정 설정

**파일: [last.yaml](configs/last.yaml)**

- 현재 B와 비교: `w_pos: 1.0 → 2.0`, `w_full_hand: 1.0 → 0.5`
- 변경 목적: 위치 제약의 가중치를 높이고 전체 손 형상 제약의 가중치를 낮춰 pinch 동작이 개선되는지 관찰하는 동시에 pose가 저하되는지 확인

> B/C를 사용하려면 retargeting 코드가 `fingertip_offsets_m`을 지원해야 합니다. 수정되지 않은 upstream 코드는 이 필드를 읽지 않을 수 있습니다.

## Fingertip offset: 변경 이유와 영향

### 기본 설정에 offset이 있는 이유

Orca의 retargeting 구현은 URDF 좌표계 원점과 실제 손가락 지문면 위치 사이의 차이를 보정하기 위해 경험적으로 정한 offset을 사용합니다. 이는 기하학적 위치 보정이며, 모터 영점이나 joint angle offset이 아닙니다.

Upstream 코드에서는 이 값들을 수동으로 조정한 경험값으로 명시하고 있으며, 향후 collision geometry 정의로 대체할 예정이므로 모든 모델 버전에 적합하다고 볼 수는 없습니다. 기본적으로 엄지, 검지, 중지, 약지, 소지에 대해 각각 로컬 z축 방향으로 **30.5, 43.3, 45.3, 45.3, 38.3 mm**가 추가됩니다.

출처: [Orca upstream offset 정의](https://github.com/orcahand/orca_teleop/blob/main/src/orca_teleop/retargeting/constants.py), [관련 geometry utility 설명](https://github.com/orcahand/orca_teleop/blob/main/src/orca_teleop/retargeting/utils.py).

### Zero offset 설정 후 pinch가 가능한 이유

Optimizer가 사용하는 fingertip 위치는 다음과 같습니다.

- **기본 offset:** URDF fingertip frame의 위치에 해당 frame으로 회전된 로컬 offset을 더한 값
- **Zero offset:** URDF fingertip frame의 위치를 직접 사용

현재 v1 URDF에는 fixed joint를 통해 fingertip 위치가 이미 정의되어 있습니다. 예를 들어 검지 fingertip frame은 distal link를 기준으로 `[-9, 0, 35] mm`에 있습니다. 이 fixed joint에는 회전이 없으므로 기본 offset을 더하면 계산 지점은 `[-9, 0, 78.3] mm`가 됩니다. 이 값들은 손가락 전체 길이가 아니라 distal link 기준 좌표입니다.

추가 offset으로 인해 계산 지점이 실제 접촉 위치에서 벗어나면 optimizer는 이러한 "가상 fingertip"을 목표에 가깝게 만들 수 있지만, 시뮬레이션의 실제 fingertip 사이에는 여전히 간격이 남을 수 있습니다. Zero offset을 설정하면 계산 지점이 URDF fingertip frame으로 돌아오므로 실제 fingertip의 정렬과 pinch가 개선될 수 있습니다.

**이번 관찰 결과:** 원본 설정에서는 엄지와 검지가 pinch를 완료하지 못했지만, zero offset을 설정한 후에는 완료할 수 있었습니다.

이 결과는 "경험적으로 정한 offset이 현재 모델과 맞지 않을 수 있다"는 설명을 뒷받침하지만, 기본 설정이 모든 모델에서 잘못되었다거나 offset이 유일한 원인임을 입증하지는 않습니다. URDF fingertip frame이 실제 접촉면에 정확히 위치한다고 단정할 수도 없으므로, 향후 더 작고 기하학적으로 검증된 보정이 필요할 수 있습니다.

**Zero offset은 retargeting에 사용되는 계산 지점만 변경하며 모델 형상, collision body 또는 joint limit은 변경하지 않습니다. 또한 smoothing이나 grip force 제어를 직접 제공하지 않습니다.** 시각적으로 닫혀 보인다고 해서 물리적 접촉이나 안정적인 grasp가 검증된 것은 아닙니다.

## YAML 다운로드 및 실행 방법

### 1. YAML을 어디에 배치해야 하나요?

이 저장소에서 **Code → Download ZIP**을 클릭하고 압축을 푼 다음, `configs/`를 로컬 Orca 프로젝트의 `orca_adaptive_test/` 안에 배치하여 아래 구조를 유지합니다. 개별 YAML 페이지에서 Raw를 클릭해 해당 파일만 다운로드할 수도 있습니다.

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

`.venv` 또는 Python의 site-packages 안에는 배치하지 마세요. 같은 이름의 실험 파일이 이미 있다면 이전 버전을 먼저 보관하여 기존 영상과 해당 파라미터의 대응 관계가 사라지지 않도록 하세요.

### 2. 실행 전제 조건

B/C의 zero-offset 필드를 사용하려면 코드 지원이 필요합니다. `orca_teleop/src/orca_teleop/retargeting/adaptive_analytical.py`를 열고 `_build_frame_indices()`에서 `fingertip_offsets_m`을 읽고 있는지 확인하세요. 이번 테스트에 사용한 코드는 이 기능을 지원합니다.

지원하지 않는 경우 함수 내부의 들여쓰기를 유지하면서 함수 끝부분에서 `_frame_offsets`를 설정하는 짧은 코드 구간을 아래 코드로 교체하고, 나머지 코드는 변경하지 마세요.

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

이 기능을 지원하지 않으면 B/C YAML을 다운로드하는 것만으로는 zero offset이 적용된다고 보장할 수 없습니다. A에는 이 필드가 설정되어 있지 않으므로 기존 offset을 계속 사용합니다. 실행 환경에서 수정한 소스 코드를 import하는지 확인하세요.

### 3. A, B 또는 C를 선택한 후 실행

먼저 `orca_teleop`으로 이동합니다. 경로는 실제 환경에 맞게 바꾸세요.

```bash
cd /xxx/Orcahand/orca_teleop
```

세 가지 중 하나를 선택하여 동일한 터미널에서 이번 실행에 사용할 설정을 지정합니다.

```bash
# A: 원본 설정
RETARGET_CONFIG=../orca_adaptive_test/configs/baseline.yaml
```

```bash
# B: 다섯 손가락 zero offset
RETARGET_CONFIG=../orca_adaptive_test/configs/baseline_zero_offsets.yaml
```

```bash
# C: 현재 최신 설정
RETARGET_CONFIG=../orca_adaptive_test/configs/last.yaml
```

그런 다음 공통 실행 명령을 실행합니다. 가상 환경을 먼저 activate할 필요 없이 가상 환경의 mjpython을 직접 사용합니다.

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

각 줄 끝의 백슬래시 뒤에 공백을 넣지 마세요. v2 YAML은 이 v1 명령에 사용할 수 없습니다.

주의할 점: 동일한 포트를 사용하는 시뮬레이션 두 개를 동시에 실행하지 마세요. 녹색 skeleton은 정상적으로 보이지만 시뮬레이션이 움직이지 않는 경우 `Publisher connected` 로그를 확인하고, `lsof -nP -iTCP:50051 -sTCP:LISTEN`으로 이전 인스턴스가 남아 있는지 확인하세요.

## S01: 손 펴기, 반쯤 쥐기, 주먹 쥐기

동작: 손 펴기 → 반쯤 쥐기 → 주먹 쥐기 → 손 펴기, 3회 반복

### A: 원본 설정

https://github.com/user-attachments/assets/c61373b0-8059-412d-8560-4a3ba6c9c63a

[원본 MOV 보기 또는 다운로드](videos/S01/S01-baseline.mov)

### B: Zero offset

https://github.com/user-attachments/assets/29e975da-077b-48ba-9e53-38181359c819

[원본 MOV 보기 또는 다운로드](videos/S01/S01-zero%20offsets.mov)

### C: 현재 최신 설정

https://github.com/user-attachments/assets/25fd48fd-ee29-44b5-9cb1-a7689365d1fd

[원본 MOV 보기 또는 다운로드](videos/S01/S01-last.mov)

**이 시나리오의 결론:** 모든 YAML 설정에서 동작을 잘 수행했습니다. B에서는 엄지 retargeting이 그다지 좋지 않았습니다.

## S02: 엄지-검지 저속 pinch 및 release

동작: 천천히 접근 → pinch → release, 3회 반복

### A: 원본 설정

https://github.com/user-attachments/assets/4c55b450-9961-4b50-bdd5-de74a9e2b264

[원본 MOV 보기 또는 다운로드](videos/S02/S02-baseline.mov)

### B: Zero offset

https://github.com/user-attachments/assets/8d29cfac-a2db-4344-89fa-7863eeb2a6a5

[원본 MOV 보기 또는 다운로드](videos/S02/S02-zero%20offsets.mov)

### C: 현재 최신 설정

https://github.com/user-attachments/assets/c37a71f9-74a2-4a70-8b3a-697967042f29

[원본 MOV 보기 또는 다운로드](videos/S02/S02-last.mov)

**이 시나리오의 결론:** 공식 기본 설정에서는 pinch 동작을 완료하지 못했으며 엄지와 검지 사이에 계속 간격이 남았습니다. B와 C는 이 동작을 비교적 잘 수행했지만, B에서는 pinch 시 엄지의 방향이 다소 부자연스러웠습니다.

## S03: 세 손가락 grasp 및 release

동작: 엄지-검지-중지 pinch → release, 3회 반복

### A: 원본 설정

https://github.com/user-attachments/assets/52aaa9e6-61f5-478b-94e8-735014e2cfac

[원본 MOV 보기 또는 다운로드](videos/S03/S03-baseline.mov)

### B: Zero offset

https://github.com/user-attachments/assets/57ebade0-5c62-416b-878a-cc3ec69e8192

[원본 MOV 보기 또는 다운로드](videos/S03/S03-zero%20offsets.mov)

### C: 현재 최신 설정

https://github.com/user-attachments/assets/505e5a70-ac18-4eff-b557-5c8502d0ffa5

[원본 MOV 보기 또는 다운로드](videos/S03/S03-last.mov)

**이 시나리오의 결론:** 빈손 테스트는 grasp pose만 검증하며 실제로 물체를 잡을 수 있다는 의미는 아닙니다. 공식 설정에서는 여전히 pinch를 완료하지 못했습니다. B와 C는 비교적 잘 수행했지만, B에서는 손을 펼 때 엄지 retargeting이 충분히 좋지 않았으며 엄지의 IP joint(손가락 끝에 가장 가까운 joint) retargeting이 다소 부자연스러웠습니다.

## 영상 추가 방법

1. 원본 영상은 각각 `videos/S01/`, `S02/`, `S03/` 디렉터리에 저장합니다.
2. 일반 저장소 영상 링크로는 README 내장 플레이어가 안정적으로 생성되지 않습니다. README에서는 GitHub Markdown 편집 영역에 업로드한 후 생성되는 `user-attachments` 주소를 사용합니다.
3. 브라우저 호환성을 높이기 위해 미리보기 영상에는 H.264 MP4를 사용합니다.
4. 녹화 시 항상 사람의 손과 시뮬레이션을 함께 표시하고 calibration gesture를 고정합니다. YAML을 업데이트할 때 이 페이지의 설명도 함께 업데이트하세요. 기존 영상이 이전 파라미터에 해당하는 경우 이전 파라미터를 덮어쓰지 말고 새 설정 번호를 추가하세요.

GitHub 영상 첨부 방법은 [공식 문서](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/attaching-files)를 참고하세요.

## 파일 구조

```text
configs/                 공유용 YAML 전체
videos/S01/              S01-baseline.mov, S01-zero offsets.mov, S01-last.mov
videos/S02/              S02-baseline.mov, S02-zero offsets.mov, S02-last.mov
videos/S03/              S03-baseline.mov, S03-zero offsets.mov, S03-last.mov
README.md                영어 버전(저장소 기본)
README.zh-CN.md          중국어 버전
README.ko.md             한국어 버전
```
