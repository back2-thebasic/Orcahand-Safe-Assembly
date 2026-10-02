# 기존 시스템 인터페이스와 Safety Layer 연결

[English](baseline.md) | [中文](baseline.zh.md) | [한국어](baseline.ko.md)

현재 로봇 모델은 **v1 오른손**이며, v2는 안전 소프트웨어 버전입니다. 구현과 검증은 [implementation.ko.md](implementation.ko.md)를 참고하세요.

## 동작 파이프라인

```text
손 랜드마크 → Retargeter → OrcaJointPositions → actions_q
→ 시뮬레이션 실행기 → Safety Layer (선택 사항) → MuJoCo env.step
```

Retargeter는 17개 관절의 절대 각도를 도(degree) 단위로 출력합니다. 시뮬레이션 실행기는 관절 이름에 따라 순서를 맞추고 rad 단위로 변환한 뒤 안전 계층에 전달합니다.

- `q_nominal`: 원래 목표값.
- `q_current`: MuJoCo의 실제 상태에서 읽은 관절 각도.
- `q_safe`: 안전 계층이 반환하여 시뮬레이션에 전달하는 목표값.

안전 계층은 아직 실제 로봇 손에 연결되지 않았습니다. OFF/ON 비교에서는 동일한 모델, 초기 상태와 nominal 입력을 사용합니다.

## 관절 매핑

MuJoCo 액추에이터 순서:

```text
wrist → thumb(mcp, abd, pip, dip)
→ index, middle, ring, pinky (abd, mcp, pip)
```

MuJoCo와 Pinocchio의 관절은 이름으로 매핑합니다. 실제 관절 위치는 `jnt_qposadr`로 읽습니다. URDF 기하 계산에는 기준 오프셋을 반영해야 합니다: `q_URDF = q_MuJoCo − model.qpos0`. 배열 위치만 기준으로 각도를 직접 복사하면 안 됩니다.

## 기하 모델과 제한값의 출처

- **충돌 기하**: v1 URDF의 collision mesh를 Pinocchio/FCL로 조회합니다.
- **관절 범위**: MuJoCo `jnt_range`와 `actuator_ctrlrange`의 교집합입니다.
- **스텝당 움직임 제한**: URDF 속도 제한 × 제어 주기입니다. 현재 100 rad/s × 0.01 s = 1 rad/step이며, `max_step_rad`로 더 엄격하게 제한할 수 있습니다.
