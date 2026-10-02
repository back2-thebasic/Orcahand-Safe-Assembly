# 자동화 테스트

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

이 디렉터리는 안전 모듈의 코드 동작을 검사합니다. OFF/ON 데이터 실험과 재생은 [scripts/README.md](../scripts/README.ko.md)를 참고하세요.

- `test_geometry.py`: 거리, 관절 매핑, 유한 차분 기울기 및 먼 링크 쌍의 사전 선별.
- `test_filter.py`: CBF-QP, 관절/단계 제한, 단계 축소 검증 및 실패 시 fallback.
- `test_integration.py`: 시뮬레이션 연결, 시작 자세 동기화, 수동 모니터링 및 로그.
- `test_replay.py`: 구간 선택 시 시뮬레이션 이력 보존, 웹캠 시간 변환 및 관절 순서 검사.

[메인 README](../README.ko.md)에 따라 의존성을 설치한 후 작업 공간의 최상위 디렉터리에서 실행합니다.

```bash
orca_teleop/.venv/bin/python -m pytest Orca_safety/tests -q
```

테스트 통과는 검사한 코드 동작이 예상과 일치한다는 뜻이며, 연속 운동이나 실제 로봇 손의 안전을 보장하지 않습니다.
