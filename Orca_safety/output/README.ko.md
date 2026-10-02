# 실험 출력 및 재생 데이터

[中文](README.zh.md) | [English](README.md) | [한국어](README.ko.md)

이 디렉터리는 Safety Layer 실험 결과, 동작 입력, 원격 조작 기록 및 데모 영상을 저장합니다. 사용 방법은 [scripts/README.md](../scripts/README.ko.md)를 참고하세요.

## 디렉터리

```text
output/
├── comparison/          일곱 가지 합성 동작의 OFF/ON 로그, 요약 및 그래프
├── video/
│   ├── demo_video/      네 가지 동작의 OFF/ON 데모 영상
│   └── camera/          웹캠 원본 영상과 시간 정렬 파일
├── passive/             다섯 정적 자세, poses.json, 거리/기울기 검증
├── replay/              합성 특징점을 retargeting하여 생성한 재생 입력
├── retarget-comparison/ 위 입력의 OFF/ON 로그와 요약
├── synthetic/           검지/중지 교차 등 직접 생성한 동작 입력
├── teleop/              실제 원격 조작의 목표, 측정 상태 및 안전 로그
└── environment/         과거 실험의 Python 및 패키지 버전
```

## 문서 안내

- [네 가지 동작 영상 데모](video/demo_video/README.ko.md): OFF/ON 동작을 시각적으로 비교합니다.
- [일곱 가지 합성 동작 비교](comparison/README.ko.md): 충돌, 간격 위반, 동작 개입 및 실행 시간 결과.
- [Retarget 재생 요약](retarget-comparison/summary.json): 합성 특징점을 기존 retargeter에 입력한 OFF/ON 결과이며, 실제 촬영 데이터 실험이 아닙니다.

## 파일 안내

- **`.jsonl`**: 단계별 로그 또는 합성 동작 입력입니다. 실행 로그는 보통 `event=start`에 설정, 관절 순서와 단위를 저장하고, 이후 `event=frame`으로 각 단계를 기록합니다.
- **`.npy` / `.npz`**: NumPy 형식의 목표 궤적이나 특징점 데이터입니다. `summary.json`은 실험 요약을 저장합니다.
- **웹캠 `.timing.json`**: 원본 영상의 구간 시간을 원격 조작 로그에 대응시킵니다. `--video-timing`과 함께 사용합니다.

주요 로그 필드: `q_nominal`은 원래 목표, `q_safe`는 안전층 출력 목표, `q_measured`는 실행 후 실제 상태입니다. `min_distance_measured`는 실행 후 최소 간격이며, `fallback`과 `error`는 fallback 여부와 원인입니다.

각도는 rad, 거리는 m 단위입니다. 관절 배열 순서는 `joint_names`를 따릅니다. 목표 자세의 거리와 실제 운동의 거리는 다르며, 안전 간격 부족이 반드시 충돌을 뜻하지는 않습니다.
