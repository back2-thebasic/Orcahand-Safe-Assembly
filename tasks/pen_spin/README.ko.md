# OrcaHand Pen Spin

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

OrcaHand v1 펜 돌리기: 모델, 공식 시연, 시연 수정, 정책, 배포, 펜 자세 추정.

```text
pen_spin/
├── models/
│   ├── pen/              공식 펜 모델 및 텍스처
│   └── hand/             OrcaHand 메시
├── hardware/             펜 STEP·3MF 출력 파일, 시각 마커
├── data/
│   ├── official/         공식 시연 40개, 인덱스, 레시피
│   ├── converted/        시연 40개의 최초 변환 결과
│   └── demo_repair/      시연 수정 파이프라인
│       ├── scripts/      최초 변환부터 학습 데이터 내보내기까지의 스크립트
│       ├── assets/       리타기팅 설정
│       ├── base_scene.xml  정렬용 기본 장면
│       └── results/      clip_002, clip_004의 단계별 결과
├── policies/             학습된 정책
├── deploy/               배포 번들 내보내기(export_bundle.py) 및 런타임(penspin_deploy/)
├── perception/           펜 자세 추정
│   ├── observer.py       카메라 → 손바닥 좌표계의 펜 자세, ZMQ로 배포 프로그램에 전송
│   ├── pen_perception/   펜 태그 인식·풀이, 손목 태그, 통신
│   ├── config/           카메라 내부 파라미터, 손목 태그 자세
│   └── tools/            카메라 캘리브레이션, 렌더링 이미지 검증
├── view.py               재생 도구
└── requirements.txt      view.py 의존성
```

- 수정된 clip_002, clip_004는 `data/demo_repair/results/trajectory/`에 있으며, `reorient/wuji-mjlab/src/wuji_mjlab/tasks/pen_spin/data/`로 내보냈습니다.
- 펜 모델 `models/pen/assets/pen_tracking.xml`은 공식 원본 XML입니다. 질량 30 g은 로딩 코드에서 설정하며 XML에는 없습니다.

## 정책

세 정책 모두 clip_002 + clip_004로 학습했습니다. 각 폴더에는 `best.pt`와 `log/params/`(`env.yaml`, `agent.yaml`)가 있습니다.

| 폴더 | 학습 방식 | 체크포인트 |
|---|---|---|
| `clip002_004/` | 랜덤화·외란 없음, 5000회 반복 | `model_4000.pt` |
| `clip002_004_dr_finetune/` | `clip002_004`의 4000회 지점에서 랜덤화·외란을 추가해 6000회 추가 학습 | `model_10000.pt` |
| `clip002_004_dr_scratch/`(권장) | 랜덤화·외란·커리큘럼으로 처음부터 10000회 학습 | `model_9999.pt` |

dr = domain randomization(도메인 랜덤화). dr 정책의 학습 설정:

- 랜덤화: 관절 감쇠 0.6～1.6배, 회전 관성 0.5～2배, 관절 마찰 0～0.03 N·m, 모터 최대 힘 0.6～1.1배, 손가락 링크 질량·관성, 엔코더 편차 ±0.01 rad, 펜 길이 205～265 mm, 펜 지름 14～16 mm.
- 외란: 펜 밀기(펜 무게의 1～2배), 손가락 끝 충격(손가락 끝 링크 무게의 최대 4배).
- 커리큘럼: 외란 강도를 0에서 최대까지 점차 증가.

### 평가

학습 시뮬레이션에서 끝까지 돌린 횟수(시연당 64회, 총 128회)와 표준 조건의 추적 오차:

| 정책 | 표준 조건 | 물리 랜덤화 | 랜덤화 + 최대 외란 | 위치 오차 | 방향 오차 |
|---|---|---|---|---|---|
| `clip002_004` | 127/128 | 37/128 | 0/128 | 0.8～1.1 cm | 3.5～4° |
| `clip002_004_dr_finetune` | 128/128 | 123/128 | 158/256 | 1.2～1.5 cm | 5～6° |
| `clip002_004_dr_scratch` | 128/128 | 127/128 | 179/256 | 1.3～1.5 cm | 5～6° |

최대 외란 열은 두 차례 합계(256회)이며 관측 노이즈를 추가했습니다.

## 배포

`policies/clip002_004_dr_scratch/deploy/`는 권장 정책의 배포 번들(ONNX 정책, 시뮬레이션 장면, 시연 클립, 대조 데이터)이며 `deploy/`의 프로그램이 사용합니다. 카메라 노이즈를 넣은 독립 MuJoCo 트윈에서 두 시연 모두 20/20 완주, 위치 오차 1.3～1.4 cm. 실행 지연 20 ms 이내는 영향이 거의 없고, 40 ms에서는 대부분 펜을 떨어뜨립니다.

## 펜 자세 추정

카메라가 펜 양 끝의 태그 18개를 인식해 펜 자세를 풀고, 손목 AprilTag를 통해 손바닥 좌표계로 변환해 배포 프로그램에 보냅니다(`--hand real --pen camera`). 코드는 Wuji에서 이식했으며 특정 카메라에 의존하지 않습니다.

- MuJoCo 렌더링 이미지로 검증(4개 시점 × 165프레임): 모두 풀림, 위치 오차 중앙값 0.2～1.5 mm, 펜대 방향 오차 중앙값 0.1～0.2°, 양 끝 뒤바뀜 없음.
- `config/`의 카메라 내부 파라미터와 손목 태그 자세는 현재 임시값이며, observer는 이 상태로 실행을 거부합니다. 내부 파라미터는 `tools/calibrate_camera.py`로 캘리브레이션합니다.
- 시뮬레이션 모델의 손목 태그는 손등에 있어, 손바닥이 위를 향하면 카메라에 보이지 않습니다. 실물 태그는 카메라에 보이고 손바닥과 강체로 연결된 위치에 달고, 손바닥 좌표계에서의 자세를 측정해 `config/wrist_tag.yaml`에 입력해야 합니다.
