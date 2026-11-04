# OrcaHand Pen Spin

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

이 디렉터리에는 펜 돌리기 작업에 사용한 모델, 공식 시연 데이터, 변환 데이터 및 실험 결과가 정리되어 있습니다.

```text
pen_spin/
├── models/
│   ├── pen/              공식 MuJoCo 펜 모델, 텍스처 및 출처 정보
│   └── hand/             재생에 필요한 OrcaHand 메시 리소스
├── hardware/             STEP·3MF 출력 파일 및 시각 마커 리소스
├── data/
│   ├── official/         공식 시연 40개 및 인덱스·레시피
│   └── converted/        OrcaHand로 처음 리타기팅한 궤적 20개
├── results/
│   ├── inspection-*     클립 003/006의 분석 결과 및 이미지
│   └── optimized003-*   클립 003의 단계별 최적화 결과 및 전후 비교 인덱스
├── view.py               참조 궤적 및 자유 동역학 재생 도구
├── portable_scene.py     재생 도구에서 사용하는 장면 경로 유틸리티
└── requirements.txt      재생에 필요한 Python 의존성
```

`data/converted`에는 최초 변환 결과가, `results`에는 이후 분석 및 최적화 결과가 저장되어 있습니다. 현재 궤적은 아직 참조 후보이며, 자유 동역학 시뮬레이션에서 펜 돌리기에 성공하지 못했습니다. RL 학습은 시작하지 않았습니다.

펜 모델의 진입 파일은 `models/pen/assets/pen_tracking.xml`이며, 제공된 디렉터리 구조의 `models/pen/textures/`를 참조합니다. 이 파일은 공식 원본 XML입니다. 공식 로딩 코드는 별도로 질량을 30 g으로 설정하고 관성 모멘트를 같은 비율로 조정하지만, 이 조정은 해당 XML에 반영되어 있지 않습니다.
