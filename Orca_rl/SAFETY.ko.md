# RL 큐브 정책을 안전 레이어에 연결해 실행하기

[English](SAFETY.md) | [中文](SAFETY.zh-CN.md) | [한국어](SAFETY.ko.md)

학습된 손 안 큐브 회전 정책(v1 오른손, MuJoCo)을 각자의 컴퓨터에서 `Orca_safety` 뒤에 연결해 실행하는 데
필요한 것이 모두 여기에 있습니다: 코드, 학습된 모델 6개, 명령어. 학습은 필요 없습니다.

## 무엇을 하는가

10 ms 제어 스텝마다 정책이 관절 목표 17개(`q_nominal`)를 출력합니다. 안전 레이어가 켜져 있으면 이 목표가
측정된 관절 각도와 함께 `orca_safety_v2.filter.CBFSafetyFilter`를 거치고, 반환된 `q_safe`가 서보로 전달됩니다.
`q_safe`는 정책의 "이전 목표"로도 되돌려집니다. 두 모드 모두 매 스텝 뒤에 필터가 쓰는 것과 같은 108개 링크 쌍으로
손의 자기 간격을 측정합니다.

필터는 설치된 `orca_safety_v2`에서 가져옵니다. 아래처럼 `Orca_safety`를 편집 가능 모드로 설치하면
**`Orca_safety/src/` 아래에서 고친 내용이 그대로 테스트됩니다.** 복사본은 없습니다.

## 설치 (약 5분)

Python 3.11. `Orcahand-Safe-Assembly/`가 들어 있는 디렉터리에서:

```bash
# 1. v1 URDF와 충돌 메시 (이미 있으면 건너뛰고 --urdf로 지정)
git clone https://github.com/orcahand/orcahand_description.git

# 2. 별도 환경 (teleop 환경은 건드리지 않습니다)
uv venv safety_venv --python 3.11 --seed          # 또는: python3.11 -m venv safety_venv
safety_venv/bin/pip install "mujoco==3.12.0" \
    -r Orcahand-Safe-Assembly/Orca_rl/requirements.txt \
    -e "./Orcahand-Safe-Assembly/Orca_safety[geometry,safety]"

# 3. 확인
cd Orcahand-Safe-Assembly/Orca_rl
../../safety_venv/bin/python -m orca_rl.evaluate_safety --model models/run23/final_model.zip --episodes 4 --workers 4
```

- `uv pip`가 아니라 환경에 들어 있는 `pip`로 설치하세요. `requirements.txt`가 orca_sim을 git에서 편집 가능 모드로
  설치하는데, `uv pip`는 이를 거부합니다.
- **우리 수치는 `mujoco==3.12.0`으로 측정했습니다** (`run22_ft_*` 두 행은 예외로, 다른 컴퓨터에서 3.13.0으로 평가).
  3.12.0에서는 위 확인 명령이 이 4개 에피소드에 대해 정확히 `57%`(off)와 `11%`(on, fallback 25.94%)를 출력합니다.
  다른 MuJoCo 버전에서도 실행되지만 접촉 동역학이 조금 달라 개별 에피소드 결과가 달라집니다: 3.15.0에서는 같은
  4개 에피소드가 38%와 20%였습니다. 100 에피소드 수치가 버전에 따라 얼마나 달라지는지는 측정하지 않았으므로,
  off/on 비교는 한 버전 안에서 하세요.
- 이 절차는 macOS(Apple 실리콘), Python 3.11의 빈 디렉터리에서 처음부터 실행해 확인했습니다.

URDF는 `Orcahand-Safe-Assembly/orcahand_description/`와 `Orcahand-Safe-Assembly/` 옆에서 찾습니다. 다른 위치라면
`--urdf /path/to/orcahand_description/v1/models/urdf/orcahand_right.urdf`를 주세요.

## 명령어

`Orcahand-Safe-Assembly/Orca_rl/`에서 해당 환경의 Python으로 실행합니다 (`PY=../../safety_venv/bin/python`).

```bash
# off / on 비교: 같은 정책, 같은 리셋 seed 100개, 45° 목표. M4 Mac 8 프로세스: off 약 1.5분, on 약 8분.
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --goal-angle 45 --episodes 100 \
    --out my_results/run23.json

# 직접 정한 안전 파라미터: Orca_safety의 v1_right.yaml을 복사해 수정한 뒤 지정
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --modes on \
    --safety-config /path/to/my_v1_right.yaml --out my_results/run23_mycfg.json

# 제어 스텝마다 JSON 한 줄 (q_current, q_nominal, q_safe, 간격, 활성 링크 쌍, fallback 원인, 단계별 시간)
# -> my_logs/run23.off.jsonl, my_logs/run23.on.jsonl
$PY -m orca_rl.evaluate_safety --model models/run23/final_model.zip --episodes 8 --log my_logs/run23

# 나란히 비교 영상: 왼쪽 off, 오른쪽 on. 아래 막대는 손가락 사이 최소 간격
$PY -m orca_rl.record_safety --model models/run23/final_model.zip --seeds 0 1 --out off_on.mp4
```

자주 쓰는 옵션: `--modes off` / `--modes on`(한쪽만 실행), `--workers N`, `--seed N`(첫 리셋 seed),
`--goal-angle`(클수록 어려움; 우리가 보고하는 값은 30 / 45 / 60).

## 모델 (`models/`)

| 폴더 | 설명 | 45° 목표 성공률, 안전 레이어 off / on |
|---|---|---|
| `run23` | v1 정책, seed 1, 3,000만 스텝, 안전 레이어 없이 학습 | 43% / 34% |
| `run22` | 위와 같음, seed 0 | 39% / 36% |
| `run23_ft_ctrl` | run23 + 500만 스텝, 여전히 안전 레이어 없이 | 50% / 31% |
| `run23_ft_safety` | run23 + 500만 스텝, 학습 루프에 필터 포함 | 40% / 42% |
| `run22_ft_ctrl` | run22 + 500만 스텝, 없이 | 45% / 35% |
| `run22_ft_safety` | run22 + 500만 스텝, 포함 | 39% / 36% |

각 폴더에는 `final_model.zip`, `vecnormalize.pkl`(관측 통계, 자동으로 불러옴), `env_kwargs.json`(손 버전, 행동 모드와
스케일, 자동으로 읽음)이 있습니다.

## 측정한 내용과 안전 레이어에 관련될 수 있는 점

자세한 내용: [`results/2026-10-12-safety-layer.md`](results/2026-10-12-safety-layer.md),
[`results/2026-10-13-safety-finetune.md`](results/2026-10-13-safety-finetune.md) (중국어).

- 필터가 없으면 정책은 스텝의 57–74%에서 서로 다른 손가락의 링크가 맞닿아 있습니다 (MuJoCo 자체 접촉 검출도 같은 결과: 59% 대 56%). 필터가 있으면 0–0.12%입니다.
- 필터는 스텝의 약 97%에서 목표를 바꾸고, 18–29%에서 fallback합니다. 4 에피소드 확인에서는 fallback 415건이 모두
  `nonlinear_command_margin_violation`이었고 QP 실패는 없었습니다 (요약에 원인이 출력됩니다). `Orca_safety` 벤치마크의
  teleop 녹화에서는 fallback이 0.2–0.7%였고 모두 QP 잔차 검사에서 나왔으므로, 이 정책은 훨씬 가혹한 입력입니다.
- 리셋 자세에서 `right_middle_mp`와 `right_ring_mp` 사이는 6.2 mm로, 안전 벤치마크의 `open` 시나리오와 같은 쌍, 같은
  거리입니다. 따라서 활성화 거리 15 mm에서는 항상 일부 제약이 활성 상태입니다.
- 필터는 큐브를 모릅니다. 큐브가 손가락을 서로 밀어붙이므로, 필터가 켜져 있어도 스텝의 19–30%에서 간격이 5 mm 미만입니다.
- 이 부하에서 필터는 스텝당 평균 65–80 ms(M4 Mac)이고, 그 절반 정도가 유한 차분 기울기입니다.
  `orca_rl/safety_fast.py`는 최근접점에서 같은 기울기를 직접 계산합니다 (23 ms → 3 ms, 활성 제약 1857행에서 최대 차이
  7e-7 m/rad). 이것은 `train.py --safety`에서만 쓰이며, 위 평가는 `Orca_safety/src/`의 필터를 그대로 실행합니다.

## 필터를 학습 루프에 넣고 학습하기 (선택, 느림)

```bash
$PY -m orca_rl.train --name my_ft_safety --version v1 --obs-rotvec --obs-fingertips --seed 1 \
    --n-envs 10 --n-steps 409 --timesteps 5000000 --curriculum-start-deg 45 \
    --resume models/run23/final_model.zip --safety
```

M4 Mac에서 초당 약 175 스텝 (500만 스텝에 약 8시간). 결과는 `runs/my_ft_safety/`에 저장됩니다 (git에 포함되지 않음).
