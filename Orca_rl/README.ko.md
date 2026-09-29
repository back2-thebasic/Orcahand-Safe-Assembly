# Orca_rl: ORCA 손의 손 안 큐브 회전 (PPO)

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

ORCA 손은 손바닥이 위를 향하고, 큐브는 손바닥 위에 놓여 있습니다. 정책은 큐브의 빨간 면을 지정된 방향으로 돌린 뒤, 15° 이내에서 연속 10 스텝 동안 유지해야 합니다.
성공할 때마다 새 목표가 주어지고 에피소드는 계속됩니다. 학습은 Stable-Baselines3의 PPO를, 시뮬레이션은 수정하지 않은 업스트림 [orca_sim](https://github.com/orcahand/orca_sim)을 사용합니다.

## 현재 결과

목표 성공률(해결한 목표 수 / 목표 시도 수). 난이도 고정, 난이도마다 100 에피소드, 학습 seed 2개.

| 손 | 설정 | 30° | 45° | 60° | 학습 중 커리큘럼이 도달한 난이도 |
|---|---|---|---|---|---|
| v2 | 베이스라인 (run10 / run12) | 49% / 53% | 39% / 37% | 28% / 25% | 42° / 43° |
| v2 | + 목표 회전 벡터 + 손끝 위치·접촉 (run20 / run21) | 74% / 77% | 65% / 62% | 54% / 52% | 82° / 79° |
| **v1** | 베이스라인 (run24, seed 0만) | 29% | 22% | 15% | 27° |
| **v1** | + 목표 회전 벡터 + 손끝 위치·접촉 (run22 / run23) | 51% / 55% | 39% / 43% | 29% / 31% | 47° / 50° |

- v2에서는 목표 회전 벡터("어느 축으로, 얼마나 더 돌려야 하는지")와 손끝 관측을 추가하자 모든 난이도에서 성공률이 약 25–30%p 올랐습니다.
- **실제 손은 v1입니다.** 같은 설정이 v1에서는 확연히 낮습니다. 가장 가능성 높은 원인은 v1 시뮬레이션 장면에서 큐브가 손가락에서 너무 멀리 놓여 있다는 점입니다(`results/2026-09-27-run22-24-v1.md` 참고). 장면을 실제 배치에 맞춰야 합니다.
- 실험별 전체 기록은 `results/`에, 발견하고 고친 문제의 시간순 기록은 `orca_rl/README.md`에 있습니다. 둘 다 중국어로 작성되어 있습니다.

- 팀원이 생성한 태스크(집기, 손동작, 집어서 놓기)에 대한 PPO 학습: `taskgen_rl/`와 `results/2026-09-29-taskgen.md` 참고.

## 디렉터리 구조

```
Orca_rl/
├── orca_rl/
│   ├── task.py           # 환경: 보상, 성공 판정, 커리큘럼, 관측, 행동 매핑
│   ├── train.py          # PPO 학습
│   ├── evaluate.py       # 고정 난이도 평가 (학습 곡선 대신 이것으로 비교)
│   ├── diagnose.py       # 실패를 "목표에 도달 못 함"과 "유지 못 함"으로 나눔
│   ├── checks.py         # 보상에 악용할 허점이 없는지 확인; 보상을 바꾼 뒤 먼저 실행
│   ├── record.py         # 정책 하나의 영상
│   ├── compare_video.py  # 여러 정책을 같은 seed로 나란히 비교
│   └── README.md         # run1부터 지금까지의 사후 분석
├── results/              # 실험별 기록
└── requirements.txt
```

## 설치

Python 3.11. `Orca_rl/` 디렉터리에서:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m orca_rl.checks          # "all checks passed"가 출력되어야 합니다
```

## 자주 쓰는 명령

모든 명령은 `Orca_rl/` 디렉터리에서 실행합니다. 결과는 `runs/<name>/`에 저장됩니다(git에는 포함되지 않음).

```bash
# 학습: v1 손, 현재 최선의 설정
python -m orca_rl.train --name v1_s0 --version v1 --seed 0 --obs-rotvec --obs-fingertips \
    --timesteps 30_000_000 --n-envs 14 --n-steps 292

# 평가: 고정 난이도, 100 에피소드
python -m orca_rl.evaluate --model runs/v1_s0/final_model.zip --goal-angle 45 --episodes 100

# 진단: 목표에 도달하지 못하는지, 도달한 뒤 유지하지 못하는지
python -m orca_rl.diagnose --model runs/v1_s0/final_model.zip --goal-angle 45 --episodes 60

# 영상
python -m orca_rl.record --model runs/v1_s0/final_model.zip --goal-angle 45 --out v1_s0.mp4
python -m orca_rl.compare_video --goal-angle 60 --out videos/compare.mp4 \
    --run runs/a "config A" --run runs/b "config B"
```

- M4 Mac에서 초당 약 6,000 스텝(3,000만 스텝에 약 1.5시간), WSL 위의 i9-11900F에서 초당 약 2,500 스텝(약 3.5시간). 네트워크가 작아서 GPU가 오히려 느리므로 `--device cpu`를 사용하세요.
- 학습에 쓴 환경 설정은 `runs/<name>/env_kwargs.json`에 기록되고, 평가·진단·영상 스크립트가 자동으로 읽습니다. 그래서 손 버전이나 행동 스케일이 섞일 일이 없습니다.
- 학습을 Ctrl-C나 `kill`로 멈추면 모델을 먼저 저장합니다. 체크포인트에서 이어서 학습하려면 `--resume runs/<name>/checkpoints/ppo_<N>_steps.zip`을 쓰고, 이때 `--timesteps`는 추가로 학습할 스텝 수입니다.

## 행동과 제어 규약

- 제어 주기 **10 ms**: MuJoCo 스텝 2 ms, 행동 하나를 5 스텝 동안 적용 (orca_sim 기본값).
- 행동은 17차원, 범위 [−1, 1], **relative** 모드: 매 스텝 이전 서보 목표에 `0.15 × a × 반범위`(반범위 = 해당 관절 가동 범위의 절반)를 더한 뒤 관절 범위로 자릅니다.
  v1에서 서보 목표는 10 ms마다 최대 4.7–9.9° 움직입니다. action_scale 0.06과 0.3도 시험했지만 둘 다 0.15보다 나빴습니다.
- **v1과 v2는 액추에이터 순서가 다릅니다** (v1: 손목, 엄지 ×4, 검지, 중지, 약지, 새끼; v2: 손목, 새끼, 약지, 중지, 검지, 엄지). 한쪽에서 학습한 정책은 다른 쪽에서 쓸 수 없습니다.

## 실제 손에 올리기 전에 남은 과제

- 관측에 시뮬레이터에서만 얻을 수 있는 정보가 들어 있습니다: 큐브의 자세와 각속도, 빨간 면의 방향, 손가락별 접촉 여부. 실제 하드웨어에서는 카메라나 모션 캡처, 그리고 촉각 센서나 추정이 필요합니다.
- 도메인 랜덤화는 아직 하지 않았습니다 (`--randomize-physics`는 구현되어 있지만 학습에 쓴 적은 없습니다).
- v1 시뮬레이션 장면(큐브 크기, 질량, 놓는 위치, 손 자세)을 실제 배치와 맞춰야 합니다.
