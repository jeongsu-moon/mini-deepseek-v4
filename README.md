# mini-deepseek-v4

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-bf16%20%7C%20SDPA-ee4c2c.svg)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.19348-b31b1b.svg)](https://arxiv.org/abs/2606.19348)
[![Roadmap](https://img.shields.io/badge/roadmap-Step%201%20%2F%208-success.svg)](ROADMAP.md)

> **단일 RTX 3090에서, from-scratch PyTorch로, DeepSeek-V4의 핵심 컴포넌트를
> "한 번에 하나씩" 토이 스케일로 재현하는 측정 중심(measurement-first) 연구 프로젝트.**

DeepSeek-V4 보고서([arXiv:2606.19348](https://arxiv.org/abs/2606.19348))의
아키텍처 혁신(CSA/HCA 압축-KV 어텐션, mHC, DeepSeekMoE 델타, Muon, MTP, FP4 QAT)을
**이기는 것이 목표가 아니라**, 각 메커니즘이 작동하는 레짐을 직접 만들어 그 정성적 거동을
재현하고 *왜* 그렇게 동작하는지 설명하는 데 목표를 둡니다. 성공의 정의는
"swap이 baseline보다 loss가 낮다"가 아니라 **"메커니즘 고유 관측량(expert load CV,
residual spectral norm, Sinkhorn 수렴, KV/FLOPs 곡선 …)의 재현 + WHY"** 입니다.

설계의 두 축:
- **baseline-always-runs** — 표준 dense 트랜스포머(RMSNorm·RoPE·SwiGLU·causal SDPA)가
  항상 돌고, config 플래그 하나만 바꿔 V4 컴포넌트로 *디스패치*합니다. 변수 1개만 바뀐
  깨끗한 비교가 가능해집니다.
- **2σ 게이트** — "1 run = 1 data point" 가정을 폐기하고, 모든 효과 주장은 노이즈
  플로어(σ_seed)의 2σ를 넘어야 "real"로 인정합니다.

현재는 **Step 1(측정 인프라 + 베이스라인)** 단계입니다. V4 컴포넌트는 `components/`에
스텁으로 배선돼 있고 호출 시 친절한 `NotImplementedError`로 "무엇을·어떻게 만들지"를
안내합니다. 전체 8단계 계획·사실 교정·방법론은 **[`ROADMAP.md`](ROADMAP.md)** 참조
(저작권상 보고서 PDF는 저장소에 포함하지 않음 — `.gitignore`).

## 설치
```bash
pip install -r requirements.txt   # torch / numpy / matplotlib (transformers는 Step 3+에서)
```

## Step 1 — 세 가지 산출물

**(A) 노이즈 플로어** — "1 run = 1 data point" 가정 폐기. init-seed/data-seed 두 축으로
σ_seed와 2σ 임계값을 뽑는다(이후 모든 효과는 이걸 넘어야 "real").
```bash
python noise_floor.py --config small --n_seeds 5          # 빠른 점검(분 단위)
python noise_floor.py --config gpu3090 --n_seeds 5        # 실제 ~85M 플로어(하루 예산)
# -> out/<cfg>/noise_floor.json
```

**(B) 패리티 하니스** — 베이스라인 원시 연산(RMSNorm·RoPE·SwiGLU·causal SDPA)을
독립 레퍼런스/불변량으로 검증(오늘 PASS). V4 케이스는 PEND(=스텁) 상태로 배선만.
```bash
python parity.py        # baseline PASS / V4 PEND; transformers 있으면 deepseek_v4 탐지
```
Steps 3-7에서 각 컴포넌트를 **핀 고정 commit**의 transformers `deepseek_v4` 서브모듈과
max-abs < 1e-4로 대조(mismatch는 라이브러리 버그일 수도 — ROADMAP §0.1).

**(C) 해석적 시스템 프로파일** — 가중치 로드 없이 config 수치만으로 KV/FLOPs vs 컨텍스트
곡선. 3090은 1.6T/284B를 못 올리고 1M도 못 돌리므로 **해석 계산**(report의 27%/10% shape 재현).
```bash
python profile_analytic.py    # -> out/analytic_profile.{csv,png}, @1M 비율 출력
```

## 베이스라인 학습
```bash
python train.py --config small                     # 즉시 학습, 손실 하강 확인
python train.py --config gpu3090 --data_path corpus.txt   # ~85M
python plot.py out/small/log.json                  # 곡선
python train.py --config gpu3090 --no_compile      # 모듈 수정 중
```

## swap 디스패치 (이후 steps)
config 플래그가 `components/`로 분기 → 미구현은 친절한 에러:
```bash
python train.py --config small   # 기본 = full / mlp / standard / adamw (항상 실행)
# attn_type=csa|hca, ffn_type=moe, residual_type=mhc, optimizer=muon 은 해당 step에서 구현
```

## 파일
```
config.py            프리셋(small/gpu3090) + swap 플래그 + 검증된 컴포넌트 하이퍼
model.py             베이스라인 GPT (RMSNorm·RoPE·SwiGLU·SDPA) + 디스패치
data.py              char-level 데이터셋 (data_seed로 샘플 순서 제어)
train.py             학습 루프(bf16·cosine·deterministic) + train_once()
noise_floor.py       Step 1A
parity.py            Step 1B
profile_analytic.py  Step 1C (가중치 없음)
plot.py              손실곡선 오버레이
components/           CSA/HCA·MoE·mHC·Muon·MTP 스텁(NotImplementedError)
```

## 3090 메모
- Ampere sm_86: **bf16 텐서코어 O**, FP8(Hopper)·FP4(Blackwell) **X** → FP4/FP8은 Step 7에서
  fake-quant(시뮬, 이득 0).
- `gpu3090` ~85M는 weights+AdamW ≈ 1.4GB로 24GB에 4배+ 여유. OOM이면 batch→block 순으로 낮춤.
- 컨텍스트 실험은 4k–16k까지(24GB), 1M은 (C)의 해석 곡선으로만.

## 로드맵 진행 ([`ROADMAP.md`](ROADMAP.md))
| Step | 내용 | 상태 |
|---|---|:--:|
| 1 | 측정 인프라 · 노이즈 플로어 · 시스템 프로파일 | ✅ |
| 2 | 토크나이저 char-level → BPE (하드 게이트) | ⬜ |
| 3 | CSA/HCA 압축-KV 어텐션 *(long pole)* | ⬜ |
| 4 | mHC (Manifold-Constrained Hyper-Connections) | ⬜ |
| 5 | DeepSeekMoE (V3→V4 델타만) | ⬜ |
| 6 | Muon 옵티마이저 + MTP | ⬜ |
| 7 | FP4 QAT (Ampere = simulation-only) | ⬜ |
| 8 | 폐막 재프로파일 + 귀속 원장 | ⬜ |

## 라이선스
MIT (`LICENSE` 참조). 참조하는 DeepSeek-V4 보고서·모델의 라이선스(arXiv:2606.19348)는 이와 별개.
