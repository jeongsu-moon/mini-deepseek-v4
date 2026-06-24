# mini-deepseek-v4 — Step 1 scaffold

베이스라인 + 측정 인프라. V4 컴포넌트(CSA/HCA/MoE/mHC/Muon/MTP)는 `components/`에
스텁으로 배선돼 있고, 해당 step에서 `NotImplementedError`를 채우면 됩니다.
전체 계획·검증은 `ROADMAP.md`, 보고서 원문은 [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
(저작권상 PDF는 저장소에 포함하지 않음 — `.gitignore` 참조).

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
