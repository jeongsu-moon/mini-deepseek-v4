# mini-deepseek-v4 학습 로드맵 (개정판)

> 단일 RTX 3090(Ampere sm_86, 24GB) · from-scratch PyTorch · 토이 스케일(~85M~수백M) 재현 학습용
> 기준 모델: DeepSeek-V4 (2026-04-24 **preview** 공개, arXiv:2606.19348, MIT, open-weights)
> 성공의 정의: "swap이 baseline을 이긴다"가 아니라 **"메커니즘이 작동하는 레짐을 직접 만들어, 그 정성적 거동을 재현하고 WHY를 설명한다"**

---

## 0. 사실 교정 (시작 전 필독)

원래 계획서의 여러 전제가 1차 출처(arXiv 보고서 / HuggingFace transformers .py 소스 / HF 모델 카드)와 대조했을 때 **반증되거나 오귀속**되었다. 아래를 교정하지 않고 시작하면 "존재하지 않는 것" 또는 "V4가 발명하지 않은 것"을 V4 혁신으로 착각하며 공부하게 된다.

태그 규칙: `[보고서 확인]` = 1차 출처(arXiv:2606.19348 / 2512.24880 / 2512.02556 / 2412.19437 / 2408.15664, 또는 transformers `src/.../deepseek_v4/*.py`)에서 직접 확인. `[추정/미확인]` = 2차 블로그·커뮤니티 요약·자체 추론(반드시 1차 확인 후 인용).

> 웹 가용성 메모: 본 교정에 사용한 5개 클러스터(존재/스펙, 하이브리드 어텐션, mHC, 학습 스택, transformers 레퍼런스)는 **전부 web_available=true**로 1차 대조가 가능했다. 다만 fact-checker가 명시한 두 가지 주의: (1) 일부 검색 스니펫이 검증 대상 주장과 비정상적으로 정확히 일치(환각 위험)했으므로, 아키텍처 세부 수치는 SEO 블로그가 아니라 보고서 PDF/transformers 소스에서 재확인할 것. (2) fact-checker가 **DeepSeek_V4.pdf 자체를 직접 열지는 못했다** — 즉 보고서 내부 수치(61층/384전문가/27%/10%/k값 등)는 모델 카드·2차 출처에서 유도된 것이므로 인용 전 PDF 도표에서 직접 재확인할 것.

> **✅ 1차 PDF 검증 완료 (2026-06-24)**: `DeepSeek_V4.pdf`(arXiv:2606.19348, **58페이지**, DeepSeek-AI)를 직접 열어 아래 `[추정/미확인]` 항목을 확정했다. 위 메모의 "PDF 미개봉" 한계는 **해소**됨. 보고서는 **Pro/Flash 두 config를 따로** 명시(§4.2.1)하므로, 모든 수치엔 **변형(Pro/Flash)**을 반드시 붙일 것.
>
> | 설정 | V4-Pro `[보고서 확인]` | V4-Flash `[보고서 확인]` |
> |---|---|---|
> | layers / hidden d | **61 / 7168** | **43 / 4096** |
> | 부트스트랩 첫 2층 | **HCA** | **pure sliding-window** |
> | CSA attention top-k | **1024** | **512** |
> | CSA m / HCA m′ | 4 / 128 | 4 / 128 |
> | query heads n_h / head dim c | 128 / 512 | 64 / 512 |
> | query-compress d_c / out groups g / d_g | 1536 / 16 / 1024 | 1024 / 8 / 1024 |
> | indexer heads n_Iʰ / dim c_I | 64 / 128 | 64 / 128 |
> | sliding window n_win | 128 | 128 |
> | routed / shared experts, active | **384 / 1, 6** | **256 / 1, 6** |
> | expert intermediate dim | 3072 | 2048 |
> | hash-routing 부트스트랩 | 첫 **3** MoE층 | 첫 **3** MoE층 |
> | MTP depth | 1 | 1 |
> | n_hc / Sinkhorn iters | 4 / 20 | 4 / 20 |
> | 학습 토큰 | **33T** | **32T** |
> | 1M-context vs V3.2 | **27% FLOPs / 10% KV** | **10% FLOPs / 7% KV** |
> | Newton-Schulz | 10 iters = 8×(3.4445,−4.7750,2.0315) → 2×(2,−1.5,0.5) | (동일) |
> | KV 저장 | RoPE 차원 BF16 / 나머지 FP8 (→ pure-BF16 대비 ~절반) | (동일) |
> | FP4 범위 | routed-expert 가중치 + indexer QK 경로, **FP4×FP8 GEMM**(현 HW peak는 FP8×FP8와 동일, 미래 HW 1/3) | (동일) |
>
> 토이 스케일에선 위 값을 그대로 못 쓰지만, **비례 축소의 기준 앵커**이자 transformers 패리티 대조의 정답값이다. ('27% FLOPs'는 보고서 명기상 *equivalent FP8 FLOPs* 기준.)

### 0.1 반증된 주장 (REFUTED — 그대로 공부하면 안 됨)

| 원 계획의 주장 | 판정 | 교정된 사실 | 출처 힌트 |
|---|---|---|---|
| "Muon을 1.6T MoE에 **최초 적용**한 것이 V4" | `[보고서 확인]` 반증 | V4 보고서에 'first' 문구 **없음**. Muon은 Keller Jordan 설계, **대규모 최초**는 Moonshot **Moonlight**(arXiv:2502.16982), **조-단위 MoE 최초**는 **Kimi-K2**(MuonClip, V4 이전). V4 기여는 *통합*(hybrid-ZeRO for Muon, BF16+stochastic-rounding grad sync)일 뿐. → **'최초' 서사 폐기**, Newton-Schulz 직교화 *메커니즘*(구체 계수/스텝 스케줄은 §6 참조)을 공부 | arXiv:2606.19348 / 2502.16982 / moonshotai.github.io/Kimi-K2 |
| "V4 docs가 R-series와 V-series를 **병합**한다고 명시" | `[보고서 확인]` 부분오류 | DeepSeek 1차 자료(HF 카드/블로그/API docs)는 **병합이라 표현하지 않음**. 실제는 단일 모델 + 호출별 추론 깊이 모드(Non-think / Think High / Think Max). 'R+V 병합' 프레임은 2차 SEO 글에만 존재 | huggingface.co/blog/deepseekv4, HF 모델 카드 |
| "deepseek_v4 transformers 구현은 성숙·고정된 ground truth" | `[보고서 확인]` 반증 | 2026-05-02 추가 후 6월까지 활발히 churn(CSA mask collapse fix 05-13, hc_head/sinks/position_bias fp32 고정 05-27, FP8/FP4 MoE 작업). 렌더링된 docs에는 가짜 예제 `mistralai/DeepseekV4-8x7B-v0.1`(Mixtral 템플릿 잔재) 존재. → **commit 핀 고정**, `.py` 소스를 docs보다 신뢰, mismatch는 라이브러리 버그 가설로 먼저 의심 | github.com/huggingface/transformers commits/.../deepseek_v4 |

### 0.2 오귀속 — 존재하지만 V4 발명이 아님 (INHERITED, V2/V3/V3.2 출처로 공부)

| 컴포넌트 | 실제 출처 | V4의 실제 델타(여기만 V4로 공부) | 출처 힌트 |
|---|---|---|---|
| aux-loss-free load balancing | Wang et al. 2024 (arXiv:2408.15664), **V3에서 최초 대규모 배포** | V4는 그 위에 **sequence-wise balance loss**만 추가 | 2408.15664 / 2412.19437 |
| DeepSeekMoE (fine-grained + shared expert) | DeepSeekMoE / V2 | scoring Sigmoid→**Sqrt(Softplus(·))**(`scoring_func='sqrtsoftplus'`), 첫 ~3층 **hash_moe** 부트스트랩, **n_group/topk_group 삭제** | transformers configuration_deepseek_v4.py |
| MTP (multi-token prediction) | V3 (arXiv:2412.19437) | 보고서가 "V3와 **identical**"이라 명시 → V3 문서로 공부 | 2606.19348 / 2412.19437 |
| DSA (DeepSeek Sparse Attention) + Lightning Indexer | **V3.2-Exp**(arXiv:2512.02556) | V4의 CSA가 이걸 **압축된 KV 스트림 위에** 재사용 | 2512.02556 |

### 0.3 용어/수치 교정 (오해 소지)

- **mHC = Manifold-Constrained Hyper-Connections** `[보고서 확인]` (arXiv:2512.24880). 원 계획의 "generic channel mixing"은 **오해**. 실제는 `hc_mult=4`개의 **병렬 residual STREAM**([B,S,4,D])을 (pre, post, comb) triplet으로 혼합하며, comb는 Sinkhorn-Knopp **20회**로 **이중확률(doubly-stochastic, Birkhoff polytope)** 투영. **hidden 채널 혼합이 아니라 스트림 혼합**. ByteDance Seed의 Hyper-Connections(arXiv:2409.19606, ICLR 2025, expansion 4)의 **제약 확장**이며 오귀속이 아닌 정당한 계보.
- **"doubly-stochastic ⇒ spectral norm ≤ 1"** `[보고서 확인, 단 정밀화]`: 일반 이중확률 행렬은 induced 1-norm = ∞-norm = 1 이므로 operator 2-norm ≤ √(‖A‖₁·‖A‖∞) = 1 (비확장적). "행/열 합이 1"만으로는 부족 — 위 논거로 진술할 것. 결론(identity-mapping 복원, HC 불안정 수정)은 타당.
- **CSA 압축은 비중첩 평균이 아님** `[보고서 확인]`: 각 압축 엔트리는 **2m 중첩 윈도우**에서 유도, 순 시퀀스 축소는 1/m. 프로덕션 m=4. HCA는 **비중첩** m'=128.
- **CSA attention top-k** `[보고서 확인, PDF §4.2.1]`: **V4-Pro=1024, V4-Flash=512** (원래 `[추정/미확인]`이었으나 PDF로 확정 — Pro 값이 1024). 토이 스케일에선 여전히 비례 축소·sweep하되, 이젠 "검증 안 됨"이 아니라 "스케일 부적합"이 sweep 사유. (참고: V3.2 DSA의 top-k=2048는 **128K context 기준** raw token 값 `[보고서 확인]` — CSA의 *압축 스트림* top-k와 **다른 객체**. 보고서도 "relative to V3.2, a smaller attention top-k is chosen"이라 명시.)
- **부트스트랩 층** `[보고서 확인]`: "처음 2층이 sliding-window full attention"은 **V4-Flash에만 참**. **V4-Pro는 첫 2층이 HCA**. 변형(variant)을 항상 명시.
- **FP4 범위** `[보고서 확인]`: 깔끔한 "experts FP4 / 나머지 FP8" 분할이 **아님**. routed-expert 가중치(+ indexer QK 경로) FP4, **FP4×FP8 GEMM**(FP4 weight × FP8 activation). "FP4×FP8 ~1/3 효율"은 **미래 하드웨어** 주장.
- **존재/스펙** `[보고서 확인, PDF 직접]`: V4-Pro = 1.6T total / 49B active (**61층**, hidden 7168, **384** routed + 1 shared, 6 active), V4-Flash = 284B / 13B (**43층**, hidden 4096, **256** routed + 1 shared, 6 active), 1M context, MIT. 정밀 config는 §0 상단 표 참조. V4는 **GA가 아니라 'preview'**로 인용.

### 0.4 "top-down" 라벨 오류 교정

원 계획은 스스로를 "top-down"이라 불렀으나 실제로는 **bottom-up 재구성**(CSA 먼저 → HCA → 쌓기)이었다. 진짜 top-down은 **시스템 전체의 헤드라인 거동을 먼저 측정**한 뒤(보고서 주장: 1M context에서 V4-Pro는 V3.2 대비 단일토큰 추론 FLOPs의 ~27%, KV 캐시의 ~10% `[보고서 확인]`), **어느 컴포넌트가 그 예산의 어느 몫을 사는지**로 분해하는 것이다. 본 로드맵은 **Step 1에 시스템 프로파일(분석곡선)**을 두고 **Step 8에 재프로파일+귀속 원장**을 두어 top-down 루프를 닫는다.

---

## 측정 철학 (전체 공통)

원 계획의 좋은 골격은 **유지**한다:
- **NotImplementedError 디스패치**: 미구현 분기는 조용히 통과시키지 말고 `raise NotImplementedError`로 강제 노출.
- **baseline-always-runs**: 모든 실험에서 dense baseline을 항상 함께 돌려 동일 조건 비교.
- **reference cross-check**: transformers `deepseek_v4` `.py` 소스(핀 고정 commit) 대비 서브모듈 수치 패리티(<1e-4).

**교체된 것은 측정 철학이다**:
- ❌ 폐기: "swap이 val loss를 더 낮추면 성공."
- ✅ 채택: **성공 = (a) 컴포넌트가 효과를 내는 레짐을 직접 엔지니어링하고, (b) 그 메커니즘 고유 관측량(전문가 활용/특화, residual spectral norm, loss-vs-LR 분지, KV bytes/FLOPs 곡선, PTQ-vs-QAT 격차)을 재현하며, (c) WHY를 설명한다. val-loss 패리티(±2σ_seed)는 실패가 아니라 PASS.**

전역 규칙:
1. **seed ≥ 3**(노이즈 플로어 Step은 ≥5), mean±std 보고, 어떤 효과든 **2σ_seed** 초과해야 "real". 단일 seed 주장 금지.
2. **HP 통제 명시**: 비교 시 변경하는 변수 1개만, 나머지(LR schedule, init seed, data order, token budget) 고정. 옵티마이저 비교는 **best-LR-vs-best-LR**.
3. 각 컴포넌트는 **additive arm과 constraint arm을 분리**(예: plain-residual vs unconstrained-HC vs mHC). full 컴포넌트를 plain baseline에만 비교 금지.
4. 실행 전 **반증 조건 사전등록**: "불안정 baseline이 안 터지면", "PTQ가 안 무너지면", "context가 짧아 FLOP 이득 없으면" → null은 "메커니즘 부재"가 아니라 "레짐 잘못/검정력 부족"으로 해석.
5. **loss뿐 아니라 내부 probe 로깅**(Newton-Schulz 특이값→1, depth별 residual spectral norm, Sinkhorn 행/열합 수렴, expert load CV, indexer top-k hit-rate, FP4 saturation). 토이 스케일은 loss는 못 재현해도 내부 거동은 충실히 재현 가능.
6. transformers commit **핀 고정**. mismatch는 *내 버그 가설*이자 *라이브러리 버그 가설* 둘 다. 정확한 config 문자열 리터럴만 사용: `'sliding_attention'`/`'compressed_sparse_attention'`/`'heavily_compressed_attention'`, `'hash_moe'`/`'moe'`, `scoring_func='sqrtsoftplus'`. 약어 CSA/HCA로 배선 금지.
7. **하드웨어 스코프 배너**: 3090 = Ampere sm_86, BF16+INT8 tensor core 보유, **FP8(sm_90)·FP4(sm_100) 경로 없음**. 모든 FP4/FP8 작업은 fake-quant(수치 전용, throughput 이득 0). 모든 1M-context FLOP/KV 수치는 **해석적 계산**이지 벤치마크가 아님(이 카드는 1M context를 못 담음).

---

## 1. 측정 인프라 + 노이즈 플로어 + 시스템 프로파일 (top-down 진입)

이 단계는 두 가지를 동시에 깐다: (A) 이후 모든 임계값의 기준이 될 σ_seed, (B) top-down 프레임의 시스템 예산.

| 항목 | 내용 |
|---|---|
| **교체 대상** | 아키텍처 swap 없음. "1 run = 1 data point" 가정 → 경험적 노이즈 플로어로 교체. 추가로, baseline + transformers 패리티 하니스 구축. |
| **regime engineering** | (A) 노이즈 플로어: 85M dense baseline(pre-LN, RMSNorm, RoPE, SwiGLU, AdamW)을 val loss가 명확히 plateau할 token budget(단, 1일 내 5 seed 가능)까지 학습. init seed 변경 sub-sweep + data-order seed 변경 sub-sweep 분리. (B) 패리티: hidden 256/4층/n_routed_experts 8/hc_mult 4/hc_sinkhorn_iters 20의 tiny transformers DeepseekV4를 fixed-seed로 서브모듈 단위 대조. 결정론 모드(`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). (C) 시스템 프로파일(**분석 전용 — 가중치 로드 없음**): Flash/Pro의 **config.json만** 읽어(§0 표 값) **KV bytes/token·FLOPs/token vs seq len을 공식으로 계산**(4k/32k/128k → 1M 외삽), dense deepseek_v3 config와 대비. ※ 284B/1.6T 실가중치는 24GB에 못 올림 — 실제 forward가 아니라 config 기반 해석 계산이다. (실가중치 forward가 필요한 경험검증은 Step 3에서 토이 config로만.) |
| **반증 가능한 성공 기준** | seed≥5로 val loss σ_seed(nats/char & bits/char)와 각 probe σ를 수치로 명시, 95% CI 반폭과 MDE=2σ_seed/√n 문서화. dense block이 config-matched transformers 서브모듈과 3 seed에서 max-abs <1e-4. KV-vs-length 곡선의 기울기가 dense v3 대비 sub-linear. **사전등록 반증**: σ_seed가 너무 커서(예 >0.05 nats/char) 어떤 컴포넌트 효과도 못 넘으면 → 토이 스케일 underpowered 선언, model/token budget↑ 후 진행. |
| **측정 항목** | val loss mean±std; probe별 std; 서브모듈 max-abs/mean-abs diff; KV bytes·FLOPs vs seq len(V4 config vs v3); run당 wall-clock; MDE. |
| **함정/교란요인** | 단일 seed의 0.01–0.03 nat 차이를 신호로 착각. 모든 seed에 동일 data order 고정 시 분산 과소추정 → 후속 false positive. RoPE base/scaling, RMSNorm eps mismatch가 조용한 패리티 파괴자. **V3.2를 패리티 대상으로 쓰지 말 것**(DeepseekV32Config만 'hack' PR로 머지, DSA 완전구현 아님) — dense baseline은 deepseek_v3. 렌더 docs의 `mistralai/DeepseekV4-8x7B-v0.1` 예제는 가짜. |
| **현실적 소요일** | 5–7일 (노이즈 플로어 1.5일 + 패리티 하니스 4–6일 중첩, 시스템 프로파일 0.5일) |

---

## 2. 토크나이저 전환: char-level → BPE (하드 게이트, 이후 단계의 전제)

char-level은 이후 측정하려는 신호 자체를 파괴한다: (i) MoE 전문가 특화(형태소/주제 구조 없음), (ii) sparse long-range(토큰 거리 ~4–5배 부풀려 "long-range"가 사실상 local), (iii) MTP acceptance(다음 char가 자명해 acceptance 과대평가). **따라서 MoE/어텐션-long-range/MTP 단계 이전에 반드시 전환.**

| 항목 | 내용 |
|---|---|
| **교체 대상** | char-level → 16k–32k BPE(HF `tokenizers` 또는 tiktoken-style), 멀티도메인 코퍼스. |
| **regime engineering** | 특화가 측정되도록 **시각적으로 구분되는 3–4개 도메인** 코퍼스 선택(예: The Stack 코드 + FineWeb-Edu 산문 + arXiv 수식 + 대화). enwik8/char 데이터 금지. vocab 16k–32k로 고정해 임베딩+head가 토이 width에서 24GB에 여유. 전환 후 baseline 재학습. |
| **반증 가능한 성공 기준** | seed≥3. 2-도메인 held-out에서 BPE MoE의 per-domain routing 분포 KL > 임계값인 반면, **동일 MoE를 char-level로 돌리면 도메인별 히스토그램이 거의 동일** — char-vs-BPE 격차를 직접 시연해 전환을 정당화. |
| **측정 항목** | per-domain expert-selection 히스토그램 및 KL; 동일 텍스트 길이의 char vs BPE 토큰 수(long-range 인플레이션 정량화). |
| **함정/교란요인** | 실험마다 토크나이저 재학습 금지(비교 불가). vocab 성장으로 임베딩 테이블이 메모리 초과하지 않게. **한 번 고정 후 모든 하류 단계에서 불변** 재사용. |
| **현실적 소요일** | 2–3일 |

---

## 3. CSA/HCA 압축 KV 어텐션 — 구조적 KV/FLOPs 측정을 앞으로 (분석곡선 + 경험검증)

V4의 핵심 차별점이자 인프라 의존도가 가장 높으므로 **측정 스택이 신뢰 가능해진 직후**에 배치. 측정을 **(a) 해석적 곡선(먼저 계산)**과 **(b) 구현이 그 곡선을 따르는지 경험 검증**으로 분리한다.

빌드 순서는 의존성에 따라 **DSA → CSA → HCA**: CSA는 문자 그대로 "압축 KV 위의 DSA"이므로 DSA가 먼저 동작해야 한다.

- **DSA (V3.2 기반, `[보고서 확인]` 2512.02556)**: Lightning Indexer `I_{t,s}=Σ_j w^I_{t,j}·ReLU(q^I_{t,j}·k^I_s)`(ReLU; 프로덕션 fp8이나 Ampere에선 fp32/bf16) + fine-grained top-k raw-KV 선택(V3.2 **@128K context에서 top-k=2048**). MLA의 MQA 모드 안에서.
- **CSA (`'compressed_sparse_attention'`, paper 2.3.1)**: 학습 projection(W^aKV/W^bKV, W^aZ/W^bZ)으로 **2m 중첩 윈도우** 압축(순 1/m, m=4) → 압축 스트림 위에 DSA top-k(Pro=1024 / Flash=512 `[보고서 확인]`, 토이는 비례 축소·sweep) + **128 sliding-window** local branch + **per-head learnable sink**.
- **HCA (`'heavily_compressed_attention'`, paper 2.3.2)**: **비중첩** m'=128 압축, 압축 엔트리 위 **dense**(indexer/top-k 없음), 동일 128 window + sink.
- 층 교대(CSA/HCA, ~61층 절반씩), 부트스트랩은 **변형별**: Flash=첫 2층 pure sliding, Pro=첫 2층 HCA. (sliding window + learnable sink는 CSA·HCA 양쪽 branch에 모두 적용 `[보고서 확인]`.)

| 항목 | 내용 |
|---|---|
| **교체 대상** | full attention → 교대 CSA/HCA(+변형별 부트스트랩). "압축 후 압축스트림에서 선택"이 V3.2 DSA(raw 토큰 @128K top-k=2048) 대비 진짜 V4 진보. |
| **regime engineering** | **(a) 분석곡선 먼저**: KV-cache bytes와 attention FLOPs를 seq len의 함수로 **종이 위에서 계산**(full O(L²) vs DSA O(Lk) vs CSA 1/m·O(·) vs HCA 1/m' dense). 1M까지 외삽해 보고서의 ~27% FLOPs/~10% KV를 **해석적으로 재현**(3090은 1M 못 돌림 — SIMULATION 명시). **(b) 경험검증**: 토이 width(hidden 256–512)에서 context 4k–16k(24GB 한도), k 비례 축소(예 128–256). BPE(Step 2) 필수. needle-in-haystack/KV recall 프로브를 distance ≫ 128(sliding window)로 심어 top-k 선택만이 needle 경로가 되게. sliding window 1런 비활성으로 기여 분리. |
| **반증 가능한 성공 기준** | seed≥3, HP 통제(m·k·indexer init 고정, branch/window/sink만 토글). (1) **분석곡선**: 구현의 측정 KV/FLOPs vs L가 해석곡선과 monotone 일치(HCA가 최대 KV 절감, CSA는 indexer 선택이 관련 블록에 집중). (2) **품질**: matched context에서 CSA/HCA val loss가 full attention의 ±2σ_seed 이내. (3) **long-range**: D>128 needle에서 CSA+indexer는 recall 높고 pure-128-sliding(top-k 없음)은 실패 → 장거리는 indexer 선택이 운반함을 분리. **사전등록 반증**: 압축이 recall에서 >2σ 손실인데 내 context length에서 FLOP/KV 이득도 없으면 → context가 짧아 압축 오버헤드가 지배하거나 indexer/sink/mask 오구현(transformers 소스 대조). |
| **측정 항목** | (해석) KV bytes·FLOPs vs L 곡선, 1M 외삽치; (경험) 측정 KV/FLOPs가 곡선 추종 여부, needle recall vs distance(CSA/HCA/sliding), indexer top-k hit-rate(심은 블록), sink·window ablation, transformers 대비 패리티 diff. |
| **함정/교란요인** | 짧은 context → 압축 오버헤드 지배(메커니즘 무의미해 보임 — 레짐이 long context). CSA 2m 중첩 인덱싱 off-by-m. **CSA mask는 transformers가 'mask collapse' 버그를 냈던 바로 그 지점** — 신중 대조, mismatch가 라이브러리 버그일 수 있음(commit 핀). top-k은 PDF 확정값(Pro=1024/Flash=512)이나 토이에선 비례 축소·sweep. sink logits fp32 유지. |
| **현실적 소요일** | 9–13일 (이 로드맵의 long pole) |

---

## 4. mHC (Manifold-Constrained Hyper-Connections) — 불안정 baseline을 먼저 만들기

mHC의 목적은 **HC의 불안정성을 고치는 것**이므로, 고칠 불안정성이 없으면 측정 불가다. 따라서 **deep & unstable baseline을 의도적으로 만든다.**

| 항목 | 내용 |
|---|---|
| **교체 대상** | 표준 residual → {unconstrained HC, mHC}. mHC = `hc_mult=4` **병렬 residual 스트림**([B,S,4,D]), (pre,post,comb) triplet, comb는 Sinkhorn-Knopp **20회**로 이중확률 투영(`hc_eps=1e-6`). "채널 혼합"이 아니라 **스트림 혼합**. ByteDance HC(2409.19606)의 제약 확장. Sinkhorn backward는 **20회면 unrolled autograd**가 정답(transformers와 동일) — implicit diff 불필요. |
| **regime engineering** | baseline을 **일부러 불안정**하게: 48층+ 깊이 AND 정규화 약화/제거 또는 과대 residual init 사용 → plain-residual·unconstrained-HC가 발산/loss spike/residual spectral norm 폭발. 이 불안정 레짐에서만 mHC 안정화가 **가시화**. 얕은 12층 안정 baseline에선 mHC가 아무것도 안 보이는 게 정상(실패 아님). 3 arm(plain / HC / mHC) 동시 학습. |
| **반증 가능한 성공 기준** | seed≥3, HP 통제(동일 depth/width/seed/LR, residual scheme만 변경; hc_sinkhorn_iters 0 vs 20 sweep으로 "추가 파라미터가 아니라 제약"이 안정성 원인임을 분리). (i) unconstrained HC/plain이 측정 가능하게 더 불안정(spike↑, max residual spectral norm↑, 발산 seed↑), (ii) mHC가 depth별 residual operator-norm 성장을 ≤1 쪽으로 억제하고 발산 seed 감소, final loss는 ±2σ_seed 패리티 이상. comb가 transformers Sinkhorn 출력과 <1e-4, 행/열합 1±1e-5. **사전등록 반증**: '불안정' baseline이 실제로 안 터지면(spike 없음, norm bounded) → 레짐 엔지니어링 실패, mHC 검정 불가 → 더 깊게/더 나쁜 init로 불안정 유도. |
| **측정 항목** | 발산/spike seed 비율(plain vs HC vs mHC); depth별 residual spectral norm(power iteration); Sinkhorn 행/열합 수렴; grad-norm spike 카운트; final val loss 패리티; transformers `DeepseekV4HyperConnection` 패리티. |
| **함정/교란요인** | 안정한 얕은 망에서 mHC 검정(고칠 게 없어 노이즈 측정 → "mHC 무용" 오결론). unconstrained-HC arm 없이 plain만 비교(추가 스트림 효과 vs 제약 효과 구분 불가). **comb/sinks를 bf16로 돌리면 자체 불안정으로 교란 — fp32 유지**(transformers 05-27 fix 미러). 스트림을 hidden dim으로 reshape 금지. mHC는 always-active. mHC static bias/gating은 Muon이 아니라 AdamW로. |
| **현실적 소요일** | 4–6일 |

---

## 5. MoE — win-condition 폐기, 활용/특화 + aux-loss-free 컨트롤러 측정 (V3→V4 델타만)

DeepSeekMoE를 from-scratch 재유도하지 말 것 — V3 블록(안정적 deepseek_v3 레퍼런스)을 포팅하고 **검증된 V4 델타 4개만** 구현한다.

| 항목 | 내용 |
|---|---|
| **교체 대상** | dense FFN → V4 MoE 스케줄. **V4 델타만**: (a) scoring Sigmoid→**Sqrt(Softplus(·))**(`'sqrtsoftplus'`); (b) 첫 ~3층 `'hash_moe'` = **frozen** `tid2eid[input_ids]` lookup; (c) V3의 **n_group/topk_group 삭제**; (d) inherited aux-loss-free(`noaux_tc`/`e_score_correction_bias`) 위에 **sequence-wise balance loss** 추가. clamped SwiGLU experts, shared expert 1. aux-loss-free·DeepSeekMoE·MTP는 **V4 발명 아님**(§0.2). |
| **regime engineering** | 멀티도메인 BPE 코퍼스(Step 2)로 routing에 실구조 부여. 토이 config(예: n_routed_experts 8–16, top-k 2, shared 1; 프로덕션 256/6/1을 비례 축소). 약간 skew된 expert init 또는 도메인-불균형 배치로 **load imbalance를 가시화**한 뒤 aux-loss-free bias가 학습 중 교정하는 걸 보임. bias가 steady state 도달하도록 충분한 토큰. 두 레짐: (A) loss-win을 노리려면 active param 고정·전문가↑·token budget↑; (B) **주(主) 레짐 = routing 내부 관측**. |
| **반증 가능한 성공 기준** | seed≥3, HP 통제(전문가수·top-k 고정, 델타 1개씩만 토글; 동일 affinity logit에서 Sigmoid vs sqrtsoftplus A/B). (i) aux-loss-free가 per-expert load를 거의 균일(load CV·max/min 비 수렴)로 몰되 loss-저하 aux term 없이; bias를 0으로 freeze(control)하면 routing collapse(일부 전문가 ~0 토큰). (ii) 전문가 특화가 chance 초과(token-class↔expert MI / routing entropy가 uniform-random baseline보다 >2σ 낮음); hash_moe 부트스트랩 on/off로 early routing collapse 감소. **사전등록 반증**: routing entropy가 random과 구분 불가 AND load 균형이 "아무것도 특화 안 해서"라면 → 이 스케일에서 degenerate MoE임을 정직하게 보고(이득 주장 금지). |
| **측정 항목** | per-expert load CV·max/min over training; routing entropy; input char/domain↔expert MI; dead expert 수; sequence-wise loss on/off 히스토그램; hash_moe on/off 안정성; fixed active-FLOP에서 loss(패리티, win 아님); Sigmoid vs sqrtsoftplus A/B; transformers MoE 패리티 <1e-4. |
| **함정/교란요인** | "MoE beats dense" 추구(이 스케일에서 구조적으로 도달 불가, budget 낭비). aux-loss-free/DeepSeekMoE/MTP를 V4로 귀속. bias는 **gradient 아닌 load-기반 규칙**으로 갱신 — 옵티마이저에 넣지 말 것. 토큰 부족으로 bias controller 미수렴인데 imbalance로 오판. hash router는 **frozen(no grad)**. **dispatch/indexing 버그가 이 단계 디버깅의 최대 덩어리.** |
| **torch.compile** | top-k 선택 + token scatter/gather는 graph-break. per-expert FFN과 dense/attention 블록만 submodule 단위 compile, **routing/dispatch는 EAGER**. `TORCH_LOGS=graph_breaks python train.py`로 break 열거. fullgraph=True를 routing에 강요 금지. |
| **현실적 소요일** | 6–9일 |

---

## 6. Muon 옵티마이저 — best-LR-vs-best-LR 메커니즘 연구 (+ MTP)

| 항목 | 내용 |
|---|---|
| **교체 대상** | AdamW → Muon(2D) + AdamW(embeddings/heads/RMSNorm/mHC static bias·gating). **반증된 '1.6T MoE 최초 적용' 프레임 폐기**(§0.1). 연구 대상은 직교화 업데이트 *메커니즘*과 LR-transfer 거동. Newton-Schulz: 8스텝 (3.4445,−4.7750,2.0315) → 2스텝 (2,−1.5,0.5)로 특이값을 1에 고정 `[보고서 확인]`. + MTP 헤드(V3와 identical, V4 발명 아님)로 self-speculative decoding 시연. |
| **regime engineering** | **각 옵티마이저별 독립 LR 그리드**(최소 5점 log-spaced)로 각자 최적점에서. Muon 이득은 **더 평탄한 loss-vs-LR 분지** 및/또는 **초반 빠른 수렴**으로 나타나므로 loss-vs-tokens 곡선도 로깅. 선택적으로 Muon이 가장 유리한 레짐(살짝 under-tuned/wider 모델, AdamW conditioning 나쁨) 조성. MTP acceptance는 **BPE에서만** 의미 — char-level로 ablate하면 acceptance가 chance로 하락(Step 2 정당화). |
| **반증 가능한 성공 기준** | seed≥3, **best-LR each**, HP 통제(동일 data order·init seed, precision-scope/optimizer만 변경). final val loss가 AdamW ±2σ_seed 이내(패리티=PASS) AND Muon 정성 속성 ≥1 재현: (i) 더 넓은 LR 분지 OR (ii) 초반 ~20% 토큰에서 빠른 loss 감소. MTP: BPE held-out에서 speculative acceptance >0이고 **동일 3090에서** autoregressive 대비 wall-clock 가속; char-level ablation은 acceptance 부풀려 토크나이저 의존성 확인. **사전등록 반증**: Muon이 두 속성 다 없고 자기 best-LR에서도 2σ 넘게 나쁘면 → Newton-Schulz 구현 점검(업데이트 직교성, 특이값→1). |
| **측정 항목** | 옵티마이저별 loss-vs-LR 곡선(min+0.02 nats에서 분지 폭); loss-vs-tokens 초기 기울기; 직교화 업데이트 특이값 스펙트럼(1 근방 집중); grad/update RMS 비; Muon vs AdamW target까지 steps; MTP acceptance & tokens/sec(BPE vs char). |
| **함정/교란요인** | **핵심 교란: AdamW@best-LR vs Muon@AdamW-LR(또는 공유 LR 그리드) 비교** — Muon은 effective LR 스케일이 달라 어느 쪽이든 결과 조작. 1D/embedding params를 Newton-Schulz로 보내면 비교 오염. coefficient 순서 검증. MTP loss weight는 작게(불안정). **Muon 'first' 주장 금지.** |
| **현실적 소요일** | 4–6일 |

---

## 7. FP4 QAT — SIMULATION-ONLY (Ampere 스코프 제외), QAT-vs-PTQ 격차로 측정

> **하드웨어 사유 (필독)**: 3090 = Ampere **sm_86**에는 **FP8(sm_90/Hopper)·FP4(sm_100/Blackwell) tensor core가 없다.** 따라서 본 단계의 모든 "FP4 학습"은 **fake-quant**(bf16에서 quant→dequant)이며, **실제 메모리·연산 이득은 0**이다. 이는 QAT의 *수치/오차*(scale 선택, STE gradient, bf16 대비 정확도 델타)를 배우게 하지 throughput을 가르치지 않는다. "FP4로 학습한다"고 제시 금지. 보고서의 "FP4×FP8 ~1/3 효율"은 **미래 하드웨어 인용**일 뿐 재현 불가. 3090에서 실제 저정밀 이득을 원하면 정직한 선택지는 native **bf16** 또는 **INT8**(sm_86 INT8 tensor core 보유).

다만 FP4는 토이 스케일에서도 PTQ가 가시적으로 무너질 만큼 공격적이라, **QAT의 정확도 회복이라는 알고리즘적 효과는 이 카드에서도 측정 가능**하다 — 그래서 capstone으로 적합.

| 항목 | 내용 |
|---|---|
| **교체 대상** | full-precision experts → {PTQ-FP4 experts, QAT-FP4 experts}, fake-quant. 정밀 범위: FP4 = **routed-expert 가중치(+ indexer QK 경로)**, FP4×FP8 GEMM operand — 깔끔한 "experts FP4 / 나머지 FP8" 분할 아님(§0.3). |
| **regime engineering** | experts를 dominant param mass로(충분한 routed expert) 만들어 양자화가 의미 있게. fake-quant scale/calibration을 sweep해 **PTQ에 최선의 기회**(strawman 금지). bf16 baseline vs fake-FP4-QAT vs naive PTQ-to-FP4(동일 최종 가중치, QAT 없음) 비교. |
| **반증 가능한 성공 기준** | seed≥3, HP 통제(동일 data order·init seed, precision-scope만 변경). PTQ-FP4가 bf16 대비 val loss/ppl을 명확히(>2σ_seed) 저하시키고, QAT-FP4가 그 격차의 문서화된 비율을 회복(QAT가 PTQ보다 bf16에 >2σ 더 가까움). **명시 보고**: 3090에서 wall-clock/메모리 변화 0(fake-quant가 bf16로 돌므로) + 사유 한 단락(sm_86 FP4/FP8 부재). **사전등록 반증**: PTQ-FP4가 안 무너지면(격차 σ 이내) → FP4가 이 모델을 stress 안 함, QAT 연구 무의미 → expert param share↑ 또는 더 어려운 quant config로 PTQ를 무너뜨릴 때까지. |
| **측정 항목** | val ppl: bf16 vs FP4-QAT vs FP4-PTQ(두 격차); per-expert weight clipping/saturation; QAT 학습곡선 안정성(STE 분산); 회복된 격차 비율; (명시적으로) 3090에서 불변 wall-clock/메모리. |
| **함정/교란요인** | fake-quant를 실제 양자화 이득으로 착각(simulation 명시). 약한 PTQ baseline(나쁜 calibration)이 QAT 이득 부풀림 — PTQ 먼저 튜닝. "experts FP4 / 나머지 FP8" 과단순화로 V4가 고정밀 유지하는 텐서까지 양자화. 3090에서 FP4×FP8 hardware 가속 주장 금지(품질 회복만 측정, wall-clock 가속 절대 아님 — 그렇게 명시). |
| **현실적 소요일** | 4–6일 |

---

## 8. 폐막 top-down 재프로파일 + 귀속 원장 (top-down 루프 닫기)

| 항목 | 내용 |
|---|---|
| **교체 대상** | 추가 swap 없음. Step 1에서 세운 분해 가설표를 채운다: ~27% FLOP / ~10% KV 예산을 각 컴포넌트(CSA, HCA, schedule, mHC, MoE, FP4)가 내 빌드에서 실제로 얼마나 전달했는지. |
| **regime engineering** | Step 1과 동일 long-context 레짐(분석곡선 비교 가능하게). 내 구현 곡선을 핀 고정 commit transformers 레퍼런스와 대조. |
| **반증 가능한 성공 기준** | (capstone 질문) 내 per-component 절감 합이 Step 1 시스템 측정과 화해(reconcile)되는가, 안 되면 어느 컴포넌트/정밀도 효과를 오모델링했는지 명명 가능한가. 측정 시스템 절감과 합산 컴포넌트 절감의 잔차가 hand-wave 없이 설명됨. 핀 고정 commit 대비 per-layer logit MSE. |
| **측정 항목** | 최종 귀속 원장(component → 측정된 FLOP/KV 절감 몫) vs Step 1 가설; 핀 고정 transformers 레퍼런스 대비 per-layer logit MSE. |
| **함정/교란요인** | 일부 mismatch는 transformers-side(6월까지 active fix) — commit 핀, 모든 diff를 내 버그로 가정 금지. 하이퍼(top-k, m, n_hc 등)는 PDF로 확정됐으니 reconciliation 잔차는 **구현/스케일 차이**이지 "미확인 하이퍼" 핑계로 돌리지 말 것. |
| **현실적 소요일** | 1일 |

---

## 9. 정직한 개정 타임라인

원 계획의 "4–6주 solo-while-learning"은 **약 1.5–2배 낙관적**이다. 디버깅(특히 MoE dispatch, CSA mask 토끼굴), 학습-병행, seed≥3 반복을 반영한 현실치:

| Step | 작업 | 소요(작업일) |
|---|---|---|
| 1 | 측정 인프라 + 노이즈 플로어 + 시스템 프로파일 | 5–7 |
| 2 | BPE 토크나이저 전환(하드 게이트) | 2–3 |
| 3 | CSA/HCA(분석곡선 + 경험검증) — **long pole** | 9–13 |
| 4 | mHC(불안정 baseline 엔지니어링) | 4–6 |
| 5 | MoE 델타(활용/특화 + aux-loss-free) | 6–9 |
| 6 | Muon(best-vs-best) + MTP | 4–6 |
| 7 | FP4 QAT(SIMULATION-ONLY) | 4–6 |
| 8 | 폐막 재프로파일 + 귀속 원장 | 1 |
| **합계** | | **35–51 작업일** |

- **달력 기준: ~8–10주 part-time**(주 25–30 작업시간 가정). 변동성 큰 long pole은 **Step 3(어텐션), Step 5(MoE), Step 7(FP4-sim)**.
- 의존성 게이트: Step 2(BPE)는 Step 3의 long-range·Step 5의 특화·Step 6의 MTP **이전**에 반드시 완료. Step 1 패리티 하니스는 이후 모든 cross-check의 전제.
- 토큰 예산이 빠듯하면 우선순위: **Step 1 → 3 → 4** (인프라·핵심 차별점·안정성)를 먼저 굳히고, Step 5–7은 시간 허용 범위에서.

---

## 부록: 귀속 위생 (writeup/저널에 그대로 적용)

- **V4-DISTINCTIVE (진짜 V4로 공부)**: CSA/HCA 압축-KV 하이브리드, mHC, FP4 QAT, hash_moe 부트스트랩 + Sqrt(Softplus) affinity + n_group 삭제, Muon/hybrid-ZeRO *통합*.
- **INHERITED (V2/V3/V3.2 출처로 공부)**: DSA+Lightning Indexer(V3.2), DeepSeekMoE(V2), aux-loss-free balancing(Wang et al. 2024 / V3), MTP(V3와 identical).
- **REFUTED/MISATTRIBUTED (V4 first로 공부 금지)**: 'Muon 1.6T MoE 최초'(보고서는 Jordan 2024 + Liu/Moonlight 인용, 'first' 문구 없음), 'V4가 R/V series 병합', deepseek_v4 docs가 고정 ground truth.
- **학습 토큰 수 정정** `[보고서 확인, PDF §4]`: 초안에서 '~32T'를 `[추정/미확인]`으로 깔았으나 PDF로 확정 — **Flash 32T, Pro 33T 토큰**으로 사전학습. ('초안 작성 시점엔 fact-checker가 PDF를 못 열어 미확인이었음 → 직접 확인 후 확정'의 사례.)
- **저널 규율**: 모든 사실 주장에 `[보고서 확인]` vs `[추정/미확인]` 태그. 각 단계 저널은 (a) 코딩 전 예측, (b) 본인 곡선, (c) 그 곡선에서 답한 ablation 질문을 포함 — ablation 답 없는 엔트리는 미완.
- **레퍼런스 신뢰순**: `.py` 소스(modular_/modeling_/configuration_deepseek_v4.py, 핀 commit) > 렌더 docs(Mixtral 템플릿 잔재 있음). V4는 2026-04 **preview**로 인용. 세부 수치(61층/384전문가/FLOP·KV)는 PDF 도표에서 직접.