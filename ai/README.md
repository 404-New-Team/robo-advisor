# Robo-Advisor AI

강화학습(PPO) 기반 동적 자산배분 엔진과 에이전틱 RAG 리서치 모듈을 결합한 로보어드바이저 AI 실험 패키지입니다. 이 모듈은 가격 데이터 수집, 피처 전처리, Gymnasium 포트폴리오 환경, PPO 학습, Walk-Forward 백테스트, MVO 비교, SHAP 의사결정 해석, ANOVA 통계 검증을 담당합니다.

본 프로젝트의 목표는 단순히 수익률을 높이는 것이 아니라, 다음 질문에 답할 수 있는 검증 가능한 파이프라인을 만드는 것입니다.

- 왜 이 보상 함수를 선택했는가?
- 어떤 시장 조건에서 전략이 유효하거나 약한가?
- 모델이 특정 포트폴리오 비중을 결정할 때 어떤 피처가 영향을 주었는가?
- DRL, MVO, 동일가중 전략 간 성과 차이가 통계적으로 유의한가?

## 아키텍처

```text
시장 가격 데이터
  -> market_data.py
  -> preprocessors.py
     - 로그 수익률
     - rolling z-score 정규화
     - RSI, MACD, Bollinger Band 등 기술적 지표
  -> PortfolioEnv
     - 관측 공간: 시장 피처 + 리스크 태그 + 현재 비중
     - 행동 공간: 연속형 자산별 allocation logit
     - 거래비용, 슬리피지, MDD Safe-Guard 반영
  -> PPOAgent
     - Stable-Baselines3 PPO 학습
     - checkpoints/portfolio_ppo_best.zip 저장
  -> 실험 모듈
     - Walk-Forward 백테스트
     - MVO 및 동일가중 비교
     - SHAP 의사결정 해석
     - ANOVA 통계 검증

뉴스/RAG 데이터
  -> NewsFetcher
  -> NewsStore(ChromaDB)
  -> AgenticRAGResearchAgent
  -> RiskDetector
  -> RiskState
  -> PortfolioEnv 관측 공간 및 Safe-Guard 연동
```

## 폴더 구조

```text
ai/
├── train.py
├── requirements.txt
├── checkpoints/
│   ├── portfolio_ppo_best.zip
│   └── best_score.txt
├── experiments/
│   ├── reward_experiment.py
│   ├── walk_forward_experiment.py
│   ├── mvo_experiment.py
│   ├── shap_experiment.py
│   ├── strategy_anova_experiment.py
│   ├── market_regime_anova_experiment.py
│   └── results/
├── src/
│   ├── api/            ← FastAPI 서버 (main.py)
│   ├── agents/
│   ├── backtest/
│   ├── config/
│   ├── data/
│   ├── envs/
│   ├── pipeline/
│   ├── research/
│   ├── safeguard/
│   └── xai/
└── tests/
```

## 환경 설정

Python 3.10 이상을 권장합니다.

```powershell
cd ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt`는 실제 검증된 버전으로 고정되어 있습니다. 주요 패키지:

```
gymnasium==1.2.3  stable-baselines3==2.8.0  torch==2.11.0
shap==0.51.0      langgraph==1.1.10          fastapi==0.136.1
```

리서치 에이전트에서 Claude API를 사용할 경우 환경 변수를 설정합니다. RL 학습, 백테스트, MVO, SHAP, ANOVA 실험만 실행할 때는 필수는 아닙니다.

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

## API 서버

FastAPI 서버는 포트 8001에서 실행됩니다. PPO 체크포인트와 Research Agent는 서버 시작 시 자동으로 로딩됩니다.

### 로컬 실행

```powershell
cd ai
.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8001
```

### 엔드포인트

| Method | Path | 기능 |
| --- | --- | --- |
| GET | /health | 서버 상태 및 모델 로드 여부 |
| POST | /ai/optimize | PPO 포트폴리오 최적화 (MVO 폴백) |
| POST | /ai/shap | SHAP 의사결정 해석 + Force/Summary Plot |
| POST | /ai/research | AgenticRAG 뉴스 분석 |
| POST | /ai/backtest | Walk-Forward 백테스트 (drl / mvo / equal_weight) |

### 요청 예시

```powershell
# 포트폴리오 최적화
Invoke-RestMethod -Uri http://localhost:8001/ai/optimize -Method Post `
  -ContentType "application/json" `
  -Body '{"tickers":["SPY","QQQ","GLD"],"risk_level":"moderate","start_date":"2022-01-01","end_date":"2024-12-31"}'

# SHAP 해석
Invoke-RestMethod -Uri http://localhost:8001/ai/shap -Method Post `
  -ContentType "application/json" `
  -Body '{"tickers":["SPY","QQQ","GLD"],"target_asset":"SPY","date":"2024-01-02"}'

# 백테스트 (mvo 전략)
Invoke-RestMethod -Uri http://localhost:8001/ai/backtest -Method Post `
  -ContentType "application/json" `
  -Body '{"tickers":["SPY","QQQ","GLD"],"strategy":"mvo","start_date":"2021-01-01","end_date":"2024-12-31"}'
```

### 타임아웃 정책

| 엔드포인트 | 타임아웃 | 비고 |
| --- | ---: | --- |
| /ai/optimize | 4s | 캐시 히트 시 ~26ms |
| /ai/shap | 10s | KernelExplainer nsamples=50 |
| /ai/research | 25s | LLM 호출 포함 |
| /ai/backtest | 60s | Walk-Forward 훈련 포함 |

PPO 관측 공간이 현재 티커 수와 맞지 않으면 자동으로 MVO 폴백이 적용됩니다 (기학습 모델은 10개 자산 기준 obs=125).

## Docker 실행

`ai/` 모듈만 독립적으로 재현 실행할 수 있도록 Dockerfile과 compose 파일을 제공합니다. backend/frontend는 아직 통합 대상에서 제외하고, AI 학습 및 실험 스크립트 실행 환경만 고정합니다.

이미지 빌드:

```powershell
cd ai
docker compose build ai
```

리서치 에이전트까지 Docker 안에서 실행하려면 현재 셸에 API 키를 먼저 설정합니다. 키를 사용하지 않는 RL/백테스트/ANOVA 실험은 이 단계가 필요 없습니다.

```powershell
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

테스트 실행:

```powershell
docker compose run --rm ai
```

AI API 서버 실행:

```powershell
docker compose --profile serve up serve
```

PPO 학습 실행:

```powershell
docker compose --profile train run --rm train
```

SHAP 실험 실행:

```powershell
docker compose --profile experiments run --rm shap
```

임의의 실험 스크립트를 실행할 수도 있습니다.

```powershell
docker compose run --rm ai python experiments/reward_experiment.py
docker compose run --rm ai python experiments/walk_forward_experiment.py
docker compose run --rm ai python experiments/mvo_experiment.py
docker compose run --rm ai python experiments/strategy_anova_experiment.py
docker compose run --rm ai python experiments/market_regime_anova_experiment.py
```

컨테이너에서 생성한 산출물은 호스트의 다음 경로에 유지됩니다.

| 호스트 경로 | 컨테이너 경로 | 용도 |
| --- | --- | --- |
| `ai/checkpoints/` | `/app/checkpoints/` | PPO 체크포인트, 학습 곡선 |
| `ai/experiments/results/` | `/app/experiments/results/` | 실험 JSON, SHAP plot |
| `ai/.cache/` | `/app/.cache/` | 시장 데이터 및 ChromaDB 캐시 |

## 실행 순서

모든 명령은 `ai/` 디렉터리에서 실행하는 것을 기준으로 작성했습니다.

### 1. PPO 학습

```powershell
python train.py
```

설정 파일의 기본 학습 스텝은 500,000입니다.

```yaml
training:
  total_timesteps: 500000
  n_envs: 4
  learning_rate: 0.0003
  batch_size: 256
```

학습 결과는 항상 `ai/checkpoints/` 아래에 저장됩니다.

- `checkpoints/portfolio_ppo_best.zip`
- `checkpoints/best_score.txt`
- `checkpoints/learning_curve.png`

실행 위치에 따라 체크포인트 경로가 달라지는 문제를 막기 위해 `train.py`는 스크립트 위치 기준의 절대 경로로 저장합니다.

### 2. Walk-Forward 백테스트

```powershell
python experiments/walk_forward_experiment.py
```

결과 파일:

- `experiments/results/walk_forward_result.json`

현재 결과 요약:

| 지표 | 값 |
| --- | ---: |
| Fold 수 | 7 |
| 평균 CAGR | 0.0671 |
| CAGR 표준편차 | 0.1958 |
| 평균 Sharpe | 0.9264 |
| Sharpe 표준편차 | 2.0862 |
| 평균 MDD | 0.0899 |

해석: 일부 fold에서는 높은 Sharpe와 CAGR을 보였지만, 2022년 구간처럼 손실 fold도 존재합니다. 따라서 DRL 전략의 성과는 시장 국면에 민감하며, 안정적인 우월성을 주장하기보다는 국면별 강점과 약점을 함께 해석해야 합니다.

### 3. MVO 기준선 실험

```powershell
python experiments/mvo_experiment.py
```

결과 파일:

- `experiments/results/mvo_frontier.json`
- `experiments/results/mvo_walk_forward_result.json`

MVO는 Markowitz 평균-분산 최적화 기반 기준선입니다. 제약조건은 비중 합 1, 공매도 금지, 개별 자산 최대 비중 제한을 사용합니다. DRL과 비교할 때 정적 최적화 기준선으로 활용합니다.

### 4. SHAP 의사결정 해석

```powershell
python experiments/shap_experiment.py
```

또는 체크포인트를 명시합니다.

```powershell
python experiments/shap_experiment.py --checkpoint checkpoints/portfolio_ppo_best
```

결과 파일:

- `experiments/results/shap/summary_plot.png`
- `experiments/results/shap/force_plot.png`
- `experiments/results/shap/waterfall_plot.png`

주의사항:

- 체크포인트가 없으면 기본적으로 실패하도록 설계했습니다.
- 짧은 시연용 모델이 필요할 때만 `--demo_train`을 명시적으로 사용합니다.
- SHAP multi-output 반환 형태 `(samples, features, outputs)`를 내부적으로 `(outputs, samples, features)`로 변환하여 자산별 SHAP 값을 올바르게 해석합니다.

예시 Top-5 결과:

| 피처 | 평균 절대 SHAP |
| --- | ---: |
| SPY_macd | 0.108127 |
| GLD_bb_position | 0.099520 |
| QQQ_bb_upper | 0.074286 |
| GLD_ret5d | 0.058365 |
| 069500_bb_lower | 0.053334 |

## 강화학습 환경 설계

### 데이터

기본 자산 universe는 해외 ETF/주식과 국내 ETF를 함께 사용합니다.

```yaml
tickers:
  - SPY
  - QQQ
  - GLD
  - TLT
  - EFA
  - AAPL
  - MSFT
  - 069500
  - 102110
  - 233740
```

가격 데이터는 `yfinance`와 `pykrx`로 수집하고, 캐시는 `ai/.cache/market/`에 저장합니다.

### 전처리

- 수익률은 로그 수익률을 사용합니다.
- 결측치는 수집 및 병합 단계에서 정리합니다.
- 피처는 rolling z-score로 정규화합니다.
- 기술적 지표는 RSI, MACD, MACD signal, Bollinger upper/lower/position 등을 포함합니다.

### 관측 공간

관측 공간은 총 125차원입니다.

| 구성 | 차원 | 설명 |
| --- | ---: | --- |
| 시장 피처 | 110 | 10개 자산 x 11개 피처 |
| 리스크 태그 | 5 | regulatory, earnings, geopolitical, market stress, liquidity |
| 현재 비중 | 10 | 직전 포트폴리오 비중 |

### 행동 공간

행동 공간은 10차원 연속형 Box입니다. PPO가 각 자산의 logit을 출력하면 환경 내부에서 softmax를 적용해 비중 합이 1인 포트폴리오로 변환합니다.

### 거래 비용과 Safe-Guard

```yaml
transaction_cost: 0.00015
slippage: 0.0005
max_drawdown_threshold: 0.15
```

리밸런싱 시 turnover에 비례해 수수료와 슬리피지를 차감합니다. 에피소드 중 MDD가 15%를 초과하면 조기 종료합니다.

## 보상 함수

`RewardVariant`는 세 가지 변형을 제공합니다.

| 변형 | 수식 개요 | 목적 |
| --- | --- | --- |
| `R1_LOGRET` | `log(1 + r_t)` | 단순 수익률 기반 baseline |
| `R2_SHARPE` | rolling mean / rolling std | 위험 조정 수익률 반영 |
| `R3_FULL` | log return - risk concentration penalty - drawdown penalty | 수익률, 리스크 태그, MDD를 함께 반영 |

기본 학습은 `R3_FULL`을 사용합니다. 리스크 태그가 높고 특정 자산에 비중이 집중될수록 페널티가 커지며, drawdown도 보상에서 차감됩니다.

## ANOVA 검증 결과

ANOVA 실험은 성능의 절대적 우월성을 만들기 위한 절차가 아니라, 전략과 설정의 차이를 통계적으로 해석하기 위한 절차입니다.

### 검증 1. 보상 함수 변형 비교

실행:

```powershell
python experiments/reward_experiment.py
```

결과 파일:

- `experiments/results/anova_result.json`
- `experiments/results/reward_distribution.png`

요약:

| 항목 | 값 |
| --- | ---: |
| F-statistic | 84.1638 |
| p-value | 0.0000 |
| 유의수준 | 0.05 |
| 결론 | 유의함 |

그룹 평균:

| 보상 함수 | 평균 episode reward | 표준편차 | N |
| --- | ---: | ---: | ---: |
| R1_LOGRET | -0.0915 | 0.0139 | 30 |
| R2_SHARPE | -1.3669 | 0.8806 | 30 |
| R3_FULL | -1.7183 | 0.0890 | 30 |

해석: 보상 함수 변형 간 episode reward 분포에는 통계적으로 유의한 차이가 있습니다. 다만 보상 함수마다 스케일과 페널티 구조가 다르므로, 평균 reward가 가장 높은 `R1_LOGRET`을 곧바로 최종 전략으로 선택하기보다는 백테스트의 MDD, Sharpe, CAGR과 함께 판단해야 합니다.

### 검증 2. DRL vs MVO vs EqualWeight

실행:

```powershell
python experiments/strategy_anova_experiment.py
```

결과 파일:

- `experiments/results/strategy_anova_result.json`
- `experiments/results/strategy_anova_raw_returns.json`

요약:

| 항목 | 값 |
| --- | ---: |
| Metric | fold_cagr |
| F-statistic | 0.0988 |
| p-value | 0.906417 |
| eta squared | 0.0109 |
| 결론 | 유의하지 않음 |

전략별 평균 CAGR:

| 전략 | 평균 | 표준편차 | N |
| --- | ---: | ---: | ---: |
| DRL | 0.0772 | 0.1990 | 7 |
| MVO | 0.1646 | 0.4973 | 7 |
| EqualWeight | 0.1216 | 0.3452 | 7 |

해석: 전략 간 평균 CAGR 차이는 관측되지만 p-value가 높아 통계적으로 유의하지 않습니다. 현재 결과로는 DRL이 MVO 또는 동일가중보다 우월하다고 주장할 수 없습니다. 대신 fold별 변동성이 크고 표본 수가 작아, 성과 차이를 안정적으로 검출하기 어렵다는 한계를 보고해야 합니다.

### 검증 3. 전략 x 시장 국면 Two-way ANOVA

실행:

```powershell
python experiments/market_regime_anova_experiment.py
```

결과 파일:

- `experiments/results/market_regime_anova_result.json`
- `experiments/results/market_regime_anova_records.json`

요약:

| 요인 | F | p-value | partial eta squared | 결론 |
| --- | ---: | ---: | ---: | --- |
| Strategy | 0.1206 | 0.887419 | 0.0197 | 유의하지 않음 |
| Regime | 5.4566 | 0.020634 | 0.4763 | 유의함 |
| Interaction | 0.1774 | 0.945767 | 0.0558 | 유의하지 않음 |

국면별 평균 CAGR:

| 국면 | 평균 CAGR | Fold 수 |
| --- | ---: | ---: |
| Bear | -0.3455 | 1 |
| Bull | 0.2626 | 5 |
| Sideways | -0.1116 | 1 |

해석: 전략 자체보다 시장 국면이 fold CAGR에 더 큰 영향을 준 것으로 나타났습니다. 다만 Bear와 Sideways가 각각 1개 fold뿐이므로, 국면 효과의 통계적 해석에는 주의가 필요합니다. 보고서에서는 유의성 자체보다 표본 불균형과 시장 국면별 취약성을 함께 논의하는 것이 적절합니다.

## XAI 해석 방향

SHAP 결과는 PPO 정책이 특정 관측값에서 어떤 입력 피처에 민감했는지 설명합니다. 이 프로젝트에서는 모델 출력 action logit을 설명 대상으로 사용하고, 자산별 output에 대해 Force Plot, Waterfall Plot, Summary Plot을 생성합니다.

보고서 작성 시 다음 관점으로 해석합니다.

- 특정 자산 비중이 증가한 시점에 어떤 기술적 지표가 양의 기여를 했는가?
- 변동성, Bollinger Band, momentum 계열 피처가 리스크 회피 또는 비중 확대에 어떤 영향을 주었는가?
- SHAP 값은 인과관계가 아니라 모델 민감도 해석이라는 점을 명시합니다.

## 테스트

AI 모듈 테스트:

```powershell
pytest tests
```

루트에서 전체 테스트를 실행하려면:

```powershell
pytest ai/tests backend/tests
```

현재 테스트는 환경 smoke test, 기술적 지표 계산, 뉴스 문서 정규화, ChromaDB 기반 citation 검색, Agentic RAG 흐름, Backend schema 및 router mock 테스트를 포함합니다.

## 주요 산출물

| 산출물 | 경로 |
| --- | --- |
| PPO best checkpoint | `ai/checkpoints/portfolio_ppo_best.zip` |
| 학습 곡선 | `ai/checkpoints/learning_curve.png` |
| Walk-Forward 결과 | `ai/experiments/results/walk_forward_result.json` |
| MVO 결과 | `ai/experiments/results/mvo_walk_forward_result.json` |
| SHAP plot | `ai/experiments/results/shap/` |
| 보상 함수 ANOVA | `ai/experiments/results/anova_result.json` |
| 전략 ANOVA | `ai/experiments/results/strategy_anova_result.json` |
| 시장 국면 ANOVA | `ai/experiments/results/market_regime_anova_result.json` |

## 한계와 개선 방향

- DRL 전략은 평균적으로 MVO와 동일가중을 유의하게 초과하지 못했습니다.
- Walk-Forward fold 수가 7개라 통계 검정력이 제한됩니다.
- 시장 국면 분류에서 Bear와 Sideways 표본이 각각 1개로 불균형합니다.
- 보상 함수별 episode reward는 스케일이 달라 직접적인 우열 비교에 주의가 필요합니다.
- 향후 개선은 seed 반복 학습, 더 긴 기간의 데이터, 국면별 균형 표본 확보, lambda 탐색, 리밸런싱 주기별 민감도 분석 순서로 진행하는 것이 좋습니다.

## 면책 조항

본 프로젝트는 교육 및 연구 목적의 로보어드바이저 실험 시스템입니다. 산출된 포트폴리오 비중, 백테스트 결과, 리서치 리포트는 실제 투자 조언이 아니며 미래 수익을 보장하지 않습니다. 실제 투자 판단은 투자자 본인의 책임하에 이루어져야 합니다.
