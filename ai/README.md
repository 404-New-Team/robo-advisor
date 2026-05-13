# robo-advisor-ai

강화학습(PPO) + 에이전틱 리서치(Claude API)를 결합한 포트폴리오 관리 AI.

## 아키텍처 개요

```
RSS 뉴스
  → NewsFetcher                          # 뉴스 수집
    → NewsStore (ChromaDB)               # 임베딩 저장 & 유사 사례 검색
      → AgenticRAGResearchAgent          # LangGraph 기반 자가 교정 루프
        → RiskDetector (Claude Haiku)    # 리스크 태그 탐지 (tool_use)
          → PortfolioEnv (Gymnasium)     # RL 환경 — 관측 공간에 리스크 반영
            → PPOAgent (SB3)             # 포트폴리오 비중 결정
              → SafeGuardMonitor         # 규제 제약 검증
                → SHAPExplainer          # 의사결정 설명 (XAI)

백테스트 & 연구:
  WalkForwardBacktest → PerformanceMetrics (12개 지표)
  MVO (Markowitz) → 효율적 프론티어
  ANOVA 검증 1: 보상 함수 변형 비교 (One-way)
  ANOVA 검증 2: DRL vs MVO vs EqualWeight (One-way + η²)
  ANOVA 검증 3: 전략 × 시장 국면 (Two-way + 편부분 η²)
```

## 프로젝트 구조

```
ai/
├── src/
│   ├── agents/
│   │   └── ppo_agent.py            # stable-baselines3 PPO 래퍼
│   │
│   ├── backtest/
│   │   ├── metrics.py              # 12개 성과 지표 (CAGR, Sharpe, VaR, CVaR …)
│   │   ├── mvo.py                  # Markowitz MVO + 효율적 프론티어
│   │   └── walk_forward.py         # Walk-Forward 백테스트 (슬라이딩 윈도우)
│   │
│   ├── data/
│   │   ├── market_data.py          # yfinance / pykrx 다운로드 + parquet 캐싱
│   │   └── preprocessors.py        # 기술적 지표 11종 (RSI, MACD, BB 등)
│   │
│   ├── envs/
│   │   ├── portfolio_env.py        # Gymnasium 환경 — 관측(125), 행동(10)
│   │   └── risk_state.py           # 리스크 태그 상태 관리
│   │
│   ├── pipeline/
│   │   └── integrated_pipeline.py  # 전체 모듈 연결 파이프라인
│   │
│   ├── research/
│   │   ├── agentic_rag.py          # LangGraph 에이전틱 RAG (자가 교정 루프)
│   │   ├── anova_analysis.py       # ANOVA 검증 1: 보상 함수 변형 비교
│   │   ├── strategy_anova.py       # ANOVA 검증 2: DRL vs MVO vs EqualWeight
│   │   ├── market_regime_anova.py  # ANOVA 검증 3: 전략 × 시장 국면 Two-way
│   │   ├── documents.py            # 문서 인덱싱 & 청킹
│   │   ├── news_fetcher.py         # RSS 피드 기반 뉴스 수집
│   │   ├── news_store.py           # ChromaDB 임베딩 저장 & 유사 사례 검색
│   │   ├── risk_detector.py        # Claude API → 리스크 태그 탐지
│   │   └── risk_tags.py            # tool_use 스키마 정의
│   │
│   ├── safeguard/
│   │   └── monitor.py              # 포트폴리오 비중 한도 체크
│   │
│   ├── xai/
│   │   └── shap_explainer.py       # SHAP Summary / Force / Waterfall 플롯
│   │
│   └── config/
│       └── settings.yaml
│
├── experiments/                     # 독립 실행 실험 스크립트
│   ├── reward_experiment.py         # 보상 함수 변형 3종 비교
│   ├── walk_forward_experiment.py   # Walk-Forward 백테스트
│   ├── mvo_experiment.py            # MVO 최적화 & 효율적 프론티어
│   ├── shap_experiment.py           # SHAP 의사결정 시각화
│   ├── strategy_anova_experiment.py # ANOVA 검증 2
│   └── market_regime_anova_experiment.py # ANOVA 검증 3
│
├── tests/
│   ├── test_env.py
│   ├── test_research.py
│   ├── test_agentic_rag.py
│   ├── test_news_documents.py
│   ├── test_technical_indicators.py
│   └── conftest.py
│
└── train.py                         # PPO 학습 엔트리포인트
```

## 환경 세팅

### 요구사항

- Python 3.10 이상
- 인터넷 연결 (yfinance / pykrx 데이터, RSS 뉴스)
- `ANTHROPIC_API_KEY` — 뉴스 리스크 탐지 사용 시에만 필요

### 1. 가상환경 생성 및 활성화

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

주요 패키지:

| 패키지 | 용도 |
|--------|------|
| `gymnasium` | RL 환경 인터페이스 |
| `stable-baselines3` | PPO 학습 |
| `torch` | SB3 백엔드 |
| `yfinance` | 해외 주가 데이터 |
| `pykrx` | 국내 ETF 데이터 (6자리 KRX 종목코드) |
| `scipy` | 최적화 (MVO SLSQP), ANOVA 검정 |
| `statsmodels` | Two-way ANOVA Type-II SS |
| `shap` | SHAP KernelExplainer |
| `feedparser` | RSS 뉴스 수집 |
| `chromadb` | 뉴스 임베딩 저장 & 검색 |
| `anthropic` | Claude API (리스크 탐지, 에이전틱 RAG) |
| `pyarrow` | 가격 데이터 parquet 캐싱 |

### 3. API 키 설정 (선택)

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

RL 학습과 백테스트만 실행하는 경우 불필요합니다.

## 실행 방법

모든 명령은 `ai/` 디렉터리에서 실행합니다.

### PPO 모델 학습

```bash
python train.py
```

```
PPO   포트폴리오 가치: 1.2417  (+24.17%)
B&H   포트폴리오 가치: 1.4197  (+41.97%)
초과 수익:             -17.80%p
```

### 실험 스크립트

```bash
# Walk-Forward 백테스트 (12개 성과 지표)
python experiments/walk_forward_experiment.py

# MVO 최적화 & 효율적 프론티어
python experiments/mvo_experiment.py

# SHAP 의사결정 시각화 (학습된 모델 필요)
python experiments/shap_experiment.py --checkpoint checkpoints/portfolio_ppo_best

# ANOVA 검증 2: DRL vs MVO vs EqualWeight
python experiments/strategy_anova_experiment.py --drl_timesteps 30000

# ANOVA 검증 3: 전략 × 시장 국면 Two-way ANOVA
python experiments/market_regime_anova_experiment.py --drl_timesteps 30000
```

결과 파일은 `experiments/results/`에 JSON으로 저장됩니다.

## 관측 공간 & 보상 함수

### 관측 공간 (125차원)

| 구성 | 크기 | 내용 |
|------|------|------|
| 시장 피처 | 10 × 11 = 110 | ret1d, ret5d, ret20d, vol20d, mom20d, RSI14, MACD, MACD_signal, BB_upper, BB_lower, BB_position |
| 리스크 태그 | 5 | regulatory_risk, earnings_shock, geopolitical_risk, market_stress, liquidity_risk |
| 현재 비중 | 10 | 자산별 현재 포트폴리오 비중 |

### 보상 함수 변형 (RewardVariant)

| 변형 | 설명 |
|------|------|
| `R1_LOGRET` | 로그 수익률만 (baseline) |
| `R2_SHARPE` | 롤링 Sharpe ratio (위험 조정) |
| `R3_FULL` | 로그수익률 − 리스크 집중도 페널티 − 낙폭 페널티 (기본값) |

### MDD 조기 종료

에피소드 중 포트폴리오 낙폭(drawdown)이 `max_drawdown_threshold`(기본 15%)를 초과하면 에피소드를 즉시 종료합니다.

## 성과 지표 (12개)

Walk-Forward 백테스트와 ANOVA 실험에서 공통으로 사용됩니다.

| 지표 | 설명 |
|------|------|
| CAGR | 연환산 복리 수익률 |
| Total Return | 누적 수익률 |
| Sharpe Ratio | 위험 조정 수익률 (연환산) |
| Sortino Ratio | 하방 위험 조정 수익률 |
| Calmar Ratio | CAGR / 최대낙폭 |
| Max Drawdown | 최대 낙폭 |
| Volatility | 연환산 변동성 |
| VaR 95% | 일일 최대 손실 (95% 신뢰구간) |
| CVaR 95% | 기대 꼬리 손실 (조건부 VaR) |
| Alpha | 벤치마크 대비 초과 수익 (연환산) |
| Beta | 벤치마크 대비 민감도 |
| Information Ratio | 초과 수익의 일관성 |

## ANOVA 검증

### 검증 1 — 보상 함수 변형 비교

R1_LOGRET / R2_SHARPE / R3_FULL 세 변형의 에피소드 보상을 One-way ANOVA로 비교합니다.

```bash
python experiments/reward_experiment.py
```

### 검증 2 — 전략 비교 (One-way ANOVA)

DRL(PPO) / MVO(max_sharpe) / EqualWeight 세 전략의 Walk-Forward 폴드별 CAGR을 비교합니다.

- η² (eta-squared) 효과 크기
- Tukey HSD post-hoc 검정

```bash
python experiments/strategy_anova_experiment.py
```

### 검증 3 — 시장 국면별 성과 (Two-way ANOVA)

**Factor A**: 전략 (DRL / MVO / EqualWeight)  
**Factor B**: 시장 국면 (Bull / Bear / Sideways)  
**반응변수**: 폴드별 CAGR

- Type-II SS (statsmodels), 불균형 설계 대응
- 편부분 η² — 전략 / 국면 / 상호작용 각각 산출

```bash
python experiments/market_regime_anova_experiment.py --threshold_bull 0.10 --threshold_bear -0.10
```

## 설정

[src/config/settings.yaml](src/config/settings.yaml)에서 모든 파라미터를 조정합니다.

```yaml
environment:
  tickers: ["SPY", "QQQ", "GLD", "TLT", "EFA", "AAPL", "MSFT", "069500", "102110", "233740"]
  window_size: 20
  transaction_cost: 0.00015
  slippage: 0.0005
  max_drawdown_threshold: 0.15   # MDD 조기 종료 임계값

reward:
  risk_penalty_lambda: 0.5       # R3_FULL 리스크 집중도 페널티 계수
  drawdown_penalty_mu: 1.0       # R3_FULL 낙폭 페널티 계수

training:
  total_timesteps: 500000
  n_envs: 4
  learning_rate: 0.0003
  batch_size: 256

backtest:
  risk_free_rate: 0.02           # 무위험 수익률 (Sharpe, Alpha 계산)

research:
  model: "claude-haiku-4-5-20251001"  # 리서치 에이전트 모델
  max_tokens: 512
```

## 캐시

| 경로 | 내용 |
|------|------|
| `ai/.cache/market/` | yfinance / pykrx parquet 캐시 |
| `ai/.cache/chromadb/` | ChromaDB 뉴스 임베딩 |
| `checkpoints/` | PPO 모델 가중치 |
| `experiments/results/` | 실험 결과 JSON |

캐시가 손상된 경우 해당 디렉터리를 삭제 후 재실행하세요.
