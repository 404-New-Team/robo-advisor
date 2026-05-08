# robo-advisor-ai

강화학습(PPO) + 에이전틱 리서치(Claude API)를 결합한 포트폴리오 관리 AI.

## 아키텍처 개요

```
RSS 뉴스
  → NewsFetcher                        # 뉴스 수집
    → NewsStore (ChromaDB)             # 임베딩 저장 & 유사 사례 검색
      → RiskDetector (Claude API)      # 리스크 이벤트 탐지
        → PortfolioEnv (Gymnasium)     # RL 환경 (관측 공간에 리스크 반영)
          → PPOAgent (stable-baselines3) # 포트폴리오 비중 결정
            → SafeGuardMonitor         # 규제 제약 검증
              → SHAPExplainer          # 의사결정 설명
```

## 프로젝트 구조

```
ai/
├── data/                      # 데이터 수집 & 전처리
│   ├── market_data.py         # yfinance 다운로드 + parquet 캐싱
│   └── preprocessors.py       # 로그수익률, 롤링 Z-score, compute_features
│
├── envs/                      # RL 환경 (핵심)
│   ├── portfolio_env.py       # Gymnasium 환경 본체
│   └── risk_state.py          # 리스크 태그 상태 관리
│
├── agents/                    # RL 에이전트
│   └── ppo_agent.py           # stable-baselines3 PPO 래퍼
│
├── research/                  # 리서치 에이전트
│   ├── news_fetcher.py        # RSS 피드 기반 뉴스 수집
│   ├── news_store.py          # ChromaDB 임베딩 저장 & 유사 사례 검색
│   ├── risk_detector.py       # Claude API → 리스크 태그 탐지
│   └── risk_tags.py           # tool_use 스키마 정의
│
├── safeguard/                 # 규제 제약 검증
│   └── monitor.py             # 포트폴리오 비중 한도 체크
│
├── xai/                       # 의사결정 설명
│   └── shap_explainer.py      # SHAP 기반 피처 중요도
│
├── pipeline/
│   └── integrated_pipeline.py # 위 모듈 전체 연결
│
└── config/
    └── settings.yaml
```

루트 스크립트:

```
train.py          # PPO 에이전트 학습 (RL 환경만 사용, API 키 불필요)
test_env.py       # RL 환경 동작 확인 (무작위 에이전트)
test_research.py  # 리서치 에이전트 파이프라인 확인 (API 키 필요)
```

## 환경 세팅

### 요구사항

- Python 3.10 이상
- 인터넷 연결 (yfinance 데이터 다운로드, RSS 뉴스 수집)
- `ANTHROPIC_API_KEY` (뉴스 리스크 탐지 기능 사용 시에만 필요)

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
| `torch` | stable-baselines3 백엔드 |
| `yfinance` | 주가 데이터 다운로드 |
| `feedparser` | RSS 뉴스 수집 |
| `chromadb` | 뉴스 임베딩 저장 & 유사 사례 검색 |
| `anthropic` | Claude API (뉴스 리스크 탐지) |
| `shap` | 의사결정 설명 |
| `pyarrow` | 가격 데이터 parquet 캐싱 |

### 3. API 키 설정 (선택)

뉴스 기반 리스크 탐지(`RiskDetector`)를 사용하는 경우에만 필요합니다.

```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-...

# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...
```

RL 학습만 돌리는 경우에는 불필요합니다.

### 4. 동작 확인

**RL 환경 확인** (API 키 불필요):
```bash
python test_env.py
```

```
관측 공간 shape: (35,)
step 1 | reward: 0.0148 | value: 1.0149 | drawdown: 0.0000
...
```

**리서치 에이전트 확인** (API 키 필요):
```bash
python test_research.py
```

```
=== 1. 뉴스 수집 ===
  [yahoo_finance] 10건 수집
  ...
=== 3. 리스크 탐지 ===
  geopolitical_risk         level=0.72  confidence=0.85
    근거: 무역 분쟁 관련 뉴스 감지
...
```

**PPO 학습**:
```bash
python train.py
```

```
PPO   포트폴리오 가치: 1.2417  (+24.17%)
B&H   포트폴리오 가치: 1.4197  (+41.97%)
초과 수익:             -17.80%p
```

관측 공간 `(35,)` = 시장 피처(5종목 × 5피처) + 리스크 태그(5개) + 현재 비중(5개).

## 설정 변경

[src/config/settings.yaml](src/config/settings.yaml)에서 티커, 학습 파라미터 등을 조정할 수 있습니다.

```yaml
environment:
  tickers: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
  window_size: 20
  transaction_cost: 0.001

training:
  total_timesteps: 500000
  learning_rate: 0.0003
  batch_size: 256
```

## 캐시

| 경로 | 내용 |
|------|------|
| `ai/.cache/market/` | yfinance parquet 캐시 |
| `ai/.cache/chromadb/` | ChromaDB 뉴스 임베딩 |
| `checkpoints/` | PPO 모델 가중치 |

캐시가 손상된 경우 해당 디렉터리를 삭제 후 재실행하세요.
