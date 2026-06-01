Robo Advisor — 에이전틱 RAG 및 강화학습 기반 통합 자율 로보어드바이저
404 New 팀 | 박주영 · 손기령 · 홍지연 · 심서연 · 심은
산학프로젝트(캡스톤디자인) | 2025

목차

프로젝트 개요
시스템 아키텍처
실행 방법 (Docker)
핵심 기술 키워드 정리
보상 함수 설계 근거
성능 지표 요약
ANOVA 검증 결과 요약
에러 분석 및 개선 방향
폴더 구조
면책 조항


1. 프로젝트 개요
단순 수익률 극대화가 아닌 "왜 이 자산을 매수했는가"를 정량적으로 설명할 수 있는 AI 포트폴리오 관리 시스템이다.
핵심 기능

강화학습 자산배분 : PPO 기반 동적 포트폴리오 비중 최적화 → ai/ai/agents/ppo_agent.py
에이전틱 리스크 탐지 : Claude API로 뉴스 → 리스크 태그 자동 변환 → ai/ai/research/risk_detector.py
SHAP 의사결정 해석 : KernelExplainer로 포트폴리오 결정 근거 제시 → ai/ai/xai/shap_explainer.py
Safe-Guard 모니터링 : MDD·집중도·리스크 스코어 3중 안전장치 → ai/ai/safeguard/monitor.py
통합 파이프라인 : 뉴스 수집 → 리스크 주입 → RL 추론 → SHAP 해석 → ai/ai/pipeline/integrated_pipeline.py


2. 시스템 아키텍처
[RSS 뉴스 수집]          news_fetcher.py
      ↓
[ChromaDB 벡터 저장]     news_store.py
      ↓
[Claude API 리스크 탐지] risk_detector.py
      ↓  RiskTag(name, level, confidence, source)
[RL 환경 관측 공간 주입] portfolio_env.py  → inject_risk_tags()
      ↓  obs: [시장피처×50] + [리스크태그×5] + [현재비중×10]
[PPO 에이전트 추론]      ppo_agent.py
      ↓  action: logit → softmax → 포트폴리오 비중
[Safe-Guard 검증]        safeguard/monitor.py
      ↓  조정된 비중
[FastAPI 서빙]           backend/  :8000
      ↓  REST API
[Streamlit 대시보드]     frontend/ :8501
서비스 구성 (docker-compose.yml)

backend  : 포트 8000 / FastAPI AI 추론 서버
frontend : 포트 8501 / Streamlit 대시보드
chromadb : 포트 8001 / 뉴스 벡터 DB


3. 실행 방법 (Docker)
사전 요구사항

Docker Desktop 설치
.env 파일 준비 (.env.example 참고)

bashcp .env.example .env
# .env 파일에서 ANTHROPIC_API_KEY 설정
전체 서비스 실행
bashdocker-compose up --build
개별 서비스 접근

대시보드 : http://localhost:8501
API Swagger UI : http://localhost:8000/docs
ChromaDB : http://localhost:8001

AI 모델 학습
bash# Docker 내부에서 실행
docker-compose exec backend python ai/train.py

# 또는 로컬에서 직접 실행
cd ai
python train.py
데이터 다운로드
bashbash scripts/download_data.sh

4. 핵심 기술 키워드 정리

4.1 강화학습 (Reinforcement Learning)
DRL (Deep Reinforcement Learning)
심층 강화학습. 딥러닝(신경망)과 강화학습을 결합한 방법론. 에이전트가 시장 환경과 상호작용하며 보상을 최대화하는 방향으로 스스로 포트폴리오 비중을 학습한다. 본 프로젝트에서는 PPO 알고리즘을 사용한 DRL이 핵심 자산배분 엔진 역할을 하며 MVO·동일가중 전략과 성능을 비교하는 기준 모델이다.
→ ai/ai/agents/ppo_agent.py, ai/ai/envs/portfolio_env.py
PPO (Proximal Policy Optimization)
연속 행동 공간에서 안정적인 On-policy 학습 알고리즘. clip_range=0.2로 정책 업데이트 폭을 제한하여 학습 안정성을 확보한다.
→ ppo_agent.py: PPO("MlpPolicy", ...)
Stable-Baselines3
PPO 구현 라이브러리. total_timesteps=500,000, batch_size=256, gamma=0.99 설정.
→ ppo_agent.py
Gymnasium
OpenAI 표준 RL 환경 인터페이스. observation_space, action_space, reset(), step() 구현.
→ portfolio_env.py: class PortfolioEnv(gym.Env)
관측 공간 (Observation Space)
Box(-inf, inf, shape=(n_obs,)). 구성: 시장 피처(n_assets×5) + 리스크 태그(5) + 현재 비중(n_assets).
→ portfolio_env.py: self.observation_space
행동 공간 (Action Space)
Box(-1.0, 1.0, shape=(n_assets,)). logit 벡터를 softmax 변환하여 합=1 포트폴리오 비중으로 변환. 공매도 금지 자동 충족.
→ portfolio_env.py: self.action_space
MlpPolicy
2층 MLP 신경망 정책. 125차원 관측 → 256 유닛 은닉층 × 2 → n_assets 행동.
→ ppo_agent.py: PPO("MlpPolicy", ...)
Episode Reward
에피소드 누적 보상. TrainingLogger가 rollout/ep_rew_mean 값을 기록하여 학습 곡선 시각화.
→ train.py: class TrainingLogger
Checkpoint
save_freq=50,000 스텝마다 모델 저장. checkpoints/portfolio_ppo_best.zip이 최고 성능 모델.
→ ppo_agent.py: CheckpointCallback
DummyVecEnv
다중 환경 병렬 학습. n_envs=4 설정으로 학습 속도 향상.
→ train.py: DummyVecEnv([env_fn] * n_envs)
softmax 변환
action → exp(x-max) / sum → 비중 벡터. 합이 1이 되도록 보장.
→ portfolio_env.py: _softmax()

4.2 금융공학
로그 수익률 (Log Return)
r_t = ln(P_t / P_(t-1)). 정상성(stationarity) 확보 및 시계열 합산 가능성을 위해 단순 수익률 대신 사용.
→ preprocessors.py: log_returns()
비정상성 (Non-stationarity)
가격 시계열은 단위근을 가져 분포가 시간에 따라 변함. 로그 수익률 + 롤링 Z-score로 해소.
→ preprocessors.py: rolling_zscore()
롤링 Z-score
(x - rolling_mean) / rolling_std. window=20으로 미래 데이터 누수(Look-ahead Bias) 방지.
→ preprocessors.py: rolling_zscore(window)
MDD (Maximum Drawdown)
(peak_value - current_value) / peak_value. 최대 낙폭. Safe-Guard 20% 한도, RL 환경 내 15% 조기 종료 기준.
→ portfolio_env.py: _peak_value, safeguard/monitor.py
거래비용 (Transaction Cost)
cost = transaction_cost × turnover. turnover = sum(|new_w - old_w|). transaction_cost=0.001 (0.1%).
→ portfolio_env.py: step()
Walk-Forward Validation
시간 순서 유지 백테스트. 학습 기간(24개월) → 테스트(6개월) → 슬라이딩(6개월) × 10 fold.
→ scripts/run_backtest.sh
Sharpe Ratio
(수익률 - 무위험수익률) / 변동성. 위험조정 수익률 지표. 현재 DRL Sharpe=1.27, MVO=0.75.
Sortino Ratio
수익률 / 하방 변동성. 일반 Sharpe는 상승·하락 변동성을 모두 분모에 넣지만 Sortino는 손실이 발생한 날의 변동성만 분모로 사용. DRL처럼 손실 방어에 강점이 있는 전략을 평가할 때 Sharpe보다 유리하게 나타남. 현재 DRL Sortino=2.28.
Calmar Ratio
CAGR / |MDD|. 연환산 수익률을 최대 낙폭으로 나눈 값. 값이 높을수록 같은 손실 리스크로 더 많은 수익을 올린다는 의미. 현재 DRL Calmar=2.93으로 MVO 대비 우위.
VaR / CVaR
Value at Risk / Conditional VaR. 95% 신뢰수준 최대 손실 / 그 초과 손실 평균.
Alpha / Beta
Alpha: 벤치마크 초과 수익. Beta: 시장 민감도 (KOSPI 기준).
Herfindahl 지수
sum(w²). 포트폴리오 집중도 측정. 리스크 페널티 계산에 활용.
→ portfolio_env.py: _compute_reward()
MVO (Mean-Variance Optimization)
Markowitz 평균-분산 최적화. scipy.optimize.minimize(SLSQP), 252일 롤링 공분산, 최대 샤프비율 목적함수. DRL 비교 기준 전략.

4.3 리스크 탐지 및 에이전틱 RAG
리스크 태그 (Risk Tag)
{name, level, confidence, source} 구조체. 5종: regulatory_risk, earnings_shock, geopolitical_risk, market_stress, liquidity_risk.
→ risk_state.py: RiskTag, RISK_TAG_NAMES
RiskState
5개 리스크 레벨을 관리하는 상태 객체. decay_rate=0.95로 매 스텝 감쇠. confidence-weighted max로 갱신.
→ risk_state.py: class RiskState
시간 감쇠 (Decay)
매 스텝 level *= 0.95. 오래된 뉴스 신호를 점진적으로 희석하여 현재 시점 리스크에 집중.
→ risk_state.py: step_decay()
inject_risk_tags()
외부 리서치 에이전트가 리스크 태그를 RL 환경에 주입하는 인터페이스. E2E 파이프라인의 핵심 연결 고리.
→ portfolio_env.py: inject_risk_tags()
Claude API tool_use
tool_choice={"type":"tool","name":"report_risk_events"}로 구조화 출력 강제. JSON 파싱 실패 방지.
→ risk_detector.py: RISK_DETECTION_TOOL
RSS 피드
Yahoo Finance, MarketWatch, Reuters 3개 소스. feedparser로 수집. title + summary를 text로 저장.
→ news_fetcher.py: RSS_FEEDS
ChromaDB
뉴스 벡터 임베딩 저장소. DefaultEmbeddingFunction(). CHROMA_HOST 환경변수로 서버/로컬 모드 자동 전환.
→ news_store.py: class NewsStore
search_by_risk()
5종 리스크별 키워드로 ChromaDB 검색하여 관련 기사 선별. 단순 최신순이 아닌 의미 기반 필터링.
→ news_store.py: search_by_risk()
Self-Correction
검색 관련성 미달 시 쿼리 자동 재작성 후 재검색 (최대 2회). 환각(Hallucination) 억제.
LangGraph
계획-검색-평가-재작성-분석-검증-교정 7단계 워크플로우. 에이전틱 RAG 구현.

4.4 XAI — SHAP 의사결정 해석
SHAP (SHapley Additive exPlanations)
게임 이론 Shapley 값 기반 피처 기여도 계산. 금융 XAI 의무화 규제 대응.
→ shap_explainer.py
KernelExplainer
모델 구조 무관 범용 explainer. PPO 신경망(블랙박스)에 적용 가능. background=100개 샘플.
→ shap_explainer.py: shap.KernelExplainer
fit()
배경 데이터로 explainer 초기화. 학습 완료 후 1회 실행.
→ shap_explainer.py: fit()
explain()
특정 관측값의 SHAP 값 계산. multi-output(자산별 action)이면 절댓값 평균.
→ shap_explainer.py: explain()
top_k_features()
상위 k개 주요 결정 요인 반환. 의사결정 근거 보고용 (k=5 기본값).
→ shap_explainer.py: top_k_features()
피처 이름 구조
{ticker}_ret1d, {ticker}_ret5d, {ticker}_ret20d, {ticker}_vol20d, {ticker}mom20d + 5개 리스크 태그 + w{ticker}.
→ integrated_pipeline.py: _build_feature_names()
Summary Plot
전체 피처 중요도 시각화. MSFT_bb_upper, AAPL_macd_signal 등 상위 피처 확인.
Force Plot
특정 시점 개별 결정 해석. 기준값에서 각 피처의 +/- 기여를 누적 표시.

4.5 Safe-Guard 안전장치
SafeGuardConfig
max_drawdown=0.20, max_position=0.40, max_aggregate_risk=0.75.
→ safeguard/monitor.py: SafeGuardConfig
validate()
비중 검증 후 위반 시 조정. 3단계 순서:
1단계: MDD 초과 → 전 종목 균등 분산 강제
2단계: 단일 종목 집중 초과 → clip 후 재정규화
3단계: 집계 리스크 과다 → 비례 축소
→ safeguard/monitor.py: validate()
weights_to_action()
검증된 비중 → 환경 action logit 변환. log(w + 1e-8).
→ safeguard/monitor.py: weights_to_action()
violations 로그
모든 위반 이력을 {rule, value} 형태로 기록. XAI 감사 요건 충족.
→ safeguard/monitor.py: self.violations
학습 레이어 Safe-Guard
MDD 15% 초과 시 에피소드 조기 종료. RL 환경 내부 제약.
→ portfolio_env.py 설계

4.6 데이터 수집 및 전처리
yfinance
해외 주식·ETF 가격 수집. auto_adjust=True로 수정주가 사용.
→ market_data.py: yf.download()
Parquet 캐싱
{tickers}{start}{end}.parquet으로 저장. 중복 다운로드 방지.
→ market_data.py: CACHE_DIR
compute_features()
5개 피처 계산: ret1d, ret5d, ret20d, vol20d, mom20d. 롤링 Z-score 정규화 포함.
→ preprocessors.py: compute_features()
Forward Fill
prices.ffill(). 공휴일·거래 정지 결측치를 이전 거래일 값으로 대체.
→ market_data.py: prices.ffill()
ADF 검정
Augmented Dickey-Fuller. 가격=비정상, 로그 수익률=정상 검증.

4.7 백엔드 / 인프라
FastAPI
비동기 REST API 서버. 5개 엔드포인트: /health, /optimize, /explain, /research, /backtest.
→ backend/app/routers/
Pydantic
입출력 스키마 정의 및 자동 검증. Swagger UI 자동 생성.
→ backend/app/
MSA 구조
Streamlit은 FastAPI를 통해서만 모델 접근. 모델 파일 직접 로드 금지.
→ docker-compose.yml: API_BASE_URL
Docker Compose
backend(8000) + frontend(8501) + chromadb(8001) 단일 명령 실행.
→ docker-compose.yml
MySQL
사용자·포트폴리오·백테스트·SHAP·리서치 결과 6개 테이블.
환경 변수
ANTHROPIC_API_KEY, CHROMA_HOST, CHROMA_PORT를 .env로 관리. GitHub 업로드 금지.
→ .env.example

4.8 통계 검증 (ANOVA)
One-way ANOVA
3개 이상 그룹 간 평균 차이의 유의성 검증. F-통계량, p-value, η² 보고.
→ 4_ANOVA_Results.py
Two-way ANOVA
두 요인(전략 × 시장국면)의 주효과 및 교호작용 동시 검증.
→ 4_ANOVA_Results.py
Tukey HSD
p<0.05 달성 시 어느 그룹 간에 차이가 있는지 사후 검정.
η² (eta-squared)
효과 크기. 0.01=소, 0.06=중, 0.14=대. 통계적 유의성과 별개로 실질적 의미 판단.
검증 1 : 보상 함수 3종(R1/R2/R3) 간 episode reward 비교. F=43.04, p<0.001, η²=0.68.
검증 2 : DRL vs MVO vs 동일가중 CAGR 비교. F=0.17, p=0.844 (비유의, 표본 부족).
검증 3 : 시장 국면 주효과 유의(p<0.05, η²=0.48). Tukey: 하락장↔상승장(p=0.008).

4.9 개발 중 이슈 및 대시보드 관련 키워드
Walk-Forward 대시보드
백테스트 결과를 사용자에게 시각화하는 화면. 폴드별 수익률 추이 차트, DRL·MVO·동일가중 3개 전략 누적 수익률 비교 선 그래프, 폴드별 Sharpe·MDD 히트맵, 시장 국면 구분선 등을 표시하는 것이 목표.
현재 상태: 어떤 차트를 어떤 순서로 배치할지 팀 내 기획 논의 필요.
equal_weight (동일가중)
전체 자산에 동일 비중(1/N)을 배분하는 가장 단순한 기준 전략. DRL·MVO와 성능 비교용 기준선(baseline)으로 사용. 대시보드에서 3개 전략을 동시에 그릴 때 실시간 계산 부하로 렌더링 지연 발생.
현재 상태: 사전 계산 후 캐싱하거나 백그라운드 스레드로 분리하는 방향으로 개선 필요.
AI 파트 성능 이슈
현재 DRL CAGR(11.84%) < MVO CAGR(20.15%)로 고전적 수학 최적화보다 성능이 낮은 상황. 원인은 R3_FULL 보상 함수의 손실 회피 편향이 상승장에서도 과도하게 보수적으로 작동하기 때문.
현재 상태: risk_penalty_lambda 값을 0.5~1.0 범위로 재튜닝하거나 학습 스텝 증가(500k → 1M), 데이터 기간 확장(2018~)으로 개선 가능.
API 명세
프론트엔드와 백엔드가 주고받는 데이터 형식 정의 문서. 엔드포인트별 HTTP 메서드, 요청 Body, 응답 스키마를 Pydantic으로 정의. docs/api.md에 명세 유지 필요.
현재 상태: BE 파트에서 수정된 명세를 docs/api.md와 Pydantic 스키마에 동기화 필요.
DB 설계
MySQL 6개 테이블 구조. 수정 사항 발생 시 backend/ 내 마이그레이션 스크립트와 ERD 다이어그램을 함께 업데이트해야 함.
현재 상태: BE 파트 수정분 반영 필요.
SHAP 피처 한국어화
shap_explainer.py: _build_feature_names()에서 반환하는 피처명을 한국어로 변환.
예시: AAPL_ret1d → 애플 1일 수익률 / regulatory_risk → 규제 리스크 / w_MSFT → 마이크로소프트 현재 비중.
현재 상태: 번역만으로는 직관성이 부족할 수 있음. "최근 1일 상승폭(애플)" 같은 풀어쓰기 방식 검토 권장.
Sortino Ratio
수익률 / 하방 변동성. 일반 Sharpe는 상승·하락 변동성을 모두 분모에 넣지만 Sortino는 손실이 발생한 날의 변동성만 분모로 사용. DRL처럼 손실 방어에 강점이 있는 전략을 평가할 때 Sharpe보다 유리하게 나타남. 현재 DRL Sortino=2.28.
현재 상태: 대시보드 App 탭 수치 표시 예정.
Calmar Ratio
CAGR / |MDD|. 연환산 수익률을 최대 낙폭으로 나눈 값. 값이 높을수록 같은 손실 리스크로 더 많은 수익을 올린다는 의미. 현재 DRL Calmar=2.93으로 MVO 대비 우위.
현재 상태: 대시보드 App 탭 수치 표시 예정.
기능 추가 검토
현재 구현된 4개 페이지(Portfolio, Research Trace, SHAP Explain, ANOVA Results) 외 추가 기능 필요 여부 논의. 명세서 요구사항인 강화학습 성과·리스크 모니터링 2개 탭이 미구현 상태.
현재 상태: 추가 기능보다 기존 미구현 탭 완성이 우선순위. 여유 있으면 이상거래 탐지(Isolation Forest) 보너스 과제 검토.

5. 보상 함수 설계 근거
3종 보상 함수 비교
R1_LOGRET
수식: reward = log(1 + r_t)
특성: 단순 수익률 극대화. 상승장 유리, 하락장 취약.
R2_SHARPE
수식: reward = (r_t - rf) / sigma_t
특성: 위험조정 수익률. 안정적 수렴.
R3_FULL (최종 채택)
수식: reward = log_ret - λ × risk × concentration - μ × drawdown
특성: 손실 방어 최우선.
R3_FULL 수식 상세 (실제 코드 기반)
python# portfolio_env.py: _compute_reward()
log_ret = np.log1p(portfolio_return + 1e-8)

aggregate_risk = np.mean(risk_state.to_array())      # 5개 리스크 태그 평균
concentration = np.sum(weights ** 2)                  # Herfindahl 지수
risk_penalty = risk_penalty_lambda * aggregate_risk * concentration

drawdown = (peak_value - portfolio_value) / peak_value
drawdown_penalty = drawdown_penalty_mu * drawdown

reward = log_ret - risk_penalty - drawdown_penalty
파라미터 설정 (settings.yaml)
yamlreward:
  risk_penalty_lambda: 0.5   # 리스크 집중도 페널티 강도 (탐색 범위: 0.5~5.0)
  drawdown_penalty_mu: 1.0   # 낙폭 페널티 강도
채택 근거
R3_FULL을 채택한 이유는 episode reward 기준으로는 R2_SHARPE가 우수하나 (F=43.04, p<0.001, R2>R3 유의), 실제 백테스트에서 하락장 손실 방어 (DRL -21.34% vs MVO -40.37%) 성과를 종합적으로 고려했을 때 리스크 관리 목적에 더 부합하기 때문이다. 또한 뉴스 기반 리스크 태그가 보상 함수에 직접 반영되어 E2E 파이프라인이 완성된다.

6. 성능 지표 요약
Walk-Forward 백테스트 결과 (10 fold, 2021~2026)
CAGR
DRL-PPO: 11.84% / MVO: 20.15% / 동일가중: 17.18% / KOSPI: ~6% → DRL 열위
MDD
DRL-PPO: 8.58% / MVO: 11.65% / 동일가중: 높음 / KOSPI: ~-25% → DRL 우위
Sharpe
DRL-PPO: 1.27 / MVO: 0.75 / 동일가중: 낮음 → DRL 우위
Sortino
DRL-PPO: 2.28 → DRL 우위
Calmar
DRL-PPO: 2.93 → DRL 우위
VaR(95%) / CVaR(95%)
DRL-PPO: 낮음 / MVO: 높음 / 동일가중: 높음 → DRL 우위
Alpha
DRL-PPO: 양(+) / MVO: 양(+) / 동일가중: 기준 → DRL 우위
Beta
DRL-PPO: 0.5 내외 / MVO: 0.7 내외 / 동일가중: 0.9 내외 / KOSPI: 1.00 → DRL 우위
승률
DRL-PPO: 70% (7/10 fold)
시장 국면별 성과
하락장 : DRL -21.34% / MVO -40.37% / 동일가중 -43.86% → DRL 우위 (+19%p 방어)
상승장 : DRL 낮음 / MVO 높음 / 동일가중 높음 → MVO/EW 우위
횡보장 : 세 전략 모두 중간 → 혼재
해석: DRL 전략은 CAGR 기준으로 MVO를 하회하지만, Sharpe·MDD·Sortino·Calmar 등 위험조정 지표에서 우위를 보인다. 현재 R3_FULL의 손실 회피 편향이 상승장에서도 과도하게 작동한 결과다.

7. ANOVA 검증 결과 요약
검증 1 — 보상 함수 3종 비교 (One-way ANOVA)
대상: R1_LOGRET vs R2_SHARPE vs R3_FULL (Episode Reward)
F-통계량: 43.04
p-value:  < 0.001  유의
η²:       0.68 (Large 효과 — 분산의 68% 설명)

Tukey HSD 사후 검정:
  R2_SHARPE > R1_LOGRET  (p < 0.001)
  R2_SHARPE > R3_FULL    (p < 0.001)
  R1_LOGRET ≈ R3_FULL    (p = 0.285, 비유의)
검증 2 — DRL vs MVO vs 동일가중 (One-way ANOVA)
대상: 3개 전략 × 10 fold CAGR
F-통계량: 0.17
p-value:  0.844  비유의
η²:       0.01 (Small 효과)

비유의성 원인:
  - 표본 부족 (10개 폴드, 검정력 한계)
  - 높은 폴드 간 변동성 (DRL σ ≈ 12~13%)
  - 향후 데이터 확장(2018~) 시 유의성 확보 가능성
검증 3 — 시장 국면별 전략 성과 (Two-way ANOVA)
요인 A: 전략 (DRL / MVO / 동일가중)
요인 B: 시장 국면 (상승 / 횡보 / 하락)

시장 국면 주효과: p < 0.05  유의   η² = 0.48 (Large)
전략 주효과:     p > 0.05  비유의  η² = 0.02
교호작용:        p = 0.915  비유의  η² = 0.04

Tukey HSD (시장 국면):
  하락장 ↔ 상승장  p = 0.008  유의
  하락장 ↔ 횡보장  p = 0.031  유의
  상승장 ↔ 횡보장  p = 0.412  비유의

결론: 전략 선택보다 시장 국면이 성과의 주요 결정 요인

8. 에러 분석 및 개선 방향
알려진 이슈
MVO CAGR > DRL CAGR
원인: R3_FULL 손실 회피 편향이 상승장에서 과도하게 작동.
해결: risk_penalty_lambda 0.5~1.0 재튜닝, 하이브리드 보상 함수 검토.
equal_weight 동시 렌더링 속도 이슈
원인: 실시간 계산 부하.
해결: 백그라운드 캐싱 또는 사전 계산으로 개선.
Walk-Forward 폴드 수 부족
원인: 2021~2026 데이터로 10개 폴드 한계.
해결: 2018년부터 데이터 확장 → 20개 폴드 확보.
ANTHROPIC_API_KEY 미설정 폴백
원인: 로컬 데모 응답으로 대체.
해결: API 비용 최적화 전략 별도 수립.
강화학습 성과·리스크 모니터링 탭 미구현
원인: FE 개발 시간 부족.
해결: 추후 보완 예정.
향후 개선 방향

모델: Transformer 기반 정책 네트워크 도입, LSTM으로 N일 윈도우(N=20~30) 관측 공간 확장
데이터: 글로벌 ETF(SPY, QQQ, TLT, GLD) 추가, 2018년부터 학습 데이터 확장
인프라: MLflow 실험 추적, Airflow 일일 자동 재학습 파이프라인
리스크: Isolation Forest 이상거래 탐지 모듈 추가
서비스: 프로덕션 전환 시 실시간 스트리밍(WebSocket), 수평 스케일링 구축


9. 폴더 구조
robo-advisor/
├── ai/
│   ├── ai/
│   │   ├── agents/
│   │   │   └── ppo_agent.py          # PPO 에이전트 (Stable-Baselines3)
│   │   ├── config/
│   │   │   └── settings.yaml         # 하이퍼파라미터 설정
│   │   ├── data/
│   │   │   ├── market_data.py        # yfinance 데이터 수집 + Parquet 캐싱
│   │   │   └── preprocessors.py      # 로그 수익률, 롤링 Z-score, 피처 계산
│   │   ├── envs/
│   │   │   ├── portfolio_env.py      # Gymnasium 포트폴리오 환경
│   │   │   └── risk_state.py         # 리스크 태그 상태 관리 + 시간 감쇠
│   │   ├── pipeline/
│   │   │   └── integrated_pipeline.py  # E2E 통합 파이프라인
│   │   ├── research/
│   │   │   ├── news_fetcher.py       # RSS 피드 수집
│   │   │   ├── news_store.py         # ChromaDB 벡터 저장소
│   │   │   ├── risk_detector.py      # Claude API 리스크 탐지
│   │   │   └── risk_tags.py          # tool_use 스키마 정의
│   │   ├── safeguard/
│   │   │   └── monitor.py            # Safe-Guard 3중 안전장치
│   │   └── xai/
│   │       └── shap_explainer.py     # SHAP KernelExplainer
│   └── train.py                      # PPO 학습 스크립트
├── backend/
│   └── app/
│       ├── main.py                   # FastAPI 앱
│       └── routers/                  # 5개 엔드포인트
├── frontend/
│   └── streamlit_app/
│       ├── app.py                    # 메인 대시보드
│       └── pages/                    # Portfolio, Research, SHAP, ANOVA
├── docs/
│   ├── api.md
│   ├── architecture.md
│   └── reward_design.md
├── scripts/
│   ├── download_data.sh
│   └── run_backtest.sh
├── docker-compose.yml
├── .env.example
└── README.md

10. 면책 조항
본 시스템은 교육 목적으로 개발되었으며 실제 투자 조언에 사용할 수 없습니다.
백테스팅 성과는 미래 수익을 보장하지 않습니다.
수집한 뉴스 데이터는 교육 목적으로만 사용하며 재배포를 금지합니다.
