from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # 서버 환경: 헤드리스 렌더링
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── 경로 상수 ─────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = APP_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "portfolio_ppo_best.zip"
RESULTS_DIR = APP_DIR / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4)

# 엔드포인트별 타임아웃 (초)
TIMEOUT_OPTIMIZE = 30.0
TIMEOUT_SHAP = 30.0
TIMEOUT_RESEARCH = 30.0
TIMEOUT_BACKTEST = 60.0

# ─── 전역 상태 ─────────────────────────────────────────────────────────────────
_ppo_model: Any = None       # stable_baselines3.PPO
_ppo_tickers: list[str] = []  # PPO 학습 시 사용한 티커 순서 (settings.yaml 기준)
_research_agent: Any = None  # AgenticRAGResearchAgent
_research_agent_error: str = ""
_prices_cache: dict[str, pd.DataFrame] = {}

_global_risk_state: Any = None  # /ai/research 호출 시 업데이트, /ai/optimize·shap에서 공유


class ResearchAgentUnavailable(RuntimeError):
    pass


def _init_research_agent() -> Any:
    global _research_agent, _research_agent_error

    if _research_agent is not None:
        return _research_agent

    try:
        from ..research.agentic_rag import AgenticRAGResearchAgent

        _research_agent = AgenticRAGResearchAgent()
        _research_agent_error = ""
        logger.info("Research Agent 초기화 완료")
        return _research_agent
    except Exception as exc:
        _research_agent_error = str(exc)
        logger.warning(f"Research Agent 초기화 실패: {exc}")
        return None


def _get_research_agent() -> Any:
    agent = _init_research_agent()
    if agent is None:
        detail = _research_agent_error or "알 수 없는 초기화 오류"
        raise ResearchAgentUnavailable(f"Research Agent 초기화 실패: {detail}")
    return agent


# ─── Lifespan: 서버 시작 시 모델 사전 로딩 ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _ppo_model, _ppo_tickers, _global_risk_state
    from ..envs.risk_state import RiskState
    _global_risk_state = RiskState()

    # ── settings.yaml에서 학습 티커 로딩 ─────────────────────────────────────
    try:
        import yaml
        settings_path = APP_DIR / "src" / "config" / "settings.yaml"
        with open(settings_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        _ppo_tickers = cfg.get("environment", {}).get("tickers", [])
        logger.info(f"학습 티커 로딩 완료: {_ppo_tickers}")
    except Exception as exc:
        logger.warning(f"settings.yaml 로딩 실패: {exc}")

    # ── 데이터 관련 모듈 사전 임포트 (첫 요청 지연 방지) ──────────────────────
    try:
        import pandas as pd  # noqa: F401
        import numpy as np  # noqa: F401
        import pyarrow  # noqa: F401
        from ..data.market_data import fetch_prices  # noqa: F401
        from ..data.preprocessors import compute_features  # noqa: F401
        from ..backtest.metrics import compute_metrics  # noqa: F401
        from ..backtest.mvo import MVO  # noqa: F401
        logger.info("데이터/수치 모듈 사전 임포트 완료")
    except Exception as exc:
        logger.warning(f"사전 임포트 일부 실패: {exc}")

    # ── PPO 체크포인트 로딩 ────────────────────────────────────────────────────
    ckpt = _find_checkpoint()
    if ckpt:
        try:
            from stable_baselines3 import PPO as SB3PPO
            _ppo_model = SB3PPO.load(str(ckpt))
            logger.info(f"PPO 체크포인트 로딩 완료: {ckpt.name} (obs={_ppo_model.observation_space.shape})")
        except Exception as exc:
            logger.warning(f"PPO 로딩 실패 → MVO 폴백 사용: {exc}")
    else:
        logger.info("PPO 체크포인트 없음 → MVO 폴백 사용")

    _init_research_agent()

    yield

    _executor.shutdown(wait=False)


# ─── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Robo-Advisor AI Service",
    version="2.0.0",
    description=(
        "PPO 강화학습 포트폴리오 최적화 · SHAP 의사결정 해석 "
        "· Agentic RAG 리서치 · Walk-Forward 백테스트\n\n"
        "**면책 조항**: 본 API는 교육 목적으로 개발되었으며 실제 투자 조언에 사용할 수 없습니다. "
        "백테스팅 성과가 미래 수익을 보장하지 않습니다."
    ),
    lifespan=lifespan,
)


# ─── 공통 에러 응답 ────────────────────────────────────────────────────────────
def _error(code: int, message: str, detail: str = "") -> JSONResponse:
    return JSONResponse(
        status_code=code,
        content={"status": "error", "code": code, "message": message, "detail": detail},
    )


# ─── 비동기 실행 헬퍼 ─────────────────────────────────────────────────────────
async def _run(fn, *args, timeout: float = TIMEOUT_OPTIMIZE, **kwargs) -> Any:
    """동기 함수를 스레드 풀에서 실행. 타임아웃 초과 시 None 반환."""
    loop = asyncio.get_event_loop()
    try:
        return await asyncio.wait_for(
            loop.run_in_executor(_executor, lambda: fn(*args, **kwargs)),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return None
    except Exception as exc:
        logger.error(f"_run 오류: {exc}\n{traceback.format_exc()}")
        raise


# ─── 유틸리티 ─────────────────────────────────────────────────────────────────
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (f != f) else f  # NaN 체크
    except Exception:
        return default


def _find_checkpoint() -> Optional[Path]:
    """이용 가능한 PPO 체크포인트를 찾는다 (best → 최신 순)."""
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    if not CHECKPOINT_DIR.exists():
        return None
    zips = sorted(CHECKPOINT_DIR.glob("**/*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def _get_or_fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """캐시 히트 → 즉시 반환. 미스 → 수집 후 캐싱."""
    from ..data.market_data import fetch_prices
    key = f"{'|'.join(sorted(tickers))}_{start}_{end}"
    if key not in _prices_cache:
        _prices_cache[key] = fetch_prices(tickers, start, end, use_cache=True)
    return _prices_cache[key]


def _mvo_weight_min(n_assets: int, risk_level: str) -> float:
    """종목 수와 risk_level에 따라 MVO 최소 비중 결정. 분산투자 강제."""
    base = {"low": 0.05, "moderate": 0.03, "high": 0.02}.get(risk_level, 0.03)
    # 최소 비중 합이 1을 초과하지 않도록 조정
    return min(base, 0.9 / max(n_assets, 1))


def _apply_risk_cap(weights: np.ndarray, risk_level: str) -> np.ndarray:
    """risk_level에 따른 개별 자산 최대 비중 상한 적용."""
    cap = {"low": 0.20, "moderate": 0.35, "high": 0.40}.get(risk_level, 0.35)
    clipped = np.clip(weights, 0.0, cap)
    total = clipped.sum()
    if total < 1e-8:
        return np.ones(len(weights)) / len(weights)
    return (clipped / total).astype(float)


def _fig_to_b64(fig) -> str:
    """matplotlib Figure → PNG → Base64 문자열."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str)


def _wf_result_to_cache(result) -> dict:
    """WalkForwardResult → JSON 캐시 형식 변환."""
    folds = []
    for fm in result.folds:
        m = fm.metrics
        folds.append({
            "test_start": fm.test_start,
            "test_end": fm.test_end,
            "total_return": _safe_float(m.total_return),
            "cagr": _safe_float(m.cagr),
            "sharpe": _safe_float(m.sharpe),
            "sortino": _safe_float(m.sortino),
            "calmar": _safe_float(m.calmar),
            "max_drawdown": _safe_float(m.max_drawdown),
            "volatility": _safe_float(m.volatility),
            "var_95": _safe_float(m.var_95),
            "cvar_95": _safe_float(m.cvar_95),
            "alpha": _safe_float(m.alpha),
            "beta": _safe_float(m.beta),
            "information_ratio": _safe_float(m.information_ratio),
        })
    return {
        "summary": {
            "mean_cagr": _safe_float(result.mean_cagr),
            "mean_sharpe": _safe_float(result.mean_sharpe),
            "mean_sortino": _safe_float(result.mean_sortino),
            "mean_calmar": _safe_float(result.mean_calmar),
            "mean_max_drawdown": _safe_float(result.mean_max_drawdown),
            "mean_volatility": _safe_float(getattr(result, "mean_volatility", 0.0)),
            "mean_var_95": _safe_float(result.mean_var_95),
            "mean_cvar_95": _safe_float(result.mean_cvar_95),
            "mean_alpha": _safe_float(result.mean_alpha),
            "mean_beta": _safe_float(result.mean_beta),
            "mean_information_ratio": _safe_float(result.mean_information_ratio),
        },
        "folds": folds,
    }


def _fetch_benchmark_returns(start: str, end: str) -> tuple[float, float]:
    """(kospi_total_return, sp500_total_return) 수집. 실패 시 (0.0, 0.0)."""
    try:
        import yfinance as yf
        data = yf.download(["^GSPC", "^KS11"], start=start, end=end,
                           auto_adjust=True, progress=False)
        closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
        results = {}
        for sym in ["^GSPC", "^KS11"]:
            if sym in closes.columns:
                s = closes[sym].dropna()
                if len(s) >= 2:
                    results[sym] = float(s.iloc[-1] / s.iloc[0] - 1)
                else:
                    results[sym] = 0.0
            else:
                results[sym] = 0.0
        return results.get("^KS11", 0.0), results.get("^GSPC", 0.0)
    except Exception:
        return 0.0, 0.0


def _build_wf_response(strategy: str, cached: dict, kospi_ret: float, sp500_ret: float) -> dict:
    """캐시된 Walk-Forward 결과 → 응답 형식 변환."""
    summary = cached.get("summary", {})
    folds = cached.get("folds", [])

    mean_return = _safe_float(summary.get("mean_cagr", 0.0))
    fold_returns = [_safe_float(f.get("total_return", f.get("cagr", 0.0))) for f in folds]
    win_rate = float(np.mean([r > 0 for r in fold_returns])) if fold_returns else 0.0

    return {
        "strategy": strategy,
        "metrics": {
            "total_return": mean_return,
            "sharpe_ratio": _safe_float(summary.get("mean_sharpe", 0.0)),
            "sortino_ratio": _safe_float(summary.get("mean_sortino", 0.0)),
            "calmar_ratio": _safe_float(summary.get("mean_calmar", 0.0)),
            "max_drawdown": -abs(_safe_float(summary.get("mean_max_drawdown", 0.0))),
            "volatility": _safe_float(summary.get("mean_volatility", summary.get("std_cagr", 0.0))),
            "win_rate": win_rate,
        },
        "benchmark_comparison": {
            "kospi_return": kospi_ret,
            "sp500_return": sp500_ret,
            "strategy_alpha": mean_return - sp500_ret,
        },
        "walk_forward_results": [
            {
                "period": f"{f.get('test_start', '')}~{f.get('test_end', '')}",
                "return": _safe_float(f.get("total_return", f.get("cagr", 0.0))),
                "sharpe": _safe_float(f.get("sharpe", 0.0)),
            }
            for f in folds
        ],
    }


def _build_feature_names(tickers: list[str]) -> list[str]:
    """PortfolioEnv 관측 벡터의 피처명 리스트 생성."""
    suffixes = [
        "ret1d", "ret5d", "ret20d", "vol20d", "mom20d",
        "rsi14", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_position",
    ]
    names: list[str] = []
    for s in suffixes:
        for t in tickers:
            names.append(f"{t}_{s}")
    names += ["regulatory_risk", "earnings_shock", "geopolitical_risk", "market_stress", "liquidity_risk"]
    names += [f"weight_{t}" for t in tickers]
    return names


# ─── Pydantic 요청 스키마 ──────────────────────────────────────────────────────
class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    risk_level: str = Field("moderate", pattern="^(low|moderate|high)$")
    start_date: str
    end_date: str


class ShapRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    target_asset: str
    date: str


class ResearchRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    max_results: int = Field(5, ge=1, le=20)


class BacktestRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    strategy: str = Field("drl", pattern="^(drl|mvo|equal_weight)$")
    start_date: str
    end_date: str


# ─── GET /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "rl_engine": _ppo_model is not None,
        "rl_checkpoint": str(_find_checkpoint() or "없음"),
        "rag_agent": _research_agent is not None,
        "rag_agent_error": _research_agent_error,
        "shap_explainer": True,
        "timestamp": _utc_now(),
    }


# ─── POST /ai/optimize ────────────────────────────────────────────────────────
@app.post("/ai/optimize")
async def optimize(req: OptimizeRequest):
    if not req.tickers:
        return _error(400, "tickers가 비어 있습니다.", "최소 1개 이상의 티커가 필요합니다.")

    def _compute():
        from ..data.preprocessors import log_returns
        from ..backtest.metrics import compute_metrics
        from ..backtest.mvo import MVO, MVOConfig
        from ..envs.portfolio_env import PortfolioEnv
        from ..envs.risk_state import RiskState

        prices = _get_or_fetch_prices(req.tickers, req.start_date, req.end_date)
        if prices.empty or len(prices) < 30:
            raise ValueError(f"데이터 부족: {len(prices)}행 (최소 30 거래일 필요)")

        tickers = list(prices.columns)
        n = len(tickers)
        window = min(20, len(prices) // 3)

        # ── 비중 결정: PPO 우선 → MVO 폴백 ───────────────────────────────────
        weights: np.ndarray
        use_ppo = False

        if _ppo_model is not None and _ppo_tickers:
            req_set = set(tickers)
            train_set = set(_ppo_tickers)
            if req_set.issubset(train_set):
                # 요청 티커가 학습 티커 범위 내 → 학습 티커 전체로 env 빌드 후 부분 추출
                try:
                    train_prices = _get_or_fetch_prices(_ppo_tickers, req.start_date, req.end_date)
                    if not train_prices.empty and len(train_prices) >= 30:
                        train_window = min(20, len(train_prices) // 3)
                        env = PortfolioEnv(
                            prices=train_prices,
                            risk_state=_global_risk_state or RiskState(),
                            window_size=train_window,
                        )
                        obs, _ = env.reset()
                        action, _ = _ppo_model.predict(obs, deterministic=True)
                        all_weights = env._softmax(action)
                        # 요청 티커에 해당하는 인덱스만 추출 → 재정규화
                        idx = [_ppo_tickers.index(t) for t in tickers]
                        sub = all_weights[idx]
                        weights = (sub / sub.sum()).astype(float)
                        use_ppo = True
                        logger.info(f"PPO 예측 성공: 학습={len(_ppo_tickers)}종목 → 요청={n}종목 추출")
                except Exception as exc:
                    logger.warning(f"PPO 예측 실패 → MVO 폴백: {exc}")
            else:
                outside = req_set - train_set
                logger.info(f"PPO 폴백: 요청 티커 {outside}가 학습 범위 밖 → MVO 사용")

        if not use_ppo:
            mvo = MVO(MVOConfig(weight_min=_mvo_weight_min(n, req.risk_level)))
            mvo.fit(prices)
            weights = mvo.get_weights().astype(float)

        # ── risk_level 캡 적용 ────────────────────────────────────────────────
        weights = _apply_risk_cap(weights, req.risk_level)

        # ── 성과 지표 계산 ─────────────────────────────────────────────────────
        rets = log_returns(prices)
        port_rets = (rets.values * weights).sum(axis=1)
        pv = np.cumprod(1 + port_rets)
        pv = np.insert(pv, 0, 1.0)
        metrics = compute_metrics(daily_returns=port_rets.tolist(), portfolio_values=pv.tolist())

        # ── risk_tags: 전역 리스크 상태 → 백엔드 스키마 형식 변환 ──────────────
        def _level_to_severity(level: float) -> str:
            if level >= 0.7:
                return "high"
            elif level >= 0.4:
                return "moderate"
            return "low"

        risk_tags = []
        if _global_risk_state is not None:
            for tag_name, tag_level in _global_risk_state._levels.items():
                if tag_level > 0.2:
                    risk_tags.append({
                        "asset": "market",
                        "type": tag_name,
                        "severity": _level_to_severity(tag_level),
                        "source": "리서치 에이전트 분석",
                    })

        return {
            "weights": {t: round(float(w), 6) for t, w in zip(tickers, weights)},
            "metrics": {
                "expected_return": _safe_float(metrics.cagr),
                "sharpe_ratio": _safe_float(metrics.sharpe),
                "max_drawdown": -abs(_safe_float(metrics.max_drawdown)),
                "volatility": _safe_float(metrics.volatility),
            },
            "risk_tags": risk_tags,
        }

    try:
        result = await _run(_compute, timeout=TIMEOUT_OPTIMIZE)
    except Exception as exc:
        return _error(400, "최적화 실패", str(exc))

    if result is None:
        return _error(504, "처리 시간 초과", "4초 이내에 포트폴리오 최적화를 완료하지 못했습니다.")

    return {"status": "success", **result}


# ─── POST /ai/shap ────────────────────────────────────────────────────────────
@app.post("/ai/shap")
async def shap_explain(req: ShapRequest):
    if req.target_asset not in req.tickers:
        return _error(404, f"target_asset '{req.target_asset}'이(가) tickers에 없습니다.",
                      f"요청한 tickers: {req.tickers}")

    # 현재 PPO 모델 로컬 참조 (스레드 클로저 캡처용)
    _local_ppo = _ppo_model

    def _compute():
        import shap as shap_lib
        from ..data.preprocessors import log_returns
        from ..envs.portfolio_env import PortfolioEnv
        from ..envs.risk_state import RiskState
        from ..backtest.mvo import MVO, MVOConfig

        # 최소 6개월 이력 확보
        end_dt = pd.Timestamp(req.date)
        start_dt = (end_dt - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
        end_str = end_dt.strftime("%Y-%m-%d")
        prices = _get_or_fetch_prices(req.tickers, start_dt, end_str)

        if prices.empty or len(prices) < 40:
            raise ValueError("SHAP 계산을 위한 데이터가 부족합니다 (최소 40 거래일).")

        tickers = list(prices.columns)
        n = len(tickers)
        target_idx = tickers.index(req.target_asset)
        window = min(20, len(prices) // 3)

        env = PortfolioEnv(prices=prices, risk_state=_global_risk_state or RiskState(), window_size=window)

        # ── 배경 관측값 수집 (최대 20개) ─────────────────────────────────────
        obs_list: list[np.ndarray] = []
        obs, _ = env.reset()
        obs_list.append(obs.copy())
        for _ in range(min(19, len(env.valid_dates) - 1)):
            action = np.zeros(n, dtype=np.float32)
            obs, _, done, trunc, _ = env.step(action)
            obs_list.append(obs.copy())
            if done or trunc:
                break

        background = np.array(obs_list, dtype=float)
        target_obs = obs_list[-1].astype(float)

        # ── predict_fn 정의 (PPO 관측 공간 호환성 확인) ──────────────────────
        expected_obs = n * 11 + 5 + n
        ppo_ok = (
            _local_ppo is not None
            and _local_ppo.observation_space.shape[0] == expected_obs
        )
        logger.info(f"[SHAP] n={n}, expected_obs={expected_obs}, ppo_ok={ppo_ok}")

        if ppo_ok:
            def predict_fn(batch: np.ndarray) -> np.ndarray:
                return np.array([_local_ppo.predict(row, deterministic=True)[0] for row in batch])
            action, _ = _local_ppo.predict(target_obs, deterministic=True)
            final_weight = float(env._softmax(action.astype(np.float32))[target_idx])
        else:
            # MVO 기반 예측 대리 함수 — 롤링 윈도우로 다양한 가중치 생성 후 Ridge 회귀
            from sklearn.linear_model import Ridge

            _mvo = MVO(MVOConfig())
            _mvo.fit(prices)
            _w = _mvo.get_weights()
            final_weight = float(_w[target_idx])

            # 롤링 윈도우마다 MVO 재최적화 → (obs, weight) 쌍 수집
            step_size = max(1, len(prices) // 40)
            win = max(30, len(prices) // 4)
            scenario_obs, scenario_weights = [], []
            env2 = PortfolioEnv(prices=prices, risk_state=_global_risk_state or RiskState(), window_size=window)
            obs2, _ = env2.reset()
            for step_i in range(min(len(obs_list), len(env2.valid_dates))):
                end_i = min(len(prices), win + step_i * step_size)
                start_i = max(0, end_i - win)
                try:
                    _mvo_tmp = MVO(MVOConfig())
                    _mvo_tmp.fit(prices.iloc[start_i:end_i])
                    scenario_obs.append(obs_list[min(step_i, len(obs_list) - 1)])
                    scenario_weights.append(_mvo_tmp.get_weights())
                except Exception:
                    pass

            if len(scenario_obs) >= 3:
                reg = Ridge(alpha=1.0).fit(np.array(scenario_obs), np.array(scenario_weights))
                def predict_fn(batch: np.ndarray) -> np.ndarray:
                    return reg.predict(batch)
            else:
                def predict_fn(batch: np.ndarray) -> np.ndarray:
                    return np.tile(_w, (len(batch), 1))

        # ── KernelExplainer (nsamples=50 빠른 근사) ───────────────────────────
        bg_sample = shap_lib.sample(background, min(10, len(background)))
        explainer = shap_lib.KernelExplainer(predict_fn, bg_sample)
        sv = explainer.shap_values(target_obs.reshape(1, -1), nsamples=50, silent=True)

        # sv → (n_outputs, n_samples, n_features) 정규화
        n_input_features = target_obs.reshape(1, -1).shape[1]
        if isinstance(sv, list):
            sv_arr = np.stack([np.asarray(s, dtype=float) for s in sv], axis=0)
            # (n_outputs, n_samples, n_features)
        else:
            sv_arr = np.asarray(sv, dtype=float)
        if sv_arr.ndim == 2:
            if sv_arr.shape[-1] == n_input_features:
                sv_arr = sv_arr[np.newaxis]         # (n_samples, n_features) → (1, n_samples, n_features)
            else:
                sv_arr = sv_arr[:, np.newaxis, :]   # (n_outputs, n_features) → (n_outputs, 1, n_features)
        elif sv_arr.ndim == 3:
            if sv_arr.shape[-1] == n_input_features:
                # (n_outputs, n_samples, n_features) — 이미 올바른 형태
                pass
            elif sv_arr.shape[1] == n_input_features:
                # (n_samples, n_features, n_outputs) → (n_outputs, n_samples, n_features)
                sv_arr = np.moveaxis(sv_arr, -1, 0)
        idx = min(target_idx, sv_arr.shape[0] - 1)
        sv_target = sv_arr[idx, 0]  # (n_features,)

        # base value 추출
        ev = explainer.expected_value
        if hasattr(ev, "__len__"):
            base_val = float(np.mean(ev))
        else:
            base_val = float(ev)

        # ── 피처명 생성 ────────────────────────────────────────────────────────
        feature_names = _build_feature_names(tickers)
        n_feat = min(len(feature_names), len(sv_target))
        feature_names = feature_names[:n_feat]
        sv_target = sv_target[:n_feat]

        # 상위 10개 SHAP 값
        top_k = min(10, n_feat)
        top_idx = np.argsort(np.abs(sv_target))[-top_k:]
        shap_dict = {feature_names[int(i)]: round(float(sv_target[i]), 6) for i in top_idx}

        # ── Summary Plot ────────────────────────────────────────────────────────
        mean_abs = np.mean(np.abs(sv_arr[:, 0, :n_feat]), axis=0)
        n_show = min(15, n_feat)
        top_sum = np.argsort(mean_abs)[-n_show:]
        sum_names = [feature_names[int(i)] for i in top_sum]
        sum_vals = mean_abs[top_sum]

        fig_sum, ax_sum = plt.subplots(figsize=(8, max(4, n_show * 0.35)))
        ax_sum.barh(sum_names, sum_vals, color="#4c72b0")
        ax_sum.set_xlabel("Mean |SHAP Value|")
        ax_sum.set_title(f"SHAP Summary — {req.target_asset}")
        ax_sum.grid(axis="x", alpha=0.3)
        fig_sum.tight_layout()
        summary_b64 = _fig_to_b64(fig_sum)

        # ── Force Plot ──────────────────────────────────────────────────────────
        n_force = min(15, n_feat)
        fi = np.argsort(np.abs(sv_target))[-n_force:]
        force_names = [feature_names[int(i)] for i in fi]
        force_vals = sv_target[fi]
        order = np.argsort(force_vals)
        force_vals = force_vals[order]
        force_names = [force_names[i] for i in order]
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in force_vals]

        fig_force, ax_force = plt.subplots(figsize=(8, max(4, n_force * 0.4)))
        ax_force.barh(force_names, force_vals, color=colors)
        ax_force.axvline(0, color="black", linewidth=0.8)
        ax_force.set_xlabel("SHAP Value")
        ax_force.set_title(f"SHAP Force — {req.target_asset} (base={base_val:.4f})")
        ax_force.grid(axis="x", alpha=0.3)
        fig_force.tight_layout()
        force_b64 = _fig_to_b64(fig_force)

        # ── explanation: 상위 SHAP 피처 기반 한국어 자연어 설명 생성 ────────────
        if shap_dict:
            top_feat = max(shap_dict, key=lambda k: abs(shap_dict[k]))
            top_val = shap_dict[top_feat]
            explanation = (
                f"{req.target_asset}의 비중이 {round(final_weight * 100, 1)}%로 결정된 "
                f"주요 원인은 {top_feat}({top_val:+.3f})입니다."
            )
        else:
            explanation = f"{req.target_asset}의 비중은 {round(final_weight * 100, 1)}%입니다."

        return {
            "target_asset": req.target_asset,
            "final_weight": round(final_weight, 6),
            "shap_values": shap_dict,
            "explanation": explanation,
            "force_plot_base64": force_b64,
            "summary_plot_base64": summary_b64,
        }

    try:
        result = await _run(_compute, timeout=TIMEOUT_SHAP)
    except Exception as exc:
        return _error(400, "SHAP 계산 실패", str(exc))

    if result is None:
        return _error(504, "SHAP 처리 시간 초과", "SHAP 계산이 10초를 초과했습니다.")

    return {"status": "success", **result}


# ─── POST /ai/research ────────────────────────────────────────────────────────
@app.post("/ai/research")
async def research(req: ResearchRequest):
    tickers_str = ", ".join(req.tickers)
    query = f"{tickers_str} 주요 리스크와 비중 조정 근거를 요약해줘."

    def _compute():
        agent = _get_research_agent()
        agent.config.n_results = req.max_results
        report = agent.run(query=query, ticker=req.tickers[0] if len(req.tickers) == 1 else None)
        return report

    try:
        report = await _run(_compute, timeout=TIMEOUT_RESEARCH)
    except ResearchAgentUnavailable as exc:
        return _error(503, "리서치 준비 실패", str(exc))
    except Exception as exc:
        return _error(400, "리서치 실패", str(exc))

    if report is None:
        return _error(504, "리서치 처리 시간 초과", f"{TIMEOUT_RESEARCH}초 이내에 완료하지 못했습니다.")

    # ── 전역 RiskState 업데이트 ────────────────────────────────────────────────
    if _global_risk_state is not None and report.risk_tags:
        _global_risk_state.update(report.risk_tags)

    # ── RiskTag → risk_events 변환 ─────────────────────────────────────────
    risk_events = []
    for tag in report.risk_tags:
        level = _safe_float(getattr(tag, "level", 0.5))
        severity = "high" if level >= 0.7 else "moderate" if level >= 0.4 else "low"
        risk_events.append({
            "type": getattr(tag, "name", str(tag)),
            "description": getattr(tag, "source", ""),
            "severity": severity,
            "detected_at": _utc_now()[:10],
        })

    # ── Citation → sources 변환 ────────────────────────────────────────────
    sources = []
    for c in report.citations:
        if hasattr(c, "title"):
            sources.append({
                "title": c.title,
                "url": c.url,
                "published_at": c.published,
                "relevance_score": round(_safe_float(c.relevance_score), 4),
            })
        elif isinstance(c, dict):
            sources.append({
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "published_at": c.get("published", ""),
                "relevance_score": round(_safe_float(c.get("relevance_score", 0.0)), 4),
            })

    return {
        "status": "success",
        "tickers": req.tickers,
        "summary": report.answer,
        "risk_events": risk_events,
        "sources": sources,
        "reasoning_trace": report.reasoning_trace,
        "self_correction_count": report.correction_count,
    }


# ─── POST /ai/backtest ────────────────────────────────────────────────────────
@app.post("/ai/backtest")
async def backtest(req: BacktestRequest):
    if req.strategy not in ("drl", "mvo", "equal_weight"):
        return _error(400, f"유효하지 않은 strategy: '{req.strategy}'",
                      "drl / mvo / equal_weight 중 하나를 선택하세요.")

    def _compute():
        from ..data.preprocessors import log_returns
        from ..backtest.metrics import compute_metrics
        from ..backtest.mvo import MVO, MVOConfig, run_mvo_walk_forward
        from ..backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig

        prices = _get_or_fetch_prices(req.tickers, req.start_date, req.end_date)
        if prices.empty or len(prices) < 60:
            raise ValueError(f"백테스트 데이터 부족: {len(prices)}행 (최소 60 거래일 필요)")

        tickers = list(prices.columns)
        n = len(tickers)

        kospi_ret, sp500_ret = _fetch_benchmark_returns(req.start_date, req.end_date)

        # ── equal_weight ────────────────────────────────────────────────────
        if req.strategy == "equal_weight":
            from ..backtest.mvo import _build_fold_dates
            rets = log_returns(prices)
            weights = np.ones(n) / n
            port_rets = (rets.values * weights).sum(axis=1)
            pv = np.insert(np.cumprod(1 + port_rets), 0, 1.0)
            metrics = compute_metrics(daily_returns=port_rets.tolist(), portfolio_values=pv.tolist())
            cfg = WalkForwardConfig(train_months=24, test_months=6, step_months=6)
            folds_dates = _build_fold_dates(prices, cfg)
            fold_data: list[dict] = []
            for _, (_, _, test_start, test_end) in enumerate(folds_dates):
                test_prices = prices.loc[(prices.index >= test_start) & (prices.index <= test_end)]
                test_rets = test_prices.pct_change().dropna()
                chunk = (test_rets.values * weights).sum(axis=1)
                if len(chunk) < 2:
                    continue
                cpv = np.insert(np.cumprod(1 + chunk), 0, 1.0)
                cm = compute_metrics(daily_returns=chunk.tolist(), portfolio_values=cpv.tolist())
                fold_data.append({
                    "test_start": str(test_start.date() if hasattr(test_start, "date") else test_start),
                    "test_end": str(test_end.date() if hasattr(test_end, "date") else test_end),
                    "total_return": _safe_float(cm.total_return),
                    "sharpe": _safe_float(cm.sharpe),
                })

            cache = {
                "summary": {
                    "mean_cagr": _safe_float(metrics.cagr),
                    "mean_sharpe": _safe_float(metrics.sharpe),
                    "mean_sortino": _safe_float(metrics.sortino),
                    "mean_calmar": _safe_float(metrics.calmar),
                    "mean_max_drawdown": _safe_float(metrics.max_drawdown),
                    "mean_volatility": _safe_float(metrics.volatility),
                    "mean_var_95": _safe_float(metrics.var_95),
                    "mean_cvar_95": _safe_float(metrics.cvar_95),
                    "mean_alpha": _safe_float(metrics.alpha),
                    "mean_beta": _safe_float(metrics.beta),
                    "mean_information_ratio": _safe_float(metrics.information_ratio),
                },
                "folds": fold_data,
            }
            return _build_wf_response("equal_weight", cache, kospi_ret, sp500_ret)

        # ── mvo ─────────────────────────────────────────────────────────────
        if req.strategy == "mvo":
            cfg = WalkForwardConfig(train_months=24, test_months=6, step_months=6, train_timesteps=10_000)
            result = run_mvo_walk_forward(prices, cfg, MVOConfig(), verbose=False)
            cache = _wf_result_to_cache(result)
            _save_json(RESULTS_DIR / "mvo_walk_forward_result.json", cache)
            return _build_wf_response("mvo", cache, kospi_ret, sp500_ret)

        # ── drl ─────────────────────────────────────────────────────────────
        # 캐시된 결과 우선 반환
        cache_path = RESULTS_DIR / "walk_forward_result.json"
        cached = _load_json(cache_path, {})
        if cached and cached.get("folds"):
            return _build_wf_response("drl", cached, kospi_ret, sp500_ret)

        # 캐시 없으면 Walk-Forward 실행 (타임아웃 위험 주의)
        cfg = WalkForwardConfig(train_months=24, test_months=6, step_months=6, train_timesteps=50_000)
        wf = WalkForwardBacktest(prices=prices, config=cfg)
        result = wf.run(verbose=False)
        cache = _wf_result_to_cache(result)
        _save_json(cache_path, cache)
        return _build_wf_response("drl", cache, kospi_ret, sp500_ret)

    try:
        result = await _run(_compute, timeout=TIMEOUT_BACKTEST)
    except Exception as exc:
        return _error(400, "백테스트 실패", str(exc))

    if result is None:
        return _error(504, "백테스트 처리 시간 초과", f"{TIMEOUT_BACKTEST}초 이내에 완료하지 못했습니다.")

    return {"status": "success", **result}
