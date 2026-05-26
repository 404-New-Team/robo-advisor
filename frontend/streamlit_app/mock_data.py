from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from reference_data import (
    PROFILE_LABELS,
    STRATEGY_LABELS,
    get_asset_label,
    get_asset_name,
    get_default_tickers,
    get_order_preview,
    get_universe,
    get_weight_table,
)


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return {}
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0:
        equal_weight = 1 / len(weights)
        return {ticker: equal_weight for ticker in weights}
    return {ticker: max(value, 0.0) / total for ticker, value in weights.items()}


def _base_weights(risk_level: str) -> dict[str, float]:
    if risk_level == "low":
        weights = {
            "SPY": 0.18,
            "QQQ": 0.10,
            "GLD": 0.18,
            "TLT": 0.22,
            "EFA": 0.10,
            "AAPL": 0.05,
            "MSFT": 0.05,
            "069500": 0.05,
            "102110": 0.05,
            "233740": 0.02,
        }
    elif risk_level == "high":
        weights = {
            "SPY": 0.16,
            "QQQ": 0.18,
            "GLD": 0.05,
            "TLT": 0.03,
            "EFA": 0.08,
            "AAPL": 0.14,
            "MSFT": 0.14,
            "069500": 0.07,
            "102110": 0.05,
            "233740": 0.10,
        }
    else:
        weights = {
            "SPY": 0.20,
            "QQQ": 0.14,
            "GLD": 0.12,
            "TLT": 0.12,
            "EFA": 0.10,
            "AAPL": 0.08,
            "MSFT": 0.08,
            "069500": 0.06,
            "102110": 0.06,
            "233740": 0.04,
        }
    return weights


def get_optimize_response(
    risk_level: str = "moderate",
    tickers: list[str] | None = None,
    excluded: list[str] | None = None,
) -> dict:
    selected = tickers or get_default_tickers()
    excluded_set = set(excluded or [])
    weights = {
        ticker: value
        for ticker, value in _base_weights(risk_level).items()
        if ticker in selected and ticker not in excluded_set
    }
    if not weights:
        weights = {ticker: value for ticker, value in _base_weights(risk_level).items() if ticker in selected}
    weights = _normalize(weights)
    risk_multiplier = {"low": 0.78, "moderate": 1.0, "high": 1.22}.get(risk_level, 1.0)
    metrics = {
        "expected_return": 0.102 * risk_multiplier,
        "sharpe_ratio": 1.42 - abs(risk_multiplier - 1.0) * 0.18,
        "max_drawdown": -0.092 * risk_multiplier,
        "volatility": 0.145 * risk_multiplier,
    }
    return {
        "status": "success",
        "weights": weights,
        "metrics": metrics,
        "risk_tags": [
            {
                "asset": "TLT",
                "type": "rate_sensitivity",
                "severity": "moderate",
                "source": "장기 금리 변동성 확대",
            },
            {
                "asset": "233740",
                "type": "leverage_volatility",
                "severity": "moderate",
                "source": "코스닥 레버리지 ETF 변동성 확대",
            },
            {
                "asset": "AAPL",
                "type": "earnings_revision",
                "severity": "low",
                "source": "빅테크 실적 추정치 조정",
            },
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_performance_series() -> pd.DataFrame:
    dates = pd.date_range("2025-06-01", periods=12, freq="ME")
    data = {
        "날짜": dates,
        "DRL 로보어드바이저": [100, 102, 105, 104, 109, 113, 115, 119, 121, 126, 130, 134],
        "MVO 평균-분산": [100, 101, 103, 102, 105, 108, 109, 112, 114, 116, 119, 121],
        "동일가중": [100, 100.5, 102, 101, 103, 105, 106, 108, 109, 111, 113, 114],
        "KOSPI": [100, 99, 101, 100, 102, 103, 102, 104, 106, 107, 108, 109],
    }
    return pd.DataFrame(data)


def get_simulation_paths() -> pd.DataFrame:
    dates = pd.date_range("2026-05-31", periods=12, freq="ME")
    base = [100, 101.2, 102.5, 104.1, 105.4, 106.7, 108.2, 109.5, 111.3, 113.0, 114.8, 116.5]
    rows = []
    for i, date in enumerate(dates):
        rows.append({"날짜": date, "경로": "상위 20%", "자산가치": base[i] * 1.08})
        rows.append({"날짜": date, "경로": "기준", "자산가치": base[i]})
        rows.append({"날짜": date, "경로": "하위 20%", "자산가치": base[i] * 0.93})
    return pd.DataFrame(rows)


def get_research_response(
    tickers: list[str] | None = None,
    max_results: int = 5,
    portfolio_context: dict | None = None,
) -> dict:
    active_tickers = tickers or (portfolio_context or {}).get("active_tickers") or ["SPY"]
    target = active_tickers[0] if len(active_tickers) == 1 else "포트폴리오"
    now = datetime.now(timezone.utc)
    weights = (portfolio_context or {}).get("weights") or {}
    weight_text = ", ".join(f"{ticker}:{weights[ticker] * 100:.1f}%" for ticker in active_tickers if ticker in weights)
    context_text = f" 보유 종목 {', '.join(active_tickers)} 기준입니다." if active_tickers else ""
    weight_suffix = f" 추천 비중은 {weight_text}입니다." if weight_text else ""
    sources = [
        {
            "title": "미국 대형주 ETF, 금리 경로와 실적 전망에 주목",
            "url": "https://example.com/news/us-large-cap-etf",
            "published_at": (now - timedelta(hours=3)).isoformat(),
            "relevance_score": 0.94,
        },
        {
            "title": "나스닥 성장주, AI 투자 확대와 밸류에이션 부담 공존",
            "url": "https://example.com/news/nasdaq-growth-valuation",
            "published_at": (now - timedelta(hours=7)).isoformat(),
            "relevance_score": 0.88,
        },
        {
            "title": "금 가격, 실질금리와 달러 흐름에 민감한 박스권",
            "url": "https://example.com/news/gold-rates-dollar",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "relevance_score": 0.81,
        },
        {
            "title": "장기 국채 ETF, 금리 인하 기대 변화에 변동성 확대",
            "url": "https://example.com/news/long-duration-bonds",
            "published_at": (now - timedelta(days=1, hours=5)).isoformat(),
            "relevance_score": 0.76,
        },
        {
            "title": "국내 대표지수 ETF와 코스닥 레버리지 상품 수급 점검",
            "url": "https://example.com/news/korea-etf-flows",
            "published_at": (now - timedelta(days=2)).isoformat(),
            "relevance_score": 0.72,
        },
    ][:max_results]
    return {
        "status": "success",
        "tickers": active_tickers,
        "summary": f"{target} 관련 최근 리서치에서 성장 모멘텀은 유지되지만 정책 및 밸류에이션 리스크가 동시에 감지되었습니다.{context_text}{weight_suffix}",
        "risk_events": [
            {
                "type": "rate_sensitivity",
                "description": "장기 금리 변화가 TLT와 성장주 밸류에이션에 동시에 영향을 줄 수 있습니다.",
                "severity": "moderate",
                "detected_at": (now - timedelta(hours=6)).isoformat(),
            },
            {
                "type": "earnings_revision",
                "description": "AAPL과 MSFT의 실적 기대가 QQQ와 SPY의 위험 대비 기대수익을 지지합니다.",
                "severity": "low",
                "detected_at": (now - timedelta(hours=2)).isoformat(),
            },
        ],
        "sources": sources,
        "reasoning_trace": [
            "Plan: 보유 종목별 최신 뉴스와 공시 후보를 검색",
            "Execute: ChromaDB 유사도 상위 문서와 시장 지표를 결합",
            "Self-Correction: 중복 기사와 낮은 관련도 문서를 제거",
            "Verify: 리스크 태그와 포트폴리오 비중 변화 방향을 대조",
            "Answer: 성장주 모멘텀 반영, 금리와 레버리지 리스크는 비중 확대 폭 제한",
        ],
        "self_correction_count": 2,
        "timestamp": now.isoformat(),
    }


def get_explain_response(target_asset: str = "SPY") -> dict:
    shap_by_asset = {
        "SPY": {
            "momentum_7d": 0.037,
            "volatility_30d": -0.014,
            "news_risk_score": -0.009,
            "rsi": 0.022,
            "market_cap_weight": 0.031,
        },
        "QQQ": {
            "momentum_7d": 0.044,
            "volatility_30d": -0.018,
            "news_risk_score": 0.012,
            "rsi": 0.019,
            "market_cap_weight": 0.017,
        },
        "TLT": {
            "momentum_7d": 0.012,
            "volatility_30d": -0.019,
            "news_risk_score": -0.017,
            "rsi": 0.010,
            "market_cap_weight": 0.008,
        },
        "233740": {
            "momentum_7d": 0.028,
            "volatility_30d": -0.041,
            "news_risk_score": -0.018,
            "rsi": 0.016,
            "market_cap_weight": 0.004,
        },
    }
    shap_values = shap_by_asset.get(
        target_asset,
        {
            "momentum_7d": 0.018,
            "volatility_30d": -0.012,
            "news_risk_score": -0.006,
            "rsi": 0.014,
            "market_cap_weight": 0.010,
        },
    )
    final_weight = max(0.04, min(0.22, 0.10 + sum(shap_values.values())))
    return {
        "status": "success",
        "target_asset": target_asset,
        "final_weight": final_weight,
        "shap_values": shap_values,
        "explanation": f"{get_asset_name(target_asset)} 비중은 모멘텀과 시장대표성 기여가 높았고, 변동성과 뉴스 리스크가 확대 폭을 제한했습니다.",
        "force_plot_url": "/static/shap/mock_force.png",
        "summary_plot_url": "/static/shap/mock_summary.png",
    }


def get_shap_summary() -> pd.DataFrame:
    rows = []
    for ticker in ["SPY", "QQQ", "GLD", "TLT", "EFA", "AAPL", "MSFT", "069500", "102110", "233740"]:
        shap_values = get_explain_response(ticker)["shap_values"]
        for feature, value in shap_values.items():
            rows.append(
                {
                    "종목": get_asset_name(ticker),
                    "티커": ticker,
                    "피처": feature,
                    "기여도": value,
                }
            )
    return pd.DataFrame(rows)


def get_backtest_response(strategy: str = "drl") -> dict:
    metrics_by_strategy = {
        "drl": {
            "total_return": 0.342,
            "sharpe_ratio": 1.42,
            "sortino_ratio": 1.91,
            "calmar_ratio": 1.68,
            "max_drawdown": -0.118,
            "volatility": 0.154,
            "win_rate": 0.61,
        },
        "mvo": {
            "total_return": 0.211,
            "sharpe_ratio": 1.03,
            "sortino_ratio": 1.34,
            "calmar_ratio": 0.91,
            "max_drawdown": -0.142,
            "volatility": 0.162,
            "win_rate": 0.56,
        },
        "equal_weight": {
            "total_return": 0.143,
            "sharpe_ratio": 0.82,
            "sortino_ratio": 1.08,
            "calmar_ratio": 0.64,
            "max_drawdown": -0.158,
            "volatility": 0.171,
            "win_rate": 0.52,
        },
    }
    walk_forward = [
        {"period": "2021", "return": 0.074, "sharpe": 0.88},
        {"period": "2022", "return": -0.031, "sharpe": 0.41},
        {"period": "2023", "return": 0.126, "sharpe": 1.25},
        {"period": "2024", "return": 0.093, "sharpe": 1.18},
        {"period": "2025", "return": 0.071, "sharpe": 1.05},
    ]
    return {
        "status": "success",
        "strategy": strategy,
        "period": {"start": "2021-01-01", "end": "2025-12-31"},
        "metrics": metrics_by_strategy[strategy],
        "benchmark_comparison": {
            "kospi_return": 0.092,
            "sp500_return": 0.284,
            "strategy_alpha": metrics_by_strategy[strategy]["total_return"] - 0.092,
        },
        "walk_forward_results": walk_forward,
    }


def get_strategy_comparison() -> pd.DataFrame:
    rows = []
    for strategy, label in STRATEGY_LABELS.items():
        metrics = get_backtest_response(strategy)["metrics"]
        rows.append(
            {
                "전략": label,
                "누적수익률": metrics["total_return"],
                "Sharpe": metrics["sharpe_ratio"],
                "MDD": metrics["max_drawdown"],
                "승률": metrics["win_rate"],
            }
        )
    return pd.DataFrame(rows)


def get_anova_result() -> dict:
    return {
        "p_value": 0.018,
        "eta_squared": 0.27,
        "f_statistic": 5.84,
        "conclusion": "DRL 전략의 성과 차이가 MVO 및 동일가중 대비 통계적으로 유의합니다.",
    }


def get_regime_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"시장 국면": "상승장", "DRL": 0.182, "MVO": 0.139, "동일가중": 0.121, "우위 전략": "DRL"},
            {"시장 국면": "횡보장", "DRL": 0.061, "MVO": 0.038, "동일가중": 0.026, "우위 전략": "DRL"},
            {"시장 국면": "하락장", "DRL": -0.042, "MVO": -0.077, "동일가중": -0.091, "우위 전략": "DRL"},
        ]
    )
