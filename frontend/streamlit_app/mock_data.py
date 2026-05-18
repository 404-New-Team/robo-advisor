from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd


ASSET_UNIVERSE = [
    {"ticker": "005930", "name": "삼성전자", "sector": "반도체", "region": "KR"},
    {"ticker": "000660", "name": "SK하이닉스", "sector": "반도체", "region": "KR"},
    {"ticker": "035420", "name": "NAVER", "sector": "인터넷", "region": "KR"},
    {"ticker": "035720", "name": "카카오", "sector": "인터넷", "region": "KR"},
    {"ticker": "051910", "name": "LG화학", "sector": "2차전지", "region": "KR"},
    {"ticker": "006400", "name": "삼성SDI", "sector": "2차전지", "region": "KR"},
    {"ticker": "005380", "name": "현대차", "sector": "자동차", "region": "KR"},
    {"ticker": "000270", "name": "기아", "sector": "자동차", "region": "KR"},
    {"ticker": "068270", "name": "셀트리온", "sector": "바이오", "region": "KR"},
    {"ticker": "207940", "name": "삼성바이오로직스", "sector": "바이오", "region": "KR"},
]

PROFILE_LABELS = {
    "low": "안정형",
    "moderate": "위험중립형",
    "high": "공격투자형",
}

STRATEGY_LABELS = {
    "drl": "DRL 로보어드바이저",
    "mvo": "MVO 평균-분산",
    "equal_weight": "동일가중",
}


def get_universe() -> pd.DataFrame:
    return pd.DataFrame(ASSET_UNIVERSE)


def get_default_tickers() -> list[str]:
    return [asset["ticker"] for asset in ASSET_UNIVERSE]


def get_asset_name(ticker: str) -> str:
    match = next((asset for asset in ASSET_UNIVERSE if asset["ticker"] == ticker), None)
    return match["name"] if match else ticker


def get_asset_label(ticker: str) -> str:
    name = get_asset_name(ticker)
    return f"{name} ({ticker})"


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
            "005930": 0.18,
            "000660": 0.10,
            "035420": 0.08,
            "035720": 0.05,
            "051910": 0.10,
            "006400": 0.07,
            "005380": 0.14,
            "000270": 0.11,
            "068270": 0.09,
            "207940": 0.08,
        }
    elif risk_level == "high":
        weights = {
            "005930": 0.16,
            "000660": 0.15,
            "035420": 0.12,
            "035720": 0.09,
            "051910": 0.13,
            "006400": 0.10,
            "005380": 0.09,
            "000270": 0.06,
            "068270": 0.05,
            "207940": 0.05,
        }
    else:
        weights = {
            "005930": 0.17,
            "000660": 0.13,
            "035420": 0.10,
            "035720": 0.07,
            "051910": 0.12,
            "006400": 0.08,
            "005380": 0.12,
            "000270": 0.08,
            "068270": 0.06,
            "207940": 0.07,
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
                "asset": "000660",
                "type": "earnings_surprise",
                "severity": "low",
                "source": "반도체 업황 회복 뉴스",
            },
            {
                "asset": "051910",
                "type": "regulation_change",
                "severity": "moderate",
                "source": "2차전지 보조금 정책 변경",
            },
            {
                "asset": "035420",
                "type": "platform_competition",
                "severity": "moderate",
                "source": "인터넷 플랫폼 광고 성장 둔화",
            },
        ],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_weight_table(weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    universe = get_universe().set_index("ticker")
    for ticker, weight in weights.items():
        asset = universe.loc[ticker].to_dict()
        rows.append(
            {
                "티커": ticker,
                "종목": asset["name"],
                "섹터": asset["sector"],
                "비중": weight,
            }
        )
    return pd.DataFrame(rows).sort_values("비중", ascending=False)


def get_order_preview(weights: dict[str, float], investment_amount: int) -> pd.DataFrame:
    rows = []
    for ticker, weight in weights.items():
        rows.append(
            {
                "종목": get_asset_name(ticker),
                "티커": ticker,
                "목표 비중": weight,
                "매수 금액": int(investment_amount * weight),
            }
        )
    return pd.DataFrame(rows).sort_values("목표 비중", ascending=False)


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


def get_research_response(ticker: str | None = "005930", query: str | None = None, max_results: int = 5) -> dict:
    target = ticker or "포트폴리오"
    now = datetime.now(timezone.utc)
    sources = [
        {
            "title": "반도체 수요 회복과 고대역폭 메모리 투자 확대",
            "url": "https://example.com/news/semiconductor-cycle",
            "published_at": (now - timedelta(hours=3)).isoformat(),
            "relevance_score": 0.94,
        },
        {
            "title": "국내 2차전지 업종, 정책 불확실성에도 장기 수요 유지",
            "url": "https://example.com/news/battery-policy",
            "published_at": (now - timedelta(hours=7)).isoformat(),
            "relevance_score": 0.88,
        },
        {
            "title": "플랫폼 기업 광고 매출 성장률 둔화 분석",
            "url": "https://example.com/news/platform-ad",
            "published_at": (now - timedelta(days=1)).isoformat(),
            "relevance_score": 0.81,
        },
        {
            "title": "자동차 수출 단가 상승과 환율 민감도 점검",
            "url": "https://example.com/news/auto-export",
            "published_at": (now - timedelta(days=1, hours=5)).isoformat(),
            "relevance_score": 0.76,
        },
        {
            "title": "바이오시밀러 승인 일정과 실적 추정치 변화",
            "url": "https://example.com/news/bio-approval",
            "published_at": (now - timedelta(days=2)).isoformat(),
            "relevance_score": 0.72,
        },
    ][:max_results]
    return {
        "status": "success",
        "ticker": ticker,
        "summary": f"{target} 관련 최근 리서치에서 성장 모멘텀은 유지되지만 정책 및 밸류에이션 리스크가 동시에 감지되었습니다.",
        "risk_events": [
            {
                "type": "regulation_change",
                "description": "2차전지 보조금과 공급망 규정 변화가 마진 전망에 영향을 줄 수 있습니다.",
                "severity": "MEDIUM",
                "detected_at": (now - timedelta(hours=6)).isoformat(),
            },
            {
                "type": "earnings_revision",
                "description": "반도체 업종의 이익 추정치가 상향되어 위험 대비 기대수익이 개선되었습니다.",
                "severity": "LOW",
                "detected_at": (now - timedelta(hours=2)).isoformat(),
            },
        ],
        "sources": sources,
        "reasoning_trace": [
            "Plan: 보유 종목별 최신 뉴스와 공시 후보를 검색",
            "Execute: ChromaDB 유사도 상위 문서와 시장 지표를 결합",
            "Self-Correction: 중복 기사와 낮은 관련도 문서를 제거",
            "Verify: 리스크 태그와 포트폴리오 비중 변화 방향을 대조",
            "Answer: 성장 모멘텀 반영, 정책 리스크는 비중 확대 폭 제한",
        ],
        "self_correction_count": 2,
        "timestamp": now.isoformat(),
    }


def get_explain_response(target_asset: str = "005930") -> dict:
    shap_by_asset = {
        "005930": {
            "momentum_7d": 0.037,
            "volatility_30d": -0.014,
            "news_risk_score": -0.009,
            "rsi": 0.022,
            "market_cap_weight": 0.031,
        },
        "000660": {
            "momentum_7d": 0.044,
            "volatility_30d": -0.018,
            "news_risk_score": 0.012,
            "rsi": 0.019,
            "market_cap_weight": 0.017,
        },
        "051910": {
            "momentum_7d": 0.021,
            "volatility_30d": -0.024,
            "news_risk_score": -0.031,
            "rsi": 0.015,
            "market_cap_weight": 0.011,
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
    for ticker in ["005930", "000660", "035420", "051910", "005380", "068270"]:
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
