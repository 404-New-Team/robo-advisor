from __future__ import annotations

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


def get_asset_meta(ticker: str) -> dict:
    return next(
        (asset for asset in ASSET_UNIVERSE if asset["ticker"] == ticker),
        {"ticker": ticker, "name": ticker, "sector": "기타", "region": ""},
    )


def get_asset_name(ticker: str) -> str:
    return get_asset_meta(ticker)["name"]


def get_asset_label(ticker: str) -> str:
    name = get_asset_name(ticker)
    return f"{name} ({ticker})"


def get_weight_table(weights: dict[str, float]) -> pd.DataFrame:
    rows = []
    for ticker, weight in weights.items():
        asset = get_asset_meta(ticker)
        rows.append(
            {
                "티커": ticker,
                "종목": asset["name"],
                "섹터": asset["sector"],
                "비중": weight,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["티커", "종목", "섹터", "비중"])
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
    if not rows:
        return pd.DataFrame(columns=["종목", "티커", "목표 비중", "매수 금액"])
    return pd.DataFrame(rows).sort_values("목표 비중", ascending=False)
