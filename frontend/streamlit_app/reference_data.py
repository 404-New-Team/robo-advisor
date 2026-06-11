from __future__ import annotations

import pandas as pd


ASSET_UNIVERSE = [
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF", "sector": "미국 대형주", "region": "US"},
    {"ticker": "QQQ", "name": "Invesco QQQ ETF", "sector": "미국 성장주", "region": "US"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "sector": "금", "region": "US"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "sector": "미국 장기채", "region": "US"},
    {"ticker": "EFA", "name": "iShares MSCI EAFE ETF", "sector": "선진국 주식", "region": "Global"},
    {"ticker": "AAPL", "name": "Apple", "sector": "기술주", "region": "US"},
    {"ticker": "MSFT", "name": "Microsoft", "sector": "기술주", "region": "US"},
    {"ticker": "069500", "name": "KODEX 200", "sector": "국내 주식 ETF", "region": "KR"},
    {"ticker": "102110", "name": "TIGER 200", "sector": "국내 주식 ETF", "region": "KR"},
    {"ticker": "233740", "name": "KODEX 코스닥150 레버리지", "sector": "국내 레버리지 ETF", "region": "KR"},
]

PROFILE_LABELS = {
    "low": "안정형",
    "moderate": "보통형",
    "high": "적극형",
}

STRATEGY_LABELS = {
    "drl": "AI 추천",
    "mvo": "수익/위험 균형",
    "equal_weight": "같은 비율",
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
