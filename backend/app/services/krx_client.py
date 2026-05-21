from __future__ import annotations

import asyncio
from datetime import date, timedelta

try:
    from pykrx import stock as _krx
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class KRXClientError(Exception):
    pass


def _fetch_price_sync(ticker: str) -> int:
    if not _AVAILABLE:
        raise KRXClientError("pykrx 패키지가 설치되어 있지 않습니다.")

    today = date.today()
    for delta in range(7):  # 주말·공휴일 대비 최대 7거래일 전까지 탐색
        target = (today - timedelta(days=delta)).strftime("%Y%m%d")
        df = _krx.get_market_ohlcv(target, target, ticker)
        if not df.empty:
            return int(df.iloc[-1]["종가"])

    raise KRXClientError(f"{ticker}: 최근 7일 내 거래 데이터를 찾을 수 없습니다.")


async def fetch_current_price(ticker: str) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_price_sync, ticker)


async def fetch_prices(tickers: list[str]) -> dict[str, int]:
    results = await asyncio.gather(
        *[fetch_current_price(t) for t in tickers],
        return_exceptions=True,
    )
    prices: dict[str, int] = {}
    errors: list[str] = []
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            errors.append(f"{ticker}: {result}")
        else:
            prices[ticker] = result

    if errors:
        raise KRXClientError(f"주가 조회 실패 — {', '.join(errors)}")

    return prices
