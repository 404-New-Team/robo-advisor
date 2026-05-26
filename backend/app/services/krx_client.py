from __future__ import annotations

import asyncio
from datetime import date, timedelta

try:
    from pykrx import stock as _krx
    _PYKRX_AVAILABLE = True
except ImportError:
    _PYKRX_AVAILABLE = False


class KRXClientError(Exception):
    pass


def _is_kr_ticker(ticker: str) -> bool:
    return ticker.isdigit() and len(ticker) == 6


# ── KR (pykrx) ────────────────────────────────────────────────────────────────

def _fetch_kr_price_sync(ticker: str) -> int:
    if not _PYKRX_AVAILABLE:
        raise KRXClientError("pykrx 패키지가 설치되어 있지 않습니다.")
    today = date.today()
    for delta in range(7):
        target = (today - timedelta(days=delta)).strftime("%Y%m%d")
        df = _krx.get_market_ohlcv(target, target, ticker)
        if not df.empty:
            return int(df.iloc[-1]["종가"])
    raise KRXClientError(f"{ticker}: 최근 7일 내 거래 데이터를 찾을 수 없습니다.")


# ── US (yfinance) ─────────────────────────────────────────────────────────────

def _fetch_usd_krw_sync() -> float:
    import yfinance as yf
    hist = yf.Ticker("KRW=X").history(period="2d")
    if hist.empty:
        return 1370.0  # fallback
    return float(hist["Close"].iloc[-1])


def _fetch_us_price_sync(ticker: str, usd_krw: float) -> int:
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period="2d")
    if hist.empty:
        raise KRXClientError(f"{ticker}: yfinance에서 현재가를 가져올 수 없습니다.")
    usd_price = float(hist["Close"].iloc[-1])
    return int(round(usd_price * usd_krw))


# ── Public API ────────────────────────────────────────────────────────────────

async def fetch_prices(tickers: list[str]) -> dict[str, int]:
    kr_tickers = [t for t in tickers if _is_kr_ticker(t)]
    us_tickers = [t for t in tickers if not _is_kr_ticker(t)]

    loop = asyncio.get_event_loop()
    prices: dict[str, int] = {}
    errors: list[str] = []

    if kr_tickers:
        kr_results = await asyncio.gather(
            *[loop.run_in_executor(None, _fetch_kr_price_sync, t) for t in kr_tickers],
            return_exceptions=True,
        )
        for ticker, result in zip(kr_tickers, kr_results):
            if isinstance(result, Exception):
                errors.append(f"{ticker}: {result}")
            else:
                prices[ticker] = result

    if us_tickers:
        usd_krw = await loop.run_in_executor(None, _fetch_usd_krw_sync)
        us_results = await asyncio.gather(
            *[loop.run_in_executor(None, _fetch_us_price_sync, t, usd_krw) for t in us_tickers],
            return_exceptions=True,
        )
        for ticker, result in zip(us_tickers, us_results):
            if isinstance(result, Exception):
                errors.append(f"{ticker}: {result}")
            else:
                prices[ticker] = result

    if errors:
        raise KRXClientError(f"주가 조회 실패 — {', '.join(errors)}")

    return prices
