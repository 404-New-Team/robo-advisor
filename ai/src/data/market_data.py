import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "market"


def _is_krx(ticker: str) -> bool:
    """6자리 숫자면 KRX 종목으로 판별."""
    return ticker.isdigit() and len(ticker) == 6


def _fetch_krx(tickers: list, start: str, end: str) -> pd.DataFrame:
    """pykrx로 국내 ETF/주식 종가 수집. index=Date(datetime), columns=ticker."""
    from pykrx import stock

    start_str = start.replace("-", "")
    end_str = end.replace("-", "")

    frames = {}
    for ticker in tickers:
        try:
            df = stock.get_etf_ohlcv_by_date(start_str, end_str, ticker)
            if df is None or df.empty:
                df = stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
            if df is not None and not df.empty:
                col = "종가" if "종가" in df.columns else df.columns[3]
                frames[ticker] = df[col]
        except Exception:
            pass

    if not frames:
        raise ValueError(f"pykrx: 데이터를 가져올 수 없습니다. 티커={tickers}")

    result = pd.DataFrame(frames)
    result.index = pd.to_datetime(result.index)
    result.index.name = "Date"
    return result


def _fetch_yfinance(tickers: list, start: str, end: str) -> pd.DataFrame:
    """yfinance로 해외 주식/ETF 종가 수집."""
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"yfinance: 빈 데이터. 티커={tickers}, 기간={start}~{end}")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        prices = raw["Close"][tickers] if "Close" in lvl0 else raw.xs("Close", axis=1, level=1)[tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    return prices


def fetch_prices(tickers: list, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    국내(pykrx) + 해외(yfinance) 혼합 수집.
    6자리 숫자 티커 → pykrx, 나머지 → yfinance.
    공통 거래일 교집합(inner join) 후 ffill로 결측치 처리.
    반환: DataFrame, columns=tickers, index=Date
    """
    cache_key = f"{'_'.join(sorted(tickers))}_{start}_{end}.parquet"
    cache_path = CACHE_DIR / cache_key

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    krx_tickers = [t for t in tickers if _is_krx(t)]
    yf_tickers  = [t for t in tickers if not _is_krx(t)]

    parts = []
    if yf_tickers:
        parts.append(_fetch_yfinance(yf_tickers, start, end))
    if krx_tickers:
        parts.append(_fetch_krx(krx_tickers, start, end))

    if len(parts) == 1:
        prices = parts[0]
    else:
        prices = parts[0].join(parts[1], how="inner")

    # 원래 순서 유지, 결측치 처리
    prices = prices[tickers].dropna(how="all").ffill().dropna()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    return prices
