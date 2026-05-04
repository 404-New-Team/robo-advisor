import pandas as pd
import yfinance as yf
from pathlib import Path

CACHE_DIR = Path(__file__).parent.parent / ".cache" / "market"


def fetch_prices(tickers: list, start: str, end: str, use_cache: bool = True) -> pd.DataFrame:
    """
    yfinance로 주가 데이터를 다운로드하고 parquet 캐싱.
    반환: Adj Close 기준 DataFrame, columns=tickers, index=Date
    """
    cache_key = f"{'_'.join(sorted(tickers))}_{start}_{end}.parquet"
    cache_path = CACHE_DIR / cache_key

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            f"yfinance가 빈 데이터를 반환했습니다. 티커 {tickers}, 기간 {start}~{end}을 확인하세요."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance 버전에 따라 레벨 순서가 (Price, Ticker) 또는 (Ticker, Price)로 다름
        lvl0 = raw.columns.get_level_values(0).unique().tolist()
        if "Close" in lvl0:
            prices = raw["Close"][tickers]
        else:
            prices = raw.xs("Close", axis=1, level=1)[tickers]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill()

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)

    return prices
