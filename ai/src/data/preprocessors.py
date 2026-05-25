"""
금융 시계열의 비정상성(Non-stationarity) 처리 유틸리티.

가격은 단위근을 가지므로 직접 RL 관측에 사용하면 학습/테스트 분포 불일치 발생.
로그 수익률 변환 + 롤링 Z-score 정규화로 정상성을 확보한다.

기술적 지표: RSI, MACD, Bollinger Bands 포함
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu = df.rolling(window).mean()
    sigma = df.rolling(window).std().replace(0, 1e-8)
    return ((df - mu) / sigma).dropna()


def calculate_rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI (Relative Strength Index) 계산."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD (Moving Average Convergence Divergence) 계산."""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram


def calculate_bollinger_bands(prices: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> tuple:
    """Bollinger Bands 계산."""
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band


def calculate_bb_position(prices: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands 위치 (0~1 정규화)."""
    upper, _, lower = calculate_bollinger_bands(prices, period, num_std)
    bb_position = (prices - lower) / (upper - lower + 1e-8)
    return bb_position.clip(0, 1)


def compute_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    각 자산에 대해 11개 피처 계산:
    기본 (5개): [1d수익률, 5d수익률, 20d수익률, 20d변동성, 모멘텀]
    기술적 지표 (6개): [RSI, MACD, MACD신호, BB상단, BB하단, BB위치]
    모든 피처는 롤링 Z-score 정규화로 정상성 확보.
    반환: shape (T, n_assets * 11), columns = {ticker}_{suffix}
    """
    rets = log_returns(prices)

    feature_map = {
        # 기본 특성
        "ret1d": rets,
        "ret5d": prices.pct_change(5).dropna(),
        "ret20d": prices.pct_change(20).dropna(),
        "vol20d": rets.rolling(window).std().dropna(),
        "mom20d": (prices / prices.shift(window) - 1).dropna(),
        # 기술적 지표
        "rsi14": calculate_rsi(prices, 14),
        "macd": calculate_macd(prices)[0],
        "macd_signal": calculate_macd(prices)[1],
        "bb_upper": calculate_bollinger_bands(prices)[0],
        "bb_lower": calculate_bollinger_bands(prices)[2],
        "bb_position": calculate_bb_position(prices),
    }

    common_idx = feature_map["ret1d"].index
    for df in feature_map.values():
        common_idx = common_idx.intersection(df.index)

    blocks = []
    for suffix, df in feature_map.items():
        normalized = rolling_zscore(df.loc[common_idx], window)
        blocks.append(normalized.rename(columns=lambda c: f"{c}_{suffix}"))

    return pd.concat(blocks, axis=1).dropna()
