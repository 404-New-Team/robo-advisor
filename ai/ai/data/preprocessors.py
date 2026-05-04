"""
금융 시계열의 비정상성(Non-stationarity) 처리 유틸리티.

가격은 단위근을 가지므로 직접 RL 관측에 사용하면 학습/테스트 분포 불일치 발생.
로그 수익률 변환 + 롤링 Z-score 정규화로 정상성을 확보한다.
"""

import numpy as np
import pandas as pd


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    mu = df.rolling(window).mean()
    sigma = df.rolling(window).std().replace(0, 1e-8)
    return ((df - mu) / sigma).dropna()


def compute_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    각 자산에 대해 5개 피처 계산: [1d수익률, 5d수익률, 20d수익률, 20d변동성, 모멘텀]
    모든 피처는 롤링 Z-score 정규화로 정상성 확보.
    반환: shape (T, n_assets * 5), columns = {ticker}_{suffix}
    """
    rets = log_returns(prices)

    feature_map = {
        "ret1d": rets,
        "ret5d": prices.pct_change(5).dropna(),
        "ret20d": prices.pct_change(20).dropna(),
        "vol20d": rets.rolling(window).std().dropna(),
        "mom20d": (prices / prices.shift(window) - 1).dropna(),
    }

    common_idx = feature_map["ret1d"].index
    for df in feature_map.values():
        common_idx = common_idx.intersection(df.index)

    blocks = []
    for suffix, df in feature_map.items():
        normalized = rolling_zscore(df.loc[common_idx], window)
        blocks.append(normalized.rename(columns=lambda c: f"{c}_{suffix}"))

    return pd.concat(blocks, axis=1).dropna()
