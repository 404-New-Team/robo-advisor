from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from src.data.preprocessors import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_bb_position,
    compute_features,
)
from src.data.market_data import fetch_prices


def test_rsi_calculation():
    prices = fetch_prices(["AAPL"], "2024-01-01", "2024-12-31")["AAPL"]
    rsi = calculate_rsi(prices, period=14)
    assert rsi.shape == prices.shape
    assert rsi.notna().sum() > 0
    assert (rsi >= 0).all() or (rsi.isna()).any()
    assert (rsi <= 100).all() or (rsi.isna()).any()
    print("[OK] RSI: {} values generated, range {:.2f}-{:.2f}".format(
        rsi.notna().sum(), rsi.min(), rsi.max()))


def test_macd_calculation():
    prices = fetch_prices(["MSFT"], "2024-01-01", "2024-12-31")["MSFT"]
    macd, signal, histogram = calculate_macd(prices, fast=12, slow=26, signal=9)
    assert macd.shape == prices.shape
    assert signal.shape == prices.shape
    assert histogram.shape == prices.shape
    assert macd.notna().sum() > 0
    print("[OK] MACD: {} values generated".format(macd.notna().sum()))


def test_bollinger_bands_calculation():
    prices = fetch_prices(["GOOGL"], "2024-01-01", "2024-12-31")["GOOGL"]
    upper, middle, lower = calculate_bollinger_bands(prices, period=20, num_std=2.0)
    assert upper.shape == prices.shape
    assert middle.shape == prices.shape
    assert lower.shape == prices.shape
    valid_idx = upper.notna() & lower.notna()
    assert (upper[valid_idx] >= middle[valid_idx]).all()
    assert (lower[valid_idx] <= middle[valid_idx]).all()
    assert (upper[valid_idx] >= lower[valid_idx]).all()
    print("[OK] Bollinger Bands: {} values generated".format(upper.notna().sum()))


def test_bb_position_calculation():
    prices = fetch_prices(["AMZN"], "2024-01-01", "2024-12-31")["AMZN"]
    bb_pos = calculate_bb_position(prices, period=20, num_std=2.0)
    assert bb_pos.shape == prices.shape
    valid_idx = bb_pos.notna()
    assert (bb_pos[valid_idx] >= 0).all()
    assert (bb_pos[valid_idx] <= 1).all()
    print("[OK] BB Position: {} values generated, range 0-1".format(bb_pos.notna().sum()))


def test_compute_features_with_technical_indicators():
    prices = fetch_prices(
        ["AAPL", "MSFT", "GOOGL"],
        "2023-01-01",
        "2024-12-31",
    )
    features = compute_features(prices, window=20)
    n_assets = len(prices.columns)
    expected_n_features = n_assets * 11
    assert features.shape[1] == expected_n_features, \
        "Feature count error: {} != {}".format(features.shape[1], expected_n_features)
    col_names = features.columns.tolist()
    assert any("rsi14" in col for col in col_names)
    assert any("macd" in col for col in col_names)
    assert any("bb_upper" in col for col in col_names)
    print("[OK] Integrated features: {} x {} matrix (11 features per asset)".format(
        features.shape[0], features.shape[1]))


def test_feature_normalization():
    prices = fetch_prices(
        ["AAPL", "MSFT"],
        "2023-06-01",
        "2024-12-31",
    )
    features = compute_features(prices, window=20)

    mean_errors = []
    std_errors = []
    for col in features.columns:
        valid = features[col].dropna()
        mean = valid.mean()
        std = valid.std()
        mean_errors.append(abs(mean))
        std_errors.append(abs(std - 1.0))

    avg_mean_error = sum(mean_errors) / len(mean_errors)
    avg_std_error = sum(std_errors) / len(std_errors)

    assert avg_mean_error < 0.5, "Mean normalization failed: {:.4f}".format(avg_mean_error)
    assert avg_std_error < 0.5, "Std normalization failed: {:.4f}".format(avg_std_error)
    print("[OK] Z-score normalization verified (mean~0, std~1)")


if __name__ == "__main__":
    print("="*60)
    print("Technical Indicators Test")
    print("="*60)
    test_rsi_calculation()
    test_macd_calculation()
    test_bollinger_bands_calculation()
    test_bb_position_calculation()
    test_compute_features_with_technical_indicators()
    test_feature_normalization()
    print("="*60)
    print("All tests passed!")
    print("="*60)
