from unittest.mock import AsyncMock, patch
from tests.conftest import MOCK_BACKTEST_RESULT


BASE_PARAMS = "tickers=005930,SPY,GLD&strategy=drl&start_date=2021-01-01&end_date=2026-01-01"


def test_backtest_success(client):
    with patch("app.routers.backtest.call_backtest", new_callable=AsyncMock, return_value=MOCK_BACKTEST_RESULT):
        response = client.get(f"/backtest?{BASE_PARAMS}")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["strategy"] == "drl"


def test_backtest_metrics_structure(client):
    with patch("app.routers.backtest.call_backtest", new_callable=AsyncMock, return_value=MOCK_BACKTEST_RESULT):
        response = client.get(f"/backtest?{BASE_PARAMS}")
    metrics = response.json()["metrics"]
    required_keys = ["total_return", "sharpe_ratio", "sortino_ratio", "calmar_ratio", "max_drawdown", "volatility", "win_rate"]
    for key in required_keys:
        assert key in metrics, f"'{key}' 누락"


def test_backtest_invalid_strategy(client):
    response = client.get("/backtest?tickers=SPY&strategy=invalid")
    assert response.status_code == 422


def test_backtest_walk_forward_results(client):
    with patch("app.routers.backtest.call_backtest", new_callable=AsyncMock, return_value=MOCK_BACKTEST_RESULT):
        response = client.get(f"/backtest?{BASE_PARAMS}")
    wf = response.json()["walk_forward_results"]
    assert isinstance(wf, list)
    assert len(wf) >= 1
    assert "period" in wf[0]
    assert "sharpe" in wf[0]


def test_backtest_benchmark_comparison(client):
    with patch("app.routers.backtest.call_backtest", new_callable=AsyncMock, return_value=MOCK_BACKTEST_RESULT):
        response = client.get(f"/backtest?{BASE_PARAMS}")
    bench = response.json()["benchmark_comparison"]
    assert "kospi_return" in bench
    assert "sp500_return" in bench
    assert "strategy_alpha" in bench


def test_backtest_mvo_strategy(client):
    mvo_result = {**MOCK_BACKTEST_RESULT, "strategy": "mvo"}
    with patch("app.routers.backtest.call_backtest", new_callable=AsyncMock, return_value=mvo_result):
        response = client.get("/backtest?tickers=SPY,GLD&strategy=mvo&start_date=2021-01-01&end_date=2026-01-01")
    assert response.status_code == 200
    assert response.json()["strategy"] == "mvo"
