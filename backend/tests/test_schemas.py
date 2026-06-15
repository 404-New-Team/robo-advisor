import pytest
from pydantic import ValidationError

from app.schemas.optimize import OptimizeRequest
from app.schemas.explain import ExplainRequest
from app.schemas.research import ResearchRequest
from app.schemas.backtest import BacktestMetrics, WalkForwardPeriod


def test_optimize_request_valid():
    req = OptimizeRequest(tickers=["SPY", "GLD"], risk_level="low")
    assert req.risk_level == "low"
    assert len(req.tickers) == 2


def test_optimize_request_invalid_risk_level():
    with pytest.raises(ValidationError):
        OptimizeRequest(tickers=["SPY"], risk_level="extreme")


def test_explain_request_valid():
    req = ExplainRequest(tickers=["SPY", "GLD"], target_asset="SPY", date="2026-01-01")
    assert req.target_asset == "SPY"


def test_explain_request_missing_target():
    with pytest.raises(ValidationError):
        ExplainRequest(tickers=["SPY"])


def test_research_request_empty_query():
    with pytest.raises(ValidationError):
        ResearchRequest(query="")


def test_research_request_max_results_bounds():
    with pytest.raises(ValidationError):
        ResearchRequest(query="test", max_results=0)
    with pytest.raises(ValidationError):
        ResearchRequest(query="test", max_results=21)


def test_backtest_metrics_valid():
    m = BacktestMetrics(
        total_return=0.5,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        calmar_ratio=1.8,
        max_drawdown=-0.15,
        volatility=0.12,
        win_rate=0.6,
    )
    assert m.max_drawdown == -0.15


def test_walk_forward_period_alias():
    wf = WalkForwardPeriod(**{"period": "2021-01~2022-01", "return": 0.11, "sharpe": 1.4})
    assert wf.return_ == 0.11
    assert wf.sharpe == 1.4
