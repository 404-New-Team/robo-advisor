from datetime import date
from typing import Literal
from pydantic import BaseModel, Field


class BacktestMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float


class BenchmarkComparison(BaseModel):
    kospi_return: float
    sp500_return: float
    strategy_alpha: float


class WalkForwardPeriod(BaseModel):
    model_config = {"populate_by_name": True}

    period: str
    return_: float = Field(..., alias="return")
    sharpe: float


class BacktestPeriod(BaseModel):
    start: str
    end: str


class BacktestResponse(BaseModel):
    status: str = "success"
    strategy: str
    period: BacktestPeriod
    metrics: BacktestMetrics
    benchmark_comparison: BenchmarkComparison
    walk_forward_results: list[WalkForwardPeriod]
