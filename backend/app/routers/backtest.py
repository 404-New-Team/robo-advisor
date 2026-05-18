from datetime import date
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.auth import get_optional_user
from app.database import get_db
from app.models.portfolio import BacktestResult
from app.models.user import User
from app.schemas.backtest import BacktestResponse, BacktestMetrics, BenchmarkComparison, WalkForwardPeriod, BacktestPeriod
from app.services.ai_client import AIServiceError, call_backtest

router = APIRouter(tags=["Backtest"])

FIVE_YEARS_AGO = str(date.today().replace(year=date.today().year - 5))
TODAY = str(date.today())


@router.get(
    "/backtest",
    response_model=BacktestResponse,
    summary="백테스트 성과 지표 반환",
    responses={
        400: {"description": "유효하지 않은 Strategy 값"},
        422: {"description": "날짜 형식 오류"},
    },
)
async def run_backtest(
    tickers: list[str] = Query(..., description="티커 목록 (예: ?tickers=005930&tickers=SPY&tickers=GLD)"),
    strategy: Literal["drl", "mvo", "equal_weight"] = Query(..., description="포트폴리오 전략"),
    start_date: str = Query(default=FIVE_YEARS_AGO, description="백테스트 시작일 (YYYY-MM-DD)"),
    end_date: str = Query(default=TODAY, description="백테스트 종료일 (YYYY-MM-DD)"),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    ticker_list = [t.strip() for t in tickers if t.strip()]
    if not ticker_list:
        raise HTTPException(status_code=400, detail={"message": "tickers가 비어있습니다.", "detail": None})

    payload = {
        "tickers": ticker_list,
        "strategy": strategy,
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        ai_result = await call_backtest(payload)
    except AIServiceError as e:
        raise HTTPException(status_code=e.status_code, detail={"message": e.message, "detail": e.detail})

    if ai_result.get("status") != "success":
        raise HTTPException(status_code=502, detail={"message": "AI 서비스 오류", "detail": ai_result.get("message")})

    metrics_raw = ai_result.get("metrics", {})
    bench_raw = ai_result.get("benchmark_comparison", {})
    wf_raw = ai_result.get("walk_forward_results", [])

    metrics = BacktestMetrics(
        total_return=metrics_raw.get("total_return", 0.0),
        sharpe_ratio=metrics_raw.get("sharpe_ratio", 0.0),
        sortino_ratio=metrics_raw.get("sortino_ratio", 0.0),
        calmar_ratio=metrics_raw.get("calmar_ratio", 0.0),
        max_drawdown=metrics_raw.get("max_drawdown", 0.0),
        volatility=metrics_raw.get("volatility", 0.0),
        win_rate=metrics_raw.get("win_rate", 0.0),
    )
    benchmark = BenchmarkComparison(
        kospi_return=bench_raw.get("kospi_return", 0.0),
        sp500_return=bench_raw.get("sp500_return", 0.0),
        strategy_alpha=bench_raw.get("strategy_alpha", 0.0),
    )
    walk_forward = [
        WalkForwardPeriod(**{"period": wf["period"], "return": wf.get("return", 0.0), "sharpe": wf.get("sharpe", 0.0)})
        for wf in wf_raw
    ]

    record = BacktestResult(
        user_id=current_user.id if current_user else None,
        tickers=",".join(ticker_list),
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        metrics=metrics_raw,
        benchmark_comparison=bench_raw,
        walk_forward_results=wf_raw,
    )
    db.add(record)
    db.commit()

    return BacktestResponse(
        strategy=strategy,
        period=BacktestPeriod(start=start_date, end=end_date),
        metrics=metrics,
        benchmark_comparison=benchmark,
        walk_forward_results=walk_forward,
    )