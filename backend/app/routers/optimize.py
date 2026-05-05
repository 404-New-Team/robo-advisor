from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.portfolio import PortfolioResult
from app.schemas.optimize import OptimizeRequest, OptimizeResponse, PortfolioMetrics, RiskTag
from app.services.ai_client import AIServiceError, call_optimize

router = APIRouter(tags=["Portfolio"])


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    summary="현재 시장 데이터 기반 최적 포트폴리오 비중 반환",
    responses={
        400: {"description": "티커 목록 오류"},
        504: {"description": "5초 초과 타임아웃"},
    },
)
async def optimize_portfolio(
    request: OptimizeRequest,
    db: Session = Depends(get_db),
):
    payload = request.model_dump()
    try:
        ai_result = await call_optimize(payload)
    except AIServiceError as e:
        raise HTTPException(status_code=e.status_code, detail={"message": e.message, "detail": e.detail})

    weights = ai_result.get("weights", {})
    metrics_raw = ai_result.get("metrics", {})
    risk_tags_raw = ai_result.get("risk_tags", [])

    metrics = PortfolioMetrics(
        expected_return=metrics_raw.get("expected_return", 0.0),
        sharpe_ratio=metrics_raw.get("sharpe_ratio", 0.0),
        max_drawdown=metrics_raw.get("max_drawdown", 0.0),
        volatility=metrics_raw.get("volatility", 0.0),
    )
    risk_tags = [RiskTag(**rt) for rt in risk_tags_raw]

    record = PortfolioResult(
        tickers=",".join(request.tickers),
        risk_level=request.risk_level,
        start_date=request.start_date,
        end_date=request.end_date,
        weights=weights,
        metrics=metrics_raw,
        risk_tags=[rt.model_dump() for rt in risk_tags],
    )
    db.add(record)
    db.commit()

    return OptimizeResponse(weights=weights, metrics=metrics, risk_tags=risk_tags)
