from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.portfolio import ShapResult
from app.schemas.explain import ExplainRequest, ExplainResponse
from app.services.ai_client import AIServiceError, call_shap

router = APIRouter(tags=["SHAP"])


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="특정 의사결정에 대한 SHAP 해석 반환",
    responses={
        404: {"description": "target_asset이 tickers에 없음"},
        422: {"description": "입력값 유효성 오류"},
    },
)
async def explain_decision(
    request: ExplainRequest,
    db: Session = Depends(get_db),
):
    if request.target_asset not in request.tickers:
        raise HTTPException(
            status_code=404,
            detail={"message": f"target_asset '{request.target_asset}'이 tickers 목록에 없습니다.", "detail": None},
        )

    payload = request.model_dump()
    try:
        ai_result = await call_shap(payload)
    except AIServiceError as e:
        raise HTTPException(status_code=e.status_code, detail={"message": e.message, "detail": e.detail})

    target_asset = ai_result.get("target_asset", request.target_asset)
    final_weight = ai_result.get("final_weight", 0.0)
    shap_values = ai_result.get("shap_values", {})
    explanation = ai_result.get("explanation", "")

    date_str = request.date.replace("-", "")
    force_plot_url = ai_result.get("force_plot_url", f"/static/shap/force_{target_asset}_{date_str}.png")
    summary_plot_url = ai_result.get("summary_plot_url", f"/static/shap/summary_{date_str}.png")

    record = ShapResult(
        tickers=",".join(request.tickers),
        target_asset=target_asset,
        analysis_date=request.date,
        final_weight=final_weight,
        shap_values=shap_values,
        explanation=explanation,
    )
    db.add(record)
    db.commit()

    return ExplainResponse(
        target_asset=target_asset,
        final_weight=final_weight,
        shap_values=shap_values,
        explanation=explanation,
        force_plot_url=force_plot_url,
        summary_plot_url=summary_plot_url,
    )