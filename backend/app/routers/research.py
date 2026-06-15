from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.auth import get_optional_user
from app.database import get_db
from app.models.portfolio import ResearchResult
from app.models.user import User
from app.schemas.research import ResearchRequest, ResearchResponse, RiskEvent, NewsSource
from app.services.ai_client import AIServiceError, call_research

router = APIRouter(tags=["Research"])


@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="티커 목록에 대한 에이전틱 RAG 리서치 결과 반환",
    responses={
        504: {"description": "5초 초과 타임아웃"},
    },
)
async def research_query(
    request: ResearchRequest,
    db: Session = Depends(get_db),
    current_user: User | None = Depends(get_optional_user),
):
    payload = request.model_dump()
    try:
        ai_result = await call_research(payload)
    except AIServiceError as e:
        raise HTTPException(status_code=e.status_code, detail={"message": e.message, "detail": e.detail})

    if ai_result.get("status") != "success":
        raise HTTPException(status_code=502, detail={"message": "AI 서비스 오류", "detail": ai_result.get("message")})

    risk_events = [RiskEvent(**re) for re in ai_result.get("risk_events", [])]
    sources = [NewsSource(**s) for s in ai_result.get("sources", [])]

    record = ResearchResult(
        user_id=current_user.id if current_user else None,
        tickers=request.tickers,
        summary=ai_result.get("summary", ""),
        risk_events=[re.model_dump() for re in risk_events],
        sources=[s.model_dump() for s in sources],
        reasoning_trace=ai_result.get("reasoning_trace", []),
        self_correction_count=ai_result.get("self_correction_count", 0),
    )
    db.add(record)
    db.commit()

    return ResearchResponse(
        tickers=ai_result.get("tickers", request.tickers),
        summary=ai_result.get("summary", ""),
        risk_events=risk_events,
        sources=sources,
        reasoning_trace=ai_result.get("reasoning_trace", []),
        self_correction_count=ai_result.get("self_correction_count", 0),
    )