from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.portfolio import ResearchResult
from app.schemas.research import ResearchRequest, ResearchResponse, RiskEvent, NewsSource
from app.services.ai_client import AIServiceError, call_research

router = APIRouter(tags=["Research"])


@router.post(
    "/research",
    response_model=ResearchResponse,
    summary="투자 질문에 대한 에이전틱 RAG 리서치 결과 반환",
    responses={
        400: {"description": "query가 비어있음"},
        504: {"description": "5초 초과 타임아웃"},
    },
)
async def research_query(
    request: ResearchRequest,
    db: Session = Depends(get_db),
):
    payload = request.model_dump()
    try:
        ai_result = await call_research(payload)
    except AIServiceError as e:
        raise HTTPException(status_code=e.status_code, detail={"message": e.message, "detail": e.detail})

    risk_events = [RiskEvent(**re) for re in ai_result.get("risk_events", [])]
    sources = [NewsSource(**s) for s in ai_result.get("sources", [])]

    record = ResearchResult(
        query=request.query,
        ticker=request.ticker,
        summary=ai_result.get("summary", ""),
        risk_events=[re.model_dump() for re in risk_events],
        sources=[s.model_dump() for s in sources],
        reasoning_trace=ai_result.get("reasoning_trace", []),
        self_correction_count=ai_result.get("self_correction_count", 0),
    )
    db.add(record)
    db.commit()

    return ResearchResponse(
        ticker=ai_result.get("ticker", request.ticker),
        summary=ai_result.get("summary", ""),
        risk_events=risk_events,
        sources=sources,
        reasoning_trace=ai_result.get("reasoning_trace", []),
        self_correction_count=ai_result.get("self_correction_count", 0),
    )