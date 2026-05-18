from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="투자 관련 질문")
    ticker: str | None = Field(None, description="특정 종목 지정 시")
    max_results: int = Field(default=5, ge=1, le=20, description="참고 뉴스 최대 개수")


class RiskEvent(BaseModel):
    type: str
    description: str
    severity: Literal["low", "moderate", "high"]
    detected_at: str


class NewsSource(BaseModel):
    title: str
    url: str
    published_at: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ResearchResponse(BaseModel):
    status: str = "success"
    ticker: str | None = None
    summary: str
    risk_events: list[RiskEvent] = []
    sources: list[NewsSource] = []
    reasoning_trace: list[str] = []
    self_correction_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
