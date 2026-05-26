from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class ResearchRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="리서치 대상 티커 목록")
    max_results: int = Field(default=5, ge=1, le=20, description="참고 뉴스 최대 개수")
    portfolio_context: dict[str, Any] | None = Field(None, description="현재 포트폴리오 구성과 추천 비중")


class RiskEvent(BaseModel):
    type: str
    description: str
    severity: Literal["low", "moderate", "high"]
    detected_at: str

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        return {"medium": "moderate"}.get(normalized, normalized)


class NewsSource(BaseModel):
    title: str
    url: str
    published_at: str
    relevance_score: float = Field(..., ge=0.0, le=1.0)


class ResearchResponse(BaseModel):
    status: str = "success"
    tickers: list[str] = []
    summary: str
    risk_events: list[RiskEvent] = []
    sources: list[NewsSource] = []
    reasoning_trace: list[str] = []
    self_correction_count: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
