from datetime import date, datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field, field_validator


class OptimizeRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="자산 티커 목록 (10개 이상 권장)")
    risk_level: Literal["low", "moderate", "high"] = Field(..., description="투자 성향")
    start_date: str = Field(
        default_factory=lambda: str(date.today().replace(year=date.today().year - 5)),
        description="데이터 조회 시작일 (YYYY-MM-DD)",
    )
    end_date: str = Field(
        default_factory=lambda: str(date.today()),
        description="데이터 조회 종료일 (YYYY-MM-DD)",
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("tickers는 최소 1개 이상이어야 합니다.")
        return v


class PortfolioMetrics(BaseModel):
    expected_return: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float


class RiskTag(BaseModel):
    asset: str
    type: str
    severity: Literal["low", "moderate", "high"]
    source: str


class OptimizeResponse(BaseModel):
    status: str = "success"
    weights: dict[str, float]
    metrics: PortfolioMetrics
    risk_tags: list[RiskTag] = []
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
