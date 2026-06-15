from datetime import date, datetime
from pydantic import BaseModel, Field


class ExplainRequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1, description="포트폴리오 구성 자산 티커 목록")
    target_asset: str = Field(..., description="SHAP 해석할 특정 자산 티커")
    date: str = Field(
        default_factory=lambda: str(date.today()),
        description="데이터 조회일 (YYYY-MM-DD)",
    )


class ShapValues(BaseModel):
    model_config = {"extra": "allow"}

    momentum_7d: float | None = None
    volatility_30d: float | None = None
    news_risk_score: float | None = None
    rsi: float | None = None
    market_cap_weight: float | None = None


class ExplainResponse(BaseModel):
    status: str = "success"
    target_asset: str
    final_weight: float
    shap_values: dict[str, float]
    explanation: str
    force_plot_url: str
    summary_plot_url: str
