from __future__ import annotations

from datetime import datetime, timezone
from pydantic import BaseModel, Field, model_validator


class AllocationRequest(BaseModel):
    weights: dict[str, float] = Field(
        ...,
        description="AI 최적화 투자 비중 (티커: 비중, 합계 ≈ 1.0)",
    )
    total_amount: float = Field(..., gt=0, description="총 투자 금액 (원)")

    @model_validator(mode="after")
    def _check_weights_sum(self) -> AllocationRequest:
        total = sum(self.weights.values())
        if not (0.95 <= total <= 1.05):
            raise ValueError(f"weights 합계가 1.0에 근사해야 합니다. 현재: {total:.4f}")
        return self


class AllocationItem(BaseModel):
    ticker: str
    weight: float
    current_price: int       # KRX 현재가 (원)
    target_amount: float     # 목표 투자금 = total_amount * weight
    integer_shares: int      # 정수 매수 수량 (floor)
    fractional_shares: float # 소수점 수량 (참고용)
    actual_amount: float     # 실제 투자금 = integer_shares * current_price
    leftover: float          # 잔여금 = target_amount - actual_amount


class AllocationResponse(BaseModel):
    status: str = "success"
    total_amount: float
    total_invested: float
    total_leftover: float
    items: list[AllocationItem]
    fetched_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
