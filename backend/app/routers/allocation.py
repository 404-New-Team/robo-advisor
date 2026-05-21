from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException

from app.auth import get_optional_user
from app.models.user import User
from app.schemas.allocation import AllocationItem, AllocationRequest, AllocationResponse
from app.services.krx_client import KRXClientError, fetch_prices

router = APIRouter(tags=["Allocation"])


@router.post(
    "/allocation",
    response_model=AllocationResponse,
    summary="투자 비중 기반 KRX 현재가 반영 주문 수량 계산",
    description=(
        "AI 최적화 결과(weights)와 총 투자금액을 받아 KRX 현재가를 조회한 뒤, "
        "정수 매수 수량 및 소수점 참고 수량을 계산해 반환합니다."
    ),
    responses={
        502: {"description": "KRX 주가 조회 실패"},
    },
)
async def calculate_allocation(
    request: AllocationRequest,
    current_user: User | None = Depends(get_optional_user),
) -> AllocationResponse:
    tickers = list(request.weights.keys())

    try:
        prices = await fetch_prices(tickers)
    except KRXClientError as e:
        raise HTTPException(
            status_code=502,
            detail={"message": "KRX 주가 조회 실패", "detail": str(e)},
        )

    items: list[AllocationItem] = []
    for ticker, weight in request.weights.items():
        price = prices[ticker]
        target_amount = request.total_amount * weight
        fractional = target_amount / price
        integer_shares = math.floor(fractional)
        actual_amount = float(integer_shares * price)

        items.append(
            AllocationItem(
                ticker=ticker,
                weight=weight,
                current_price=price,
                target_amount=round(target_amount, 2),
                integer_shares=integer_shares,
                fractional_shares=round(fractional, 4),
                actual_amount=actual_amount,
                leftover=round(target_amount - actual_amount, 2),
            )
        )

    total_invested = sum(item.actual_amount for item in items)

    return AllocationResponse(
        total_amount=request.total_amount,
        total_invested=round(total_invested, 2),
        total_leftover=round(request.total_amount - total_invested, 2),
        items=items,
    )
