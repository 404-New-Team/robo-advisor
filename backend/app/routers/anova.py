from datetime import date
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.auth import get_optional_user
from app.models.user import User
from app.services.ai_client import AIServiceError, call_anova

router = APIRouter(tags=["ANOVA"])

FIVE_YEARS_AGO = str(date.today().replace(year=date.today().year - 5))
TODAY = str(date.today())


class ANOVARequest(BaseModel):
    tickers: list[str] = Field(..., min_length=1)
    start_date: str = FIVE_YEARS_AGO
    end_date: str = TODAY
    alpha: float = Field(0.05, gt=0, lt=1)
    n_episodes_reward: int = Field(20, ge=5, le=50)


def _build_oneways(raw: dict) -> dict:
    """AI 서비스 응답의 one-way ANOVA 필드를 백엔드 스키마에 맞게 변환."""
    tukey = [
        {
            "group1": t.get("group1", ""),
            "group2": t.get("group2", ""),
            "mean_diff": float(t.get("mean_diff", 0)),
            "q_statistic": float(t.get("q_statistic", 0)),
            "p_value_approx": float(t.get("p_value_approx", 1)),
            "significant": bool(t.get("significant", False)),
        }
        for t in raw.get("tukey_results", [])
    ]
    return {
        "f_statistic": float(raw.get("f_statistic", 0)),
        "p_value": float(raw.get("p_value", 1)),
        "significant": bool(raw.get("significant", False)),
        "alpha": float(raw.get("alpha", 0.05)),
        "eta_squared": float(raw.get("eta_squared", 0)),
        "eta_squared_interp": raw.get("eta_squared_interp", ""),
        "group_means": {k: float(v) for k, v in raw.get("group_means", {}).items()},
        "group_stds": {k: float(v) for k, v in raw.get("group_stds", {}).items()},
        "group_ns": {k: int(v) for k, v in raw.get("group_ns", {}).items()},
        "tukey_results": tukey,
        "metric_used": raw.get("metric_used"),
    }


def _build_twoway(raw: dict) -> dict:
    """AI 서비스 응답의 two-way ANOVA 필드를 백엔드 스키마에 맞게 변환."""
    if "error" in raw:
        return raw

    def _effect(f_key, p_key, sig_key, eta_key) -> dict:
        return {
            "f_statistic": float(raw.get(f_key, 0)),
            "p_value": float(raw.get(p_key, 1)),
            "significant": bool(raw.get(sig_key, False)),
            "eta_sq_partial": float(raw.get(eta_key, 0)),
        }

    return {
        "strategy_effect": _effect("f_strategy", "p_strategy", "sig_strategy", "eta_sq_partial_strategy"),
        "regime_effect": _effect("f_regime", "p_regime", "sig_regime", "eta_sq_partial_regime"),
        "interaction_effect": _effect("f_interaction", "p_interaction", "sig_interaction", "eta_sq_partial_interaction"),
        "strategy_means": {k: float(v) for k, v in raw.get("strategy_means", {}).items()},
        "regime_means": {k: float(v) for k, v in raw.get("regime_means", {}).items()},
        "regime_counts": {k: int(v) for k, v in raw.get("regime_counts", {}).items()},
        "cell_means": raw.get("cell_means", {}),
        "cell_ns": raw.get("cell_ns", {}),
        "anova_table": raw.get("anova_table", []),
        "alpha": float(raw.get("alpha", 0.05)),
        "metric_used": raw.get("metric_used", "fold_cagr"),
        "n_obs": int(raw.get("n_obs", 0)),
    }


@router.post(
    "/anova",
    summary="3종 ANOVA 검증 (보상함수·전략·시장국면)",
)
async def run_anova(
    req: ANOVARequest,
    current_user: User | None = Depends(get_optional_user),
) -> Any:
    payload = req.model_dump()
    try:
        ai_result = await call_anova(payload)
    except AIServiceError as e:
        raise HTTPException(status_code=e.status_code, detail={"message": e.message, "detail": e.detail})

    if ai_result.get("status") != "success":
        raise HTTPException(
            status_code=502,
            detail={"message": "AI 서비스 오류", "detail": ai_result.get("message", "")},
        )

    return {
        "status": "success",
        "verification1_reward": _build_oneways(ai_result.get("verification1_reward", {})),
        "verification2_strategy": _build_oneways(ai_result.get("verification2_strategy", {})),
        "verification3_regime": _build_twoway(ai_result.get("verification3_regime", {})),
    }
