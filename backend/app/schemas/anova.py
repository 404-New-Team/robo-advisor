from __future__ import annotations
from typing import Any
from pydantic import BaseModel


class TukeyPair(BaseModel):
    group1: str
    group2: str
    mean_diff: float
    q_statistic: float
    p_value_approx: float
    significant: bool


class VerificationOneWay(BaseModel):
    """검증 1 (보상 함수 변형) 및 검증 2 (전략 비교) 공통 구조."""
    f_statistic: float
    p_value: float
    significant: bool
    alpha: float
    eta_squared: float
    eta_squared_interp: str
    group_means: dict[str, float]
    group_stds: dict[str, float]
    group_ns: dict[str, int]
    tukey_results: list[TukeyPair]
    metric_used: str | None = None


class TwoWayEffect(BaseModel):
    f_statistic: float
    p_value: float
    significant: bool
    eta_sq_partial: float


class VerificationTwoWay(BaseModel):
    """검증 3 — 시장 국면 × 전략 Two-way ANOVA."""
    strategy_effect: TwoWayEffect
    regime_effect: TwoWayEffect
    interaction_effect: TwoWayEffect
    strategy_means: dict[str, float]
    regime_means: dict[str, float]
    regime_counts: dict[str, int]
    cell_means: dict[str, Any]
    cell_ns: dict[str, Any]
    anova_table: list[dict[str, Any]]
    alpha: float
    metric_used: str
    n_obs: int


class ANOVAResponse(BaseModel):
    status: str = "success"
    verification1_reward: VerificationOneWay
    verification2_strategy: VerificationOneWay
    verification3_regime: VerificationTwoWay | dict[str, Any]
