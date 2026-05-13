"""
시장 국면별 성과 비교 — ANOVA 검증 3.

Two-way ANOVA: 전략(DRL / MVO / EqualWeight) × 시장 국면(Bull / Bear / Sideways)

Walk-Forward 각 폴드의 테스트 기간 수익률로 국면을 자동 분류하고,
세 전략의 CAGR을 Two-way ANOVA로 비교한다.

효과 크기: 편부분 η² (partial eta-squared) — 전략 / 국면 / 상호작용 각각 산출.

흐름:
  collect_regime_returns()   — 폴드별 전략 CAGR + 국면 레이블 수집
    → run_twoway_anova()     — Type-II SS Two-way ANOVA (statsmodels)
      → report_twoway_anova() — 콘솔 출력 + JSON 저장
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────
# 시장 국면 분류
# ─────────────────────────────────────────────────────────────

class MarketRegime(str, enum.Enum):
    BULL     = "Bull"
    BEAR     = "Bear"
    SIDEWAYS = "Sideways"


def classify_fold_regime(
    test_prices: pd.DataFrame,
    threshold_bull: float = 0.10,
    threshold_bear: float = -0.10,
) -> MarketRegime:
    """
    테스트 기간 동일가중 CAGR로 시장 국면 분류.

    threshold_bull  : 이 이상이면 Bull  (연환산 기준)
    threshold_bear  : 이 이하이면 Bear
    그 사이           : Sideways
    """
    rets = test_prices.pct_change().dropna()
    avg_daily = rets.mean(axis=1).values
    years = len(avg_daily) / 252
    cagr = float(np.prod(1 + avg_daily) ** (1 / max(years, 1e-8)) - 1)
    if cagr >= threshold_bull:
        return MarketRegime.BULL
    elif cagr <= threshold_bear:
        return MarketRegime.BEAR
    else:
        return MarketRegime.SIDEWAYS


# ─────────────────────────────────────────────────────────────
# 결과 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class TwoWayANOVAResult:
    # ── 주효과: 전략 ─────────────────────────────────────────
    f_strategy: float
    p_strategy: float
    sig_strategy: bool
    eta_sq_partial_strategy: float

    # ── 주효과: 시장 국면 ─────────────────────────────────────
    f_regime: float
    p_regime: float
    sig_regime: bool
    eta_sq_partial_regime: float

    # ── 상호작용 ──────────────────────────────────────────────
    f_interaction: float
    p_interaction: float
    sig_interaction: bool
    eta_sq_partial_interaction: float

    # ── 셀 통계 ───────────────────────────────────────────────
    cell_means: dict      # {(strategy, regime): mean_cagr}
    cell_ns: dict         # {(strategy, regime): n}
    regime_counts: dict   # {regime: n_folds}
    regime_means: dict    # {regime: mean_cagr across strategies}
    strategy_means: dict  # {strategy: mean_cagr across regimes}

    alpha: float
    metric_used: str
    n_obs: int

    # ── 원시 ANOVA 테이블 ─────────────────────────────────────
    anova_table: list     # [{"source": ..., "SS": ..., "df": ..., "MS": ..., "F": ..., "p": ...}]


# ─────────────────────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────────────────────

def collect_regime_returns(
    prices: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    drl_timesteps: int = 30_000,
    window_size: int = 20,
    transaction_cost: float = 0.00015,
    slippage: float = 0.0005,
    max_drawdown_threshold: float = 0.15,
    risk_free_rate: float = 0.02,
    threshold_bull: float = 0.10,
    threshold_bear: float = -0.10,
    verbose: bool = True,
) -> list[dict]:
    """
    Walk-Forward 각 폴드에서 전략별 CAGR + 시장 국면 레이블을 수집한다.

    Returns:
        [{"strategy": str, "regime": str, "cagr": float, "fold": int}, ...]
    """
    from ..backtest.mvo import MVO, MVOConfig
    from ..envs.portfolio_env import PortfolioEnv
    from ..envs.risk_state import RiskState
    from ..agents.ppo_agent import PPOAgent
    from ..backtest.metrics import compute_metrics

    from ..research.strategy_anova import (
        _build_fold_dates, _slice,
        _equal_weight_cagr, _fixed_weight_cagr, _rollout,
    )
    from ..backtest.walk_forward import WalkForwardConfig

    wf_cfg = WalkForwardConfig(
        train_months=train_months,
        test_months=test_months,
        step_months=step_months,
        train_timesteps=drl_timesteps,
        window_size=window_size,
        transaction_cost=transaction_cost,
        slippage=slippage,
        max_drawdown_threshold=max_drawdown_threshold,
        risk_free_rate=risk_free_rate,
    )

    folds_dates = _build_fold_dates(prices, wf_cfg)
    if not folds_dates:
        raise ValueError("유효한 폴드가 없습니다.")

    if verbose:
        print(f"[국면 ANOVA] 총 {len(folds_dates)}개 폴드")

    records: list[dict] = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds_dates):
        train_prices = _slice(prices, train_start, train_end)
        test_prices  = _slice(prices, test_start, test_end)

        if len(train_prices) < wf_cfg.min_train_bars:
            if verbose:
                print(f"  Fold {fold_idx:02d} 건너뜀 (훈련 데이터 부족)")
            continue

        regime = classify_fold_regime(test_prices, threshold_bull, threshold_bear)
        if verbose:
            print(f"\n  Fold {fold_idx:02d} | 테스트: {test_start.date()} ~ {test_end.date()} | 국면: {regime.value}")

        # ── EqualWeight ────────────────────────────────────────
        ew_cagr = _equal_weight_cagr(test_prices, risk_free_rate)
        records.append({"strategy": "EqualWeight", "regime": regime.value,
                         "cagr": ew_cagr, "fold": fold_idx})
        if verbose:
            print(f"    EqualWeight CAGR: {ew_cagr:+.2%}")

        # ── MVO ────────────────────────────────────────────────
        mvo = MVO(MVOConfig(target="max_sharpe", risk_free_rate=risk_free_rate,
                             trading_days=wf_cfg.trading_days_per_year))
        mvo.fit(train_prices)
        mvo_cagr = _fixed_weight_cagr(test_prices, mvo.get_weights(), risk_free_rate)
        records.append({"strategy": "MVO", "regime": regime.value,
                         "cagr": mvo_cagr, "fold": fold_idx})
        if verbose:
            print(f"    MVO        CAGR: {mvo_cagr:+.2%}")

        # ── DRL ────────────────────────────────────────────────
        train_env = PortfolioEnv(
            prices=train_prices,
            risk_state=RiskState(),
            window_size=window_size,
            transaction_cost=transaction_cost,
            slippage=slippage,
            max_drawdown_threshold=max_drawdown_threshold,
        )
        agent = PPOAgent(env=train_env)
        agent.train(
            total_timesteps=drl_timesteps,
            checkpoint_dir=f"checkpoints/regime_anova_fold_{fold_idx:02d}/",
        )

        test_env = PortfolioEnv(
            prices=test_prices,
            risk_state=RiskState(),
            window_size=window_size,
            transaction_cost=transaction_cost,
            slippage=slippage,
            max_drawdown_threshold=max_drawdown_threshold,
        )
        pv, dr = _rollout(agent, test_env)
        perf = compute_metrics(
            daily_returns=dr,
            portfolio_values=pv,
            n_bars=len(test_prices),
            trading_days=wf_cfg.trading_days_per_year,
            risk_free_rate=risk_free_rate,
        )
        records.append({"strategy": "DRL", "regime": regime.value,
                         "cagr": perf.cagr, "fold": fold_idx})
        if verbose:
            print(f"    DRL        CAGR: {perf.cagr:+.2%}")

    if verbose:
        df = pd.DataFrame(records)
        print(f"\n수집 완료: {len(records)}개 관측치")
        print(df.groupby(["strategy", "regime"])["cagr"].count().to_string())

    return records


# ─────────────────────────────────────────────────────────────
# Two-way ANOVA
# ─────────────────────────────────────────────────────────────

def run_twoway_anova(
    records: list[dict],
    alpha: float = 0.05,
    metric_name: str = "fold_cagr",
) -> TwoWayANOVAResult:
    """
    Two-way ANOVA (Type II SS): 전략 × 시장 국면.

    statsmodels가 설치된 경우 OLS 기반 Type-II SS를 사용하고,
    없으면 scipy를 이용한 수동 계산으로 대체한다.
    """
    df = pd.DataFrame(records)
    df["cagr"] = df["cagr"].astype(float)

    strategies = sorted(df["strategy"].unique())
    regimes    = sorted(df["regime"].unique())

    # ── 셀 통계 ────────────────────────────────────────────────
    cell_means: dict = {}
    cell_ns: dict    = {}
    for s in strategies:
        for r in regimes:
            mask = (df["strategy"] == s) & (df["regime"] == r)
            sub  = df.loc[mask, "cagr"].values
            key  = (s, r)
            cell_means[key] = float(np.mean(sub)) if len(sub) > 0 else float("nan")
            cell_ns[key]    = int(len(sub))

    regime_means   = {r: float(df.loc[df["regime"] == r, "cagr"].mean()) for r in regimes}
    strategy_means = {s: float(df.loc[df["strategy"] == s, "cagr"].mean()) for s in strategies}
    regime_counts  = {r: int((df["regime"] == r).sum() // len(strategies)) for r in regimes}

    # ── ANOVA 계산 ──────────────────────────────────────────────
    try:
        table, ss_error = _statsmodels_anova(df)
    except Exception:
        table, ss_error = _manual_anova(df, strategies, regimes)

    def _row(name):
        for row in table:
            if row["source"] == name:
                return row
        return {"SS": 0.0, "df": 1, "MS": 0.0, "F": 0.0, "p": 1.0}

    row_s  = _row("C(strategy)")
    row_r  = _row("C(regime)")
    row_i  = _row("C(strategy):C(regime)")

    ss_s  = row_s["SS"]
    ss_r  = row_r["SS"]
    ss_i  = row_i["SS"]

    # 편부분 η² = SS_effect / (SS_effect + SS_error)
    def _partial_eta(ss_eff):
        denom = ss_eff + ss_error
        return float(ss_eff / denom) if denom > 1e-12 else 0.0

    return TwoWayANOVAResult(
        f_strategy=round(float(row_s["F"]), 4),
        p_strategy=round(float(row_s["p"]), 6),
        sig_strategy=bool(row_s["p"] < alpha),
        eta_sq_partial_strategy=round(_partial_eta(ss_s), 4),

        f_regime=round(float(row_r["F"]), 4),
        p_regime=round(float(row_r["p"]), 6),
        sig_regime=bool(row_r["p"] < alpha),
        eta_sq_partial_regime=round(_partial_eta(ss_r), 4),

        f_interaction=round(float(row_i["F"]), 4),
        p_interaction=round(float(row_i["p"]), 6),
        sig_interaction=bool(row_i["p"] < alpha),
        eta_sq_partial_interaction=round(_partial_eta(ss_i), 4),

        cell_means={str(k): v for k, v in cell_means.items()},
        cell_ns={str(k): v for k, v in cell_ns.items()},
        regime_counts=regime_counts,
        regime_means=regime_means,
        strategy_means=strategy_means,
        alpha=alpha,
        metric_used=metric_name,
        n_obs=len(df),
        anova_table=table,
    )


def report_twoway_anova(
    result: TwoWayANOVAResult,
    save_path: Optional[str] = None,
) -> None:
    """결과를 콘솔에 출력하고 save_path가 있으면 JSON 저장."""
    W = 70
    print("\n" + "=" * W)
    print("  TWO-WAY ANOVA  (전략 × 시장 국면)")
    print("=" * W)
    print(f"  Metric : {result.metric_used}   N={result.n_obs}   α={result.alpha}")
    print()

    def _sig(b):
        return "✓ significant" if b else "✗ n.s."

    rows = [
        ("Strategy",    result.f_strategy,    result.p_strategy,    result.sig_strategy,    result.eta_sq_partial_strategy),
        ("Regime",      result.f_regime,       result.p_regime,      result.sig_regime,      result.eta_sq_partial_regime),
        ("Interaction", result.f_interaction,  result.p_interaction, result.sig_interaction, result.eta_sq_partial_interaction),
    ]
    hdr = f"  {'Source':<14} {'F':>8} {'p':>10} {'partial η²':>12}  {'Result'}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for src, f, p, sig, eta in rows:
        print(f"  {src:<14} {f:>8.4f} {p:>10.6f} {eta:>12.4f}  {_sig(sig)}")
    print()

    # ── 전략별 평균 ──────────────────────────────────────────
    print("  [전략별 평균 CAGR]")
    for s, m in result.strategy_means.items():
        print(f"    {s:<14}  {m:+.4f}")
    print()

    # ── 국면별 평균 ──────────────────────────────────────────
    print("  [국면별 평균 CAGR  (폴드 수)]")
    for r, m in result.regime_means.items():
        n = result.regime_counts.get(r, "?")
        print(f"    {r:<10}  {m:+.4f}  (n={n})")
    print()

    # ── 셀 평균 히트맵 ────────────────────────────────────────
    strategies = sorted({eval(k)[0] for k in result.cell_means})
    regimes    = sorted({eval(k)[1] for k in result.cell_means})
    col_w = 12
    header_line = f"  {'Strategy':<14}" + "".join(f"{r:>{col_w}}" for r in regimes)
    print("  [셀 평균 CAGR (전략 × 국면)]")
    print(header_line)
    print("  " + "-" * (14 + col_w * len(regimes)))
    for s in strategies:
        row_str = f"  {s:<14}"
        for r in regimes:
            key = str((s, r))
            val = result.cell_means.get(key, float("nan"))
            n   = result.cell_ns.get(key, 0)
            cell = f"{val:+.3f}(n={n})" if not np.isnan(val) else "  N/A"
            row_str += f"{cell:>{col_w}}"
        print(row_str)
    print("=" * W)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"  결과 저장: {path}")


# ─────────────────────────────────────────────────────────────
# 내부: statsmodels 기반 Type-II ANOVA
# ─────────────────────────────────────────────────────────────

def _statsmodels_anova(df: pd.DataFrame):
    """
    statsmodels OLS로 Type-II SS Two-way ANOVA를 수행한다.
    설치되지 않은 경우 ImportError를 발생시켜 수동 계산으로 fallback.
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    model = ols("cagr ~ C(strategy) + C(regime) + C(strategy):C(regime)", data=df).fit()
    anova_table_sm = sm.stats.anova_lm(model, typ=2)

    table = []
    ss_error = float(anova_table_sm.loc["Residual", "sum_sq"])
    df_error = float(anova_table_sm.loc["Residual", "df"])
    ms_error = ss_error / df_error if df_error > 0 else 1e-8

    for idx in anova_table_sm.index:
        if idx == "Residual":
            continue
        ss  = float(anova_table_sm.loc[idx, "sum_sq"])
        dfi = float(anova_table_sm.loc[idx, "df"])
        ms  = ss / dfi if dfi > 0 else 0.0
        f   = float(anova_table_sm.loc[idx, "F"])
        p   = float(anova_table_sm.loc[idx, "PR(>F)"])
        # 인덱스명 정규화 (statsmodels 버전 간 차이 흡수)
        src = str(idx).replace("strategy", "strategy").replace("regime", "regime")
        table.append({"source": src, "SS": round(ss, 6), "df": int(dfi),
                       "MS": round(ms, 6), "F": round(f, 4), "p": round(p, 6)})

    table.append({"source": "Residual", "SS": round(ss_error, 6), "df": int(df_error),
                   "MS": round(ms_error, 6), "F": float("nan"), "p": float("nan")})
    return table, ss_error


# ─────────────────────────────────────────────────────────────
# 내부: 수동 Type-I ANOVA (balanced 가정, statsmodels fallback)
# ─────────────────────────────────────────────────────────────

def _manual_anova(df: pd.DataFrame, strategies: list, regimes: list):
    """
    statsmodels 없을 때 단순 Type-I balanced ANOVA.
    불균형 설계에서는 근사치임을 주의.
    """
    grand_mean = float(df["cagr"].mean())
    n_total    = len(df)

    # SS_A (strategy)
    ss_a = sum(
        len(df.loc[df["strategy"] == s]) * (df.loc[df["strategy"] == s, "cagr"].mean() - grand_mean) ** 2
        for s in strategies
    )
    # SS_B (regime)
    ss_b = sum(
        len(df.loc[df["regime"] == r]) * (df.loc[df["regime"] == r, "cagr"].mean() - grand_mean) ** 2
        for r in regimes
    )
    # SS_total
    ss_total = float(np.sum((df["cagr"].values - grand_mean) ** 2))

    # SS_cells
    ss_cells = sum(
        len(sub) * (float(sub.mean()) - grand_mean) ** 2
        for s in strategies for r in regimes
        for sub in [df.loc[(df["strategy"] == s) & (df["regime"] == r), "cagr"]]
        if len(sub) > 0
    )
    ss_ab     = ss_cells - ss_a - ss_b
    ss_error  = ss_total - ss_cells

    k_a, k_b  = len(strategies), len(regimes)
    df_a      = k_a - 1
    df_b      = k_b - 1
    df_ab     = df_a * df_b
    df_error  = n_total - k_a * k_b

    ms_error  = ss_error / df_error if df_error > 0 else 1e-8
    ms_a      = ss_a  / df_a  if df_a  > 0 else 0.0
    ms_b      = ss_b  / df_b  if df_b  > 0 else 0.0
    ms_ab     = ss_ab / df_ab if df_ab > 0 else 0.0

    f_a  = ms_a  / ms_error
    f_b  = ms_b  / ms_error
    f_ab = ms_ab / ms_error

    p_a  = float(stats.f.sf(f_a,  df_a,  df_error))
    p_b  = float(stats.f.sf(f_b,  df_b,  df_error))
    p_ab = float(stats.f.sf(f_ab, df_ab, df_error))

    table = [
        {"source": "C(strategy)",          "SS": round(ss_a,    6), "df": df_a,
         "MS": round(ms_a, 6),  "F": round(f_a,  4), "p": round(p_a,  6)},
        {"source": "C(regime)",             "SS": round(ss_b,    6), "df": df_b,
         "MS": round(ms_b, 6),  "F": round(f_b,  4), "p": round(p_b,  6)},
        {"source": "C(strategy):C(regime)", "SS": round(ss_ab,   6), "df": df_ab,
         "MS": round(ms_ab,6),  "F": round(f_ab, 4), "p": round(p_ab, 6)},
        {"source": "Residual",              "SS": round(ss_error,6), "df": df_error,
         "MS": round(ms_error, 6), "F": float("nan"), "p": float("nan")},
    ]
    return table, ss_error
