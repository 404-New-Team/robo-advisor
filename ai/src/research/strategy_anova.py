"""
전략 비교 ANOVA 모듈 — ANOVA 검증 2.

DRL(PPO) vs MVO vs 동일가중(Equal-Weight) 세 전략을
Walk-Forward 폴드별 CAGR로 One-way ANOVA 비교.

효과 크기 η² (eta-squared)까지 함께 산출한다.

흐름:
  collect_strategy_returns()  — 3전략 × n_folds CAGR 수집
    → run_strategy_anova()    — One-way ANOVA + Tukey HSD + η²
      → report_strategy_anova() — 콘솔 출력 + JSON 저장
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────
# 결과 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class StrategyANOVAResult:
    # ── ANOVA ──────────────────────────────────────────────────
    f_statistic: float
    p_value: float
    significant: bool
    alpha: float
    # ── 효과 크기 ────────────────────────────────────────────────
    eta_squared: float          # η² = SS_between / SS_total
    eta_squared_interp: str     # "small" / "medium" / "large"
    # ── 그룹별 통계 ──────────────────────────────────────────────
    group_means: dict
    group_stds: dict
    group_ns: dict
    metric_used: str            # e.g. "fold_cagr"
    # ── 사후 검정 ────────────────────────────────────────────────
    tukey_results: list


# ─────────────────────────────────────────────────────────────
# 데이터 수집
# ─────────────────────────────────────────────────────────────

def collect_strategy_returns(
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
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    세 전략의 Walk-Forward 폴드별 CAGR 수집.

    Returns:
        {"DRL": [...], "MVO": [...], "EqualWeight": [...]}
    """
    from ..backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig
    from ..backtest.mvo import MVO, MVOConfig, run_mvo_walk_forward

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

    results: dict[str, list[float]] = {"DRL": [], "MVO": [], "EqualWeight": []}

    # ── 폴드 날짜 공유 ─────────────────────────────────────────
    folds_dates = _build_fold_dates(prices, wf_cfg)
    if not folds_dates:
        raise ValueError("유효한 폴드가 없습니다.")

    if verbose:
        print(f"[전략 ANOVA] 총 {len(folds_dates)}개 폴드")

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds_dates):
        train_prices = _slice(prices, train_start, train_end)
        test_prices  = _slice(prices, test_start, test_end)

        if len(train_prices) < wf_cfg.min_train_bars:
            if verbose:
                print(f"  Fold {fold_idx:02d} 건너뜀 (훈련 데이터 부족)")
            continue

        if verbose:
            print(f"\n  Fold {fold_idx:02d} | 테스트: {test_start.date()} ~ {test_end.date()}")

        # ── 1. Equal-Weight ────────────────────────────────────
        ew_cagr = _equal_weight_cagr(test_prices, risk_free_rate)
        results["EqualWeight"].append(ew_cagr)
        if verbose:
            print(f"    EqualWeight CAGR: {ew_cagr:+.2%}")

        # ── 2. MVO ─────────────────────────────────────────────
        mvo = MVO(MVOConfig(target="max_sharpe",
                             risk_free_rate=risk_free_rate,
                             trading_days=wf_cfg.trading_days_per_year))
        mvo.fit(train_prices)
        mvo_weights = mvo.get_weights()
        mvo_cagr = _fixed_weight_cagr(test_prices, mvo_weights, risk_free_rate)
        results["MVO"].append(mvo_cagr)
        if verbose:
            print(f"    MVO        CAGR: {mvo_cagr:+.2%}")

        # ── 3. DRL ─────────────────────────────────────────────
        from ..envs.portfolio_env import PortfolioEnv
        from ..envs.risk_state import RiskState
        from ..agents.ppo_agent import PPOAgent
        from ..backtest.metrics import compute_metrics

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
            checkpoint_dir=f"checkpoints/anova_fold_{fold_idx:02d}/",
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
        results["DRL"].append(perf.cagr)
        if verbose:
            print(f"    DRL        CAGR: {perf.cagr:+.2%}")

    if verbose:
        print(f"\n수집 완료: {[(k, len(v)) for k, v in results.items()]}")

    return results


# ─────────────────────────────────────────────────────────────
# ANOVA 검정
# ─────────────────────────────────────────────────────────────

def run_strategy_anova(
    returns_by_strategy: dict[str, list[float]],
    alpha: float = 0.05,
    metric_name: str = "fold_cagr",
) -> StrategyANOVAResult:
    """
    One-way ANOVA + Tukey HSD post-hoc + η² 효과 크기.

    Args:
        returns_by_strategy: {"DRL": [...], "MVO": [...], "EqualWeight": [...]}
        alpha: 유의 수준
        metric_name: 메트릭 이름 (결과 레이블용)
    """
    groups = list(returns_by_strategy.keys())
    data   = [np.array(returns_by_strategy[g], dtype=float) for g in groups]

    # ── One-way ANOVA ──────────────────────────────────────────
    f_stat, p_val = stats.f_oneway(*data)

    # ── η² 효과 크기 ───────────────────────────────────────────
    all_vals   = np.concatenate(data)
    grand_mean = float(np.mean(all_vals))
    ss_between = float(sum(len(d) * (float(np.mean(d)) - grand_mean) ** 2 for d in data))
    ss_total   = float(np.sum((all_vals - grand_mean) ** 2))
    eta_sq     = ss_between / ss_total if ss_total > 1e-12 else 0.0

    # Cohen 기준: η² < 0.01 small / 0.06 medium / 0.14 large
    if eta_sq < 0.01:
        eta_interp = "small (η²<0.01)"
    elif eta_sq < 0.06:
        eta_interp = "small~medium (0.01≤η²<0.06)"
    elif eta_sq < 0.14:
        eta_interp = "medium (0.06≤η²<0.14)"
    else:
        eta_interp = "large (η²≥0.14)"

    # ── 그룹 통계 ──────────────────────────────────────────────
    group_means = {g: float(np.mean(d)) for g, d in zip(groups, data)}
    group_stds  = {g: float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
                   for g, d in zip(groups, data)}
    group_ns    = {g: len(d) for g, d in zip(groups, data)}

    # ── Tukey HSD post-hoc ─────────────────────────────────────
    tukey = _tukey_hsd(groups, data, alpha)

    return StrategyANOVAResult(
        f_statistic=round(float(f_stat), 4),
        p_value=round(float(p_val), 6),
        significant=bool(p_val < alpha),
        alpha=alpha,
        eta_squared=round(eta_sq, 4),
        eta_squared_interp=eta_interp,
        group_means=group_means,
        group_stds=group_stds,
        group_ns=group_ns,
        metric_used=metric_name,
        tukey_results=tukey,
    )


def report_strategy_anova(
    result: StrategyANOVAResult,
    save_path: Optional[str] = None,
) -> None:
    """결과를 콘솔에 출력하고 save_path가 있으면 JSON 저장."""
    sig_str = f"significant ✓" if result.significant else "not significant ✗"
    print("\n" + "=" * 64)
    print("  STRATEGY ANOVA  (DRL vs MVO vs EqualWeight)")
    print("=" * 64)
    print(f"  Metric      : {result.metric_used}")
    print(f"  F-statistic : {result.f_statistic:.4f}")
    print(f"  p-value     : {result.p_value:.6f}  ({sig_str} at α={result.alpha})")
    print(f"  η² (effect) : {result.eta_squared:.4f}  → {result.eta_squared_interp}")
    print()
    header = f"  {'Strategy':<14} {'Mean':>10} {'Std':>10} {'N':>5}"
    print(header)
    print("  " + "-" * 42)
    for g in result.group_means:
        print(f"  {g:<14} {result.group_means[g]:>+10.4f} "
              f"{result.group_stds[g]:>10.4f} {result.group_ns[g]:>5d}")
    print()
    print("  Tukey HSD Pairwise:")
    print(f"  {'Strat1':<12} {'Strat2':<12} {'Diff':>8} {'q':>7} {'p-approx':>10} {'Sig':>5}")
    print("  " + "-" * 58)
    for t in result.tukey_results:
        sig = "✓" if t["significant"] else " "
        print(f"  {t['group1']:<12} {t['group2']:<12} "
              f"{t['mean_diff']:>+8.4f} {t['q_statistic']:>7.4f} "
              f"{t['p_value_approx']:>10.6f} {sig:>5}")
    print("=" * 64)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        print(f"  결과 저장: {path}")


# ─────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────

def _build_fold_dates(prices, cfg):
    from ..backtest.walk_forward import WalkForwardConfig
    idx = prices.index
    total_start, total_end = idx[0], idx[-1]
    folds = []
    train_start = total_start
    while True:
        train_end = train_start + pd.DateOffset(months=cfg.train_months)
        test_start = train_end
        test_end   = test_start + pd.DateOffset(months=cfg.test_months)
        if test_end > total_end:
            break
        te_snap = idx[idx <= train_end][-1] if any(idx <= train_end) else None
        ts_snap = idx[idx <= test_end][-1]  if any(idx <= test_end)  else None
        if te_snap is None or ts_snap is None:
            break
        folds.append((train_start, te_snap, test_start, ts_snap))
        train_start = train_start + pd.DateOffset(months=cfg.step_months)
    return folds


def _slice(prices, start, end) -> pd.DataFrame:
    return prices.loc[(prices.index >= start) & (prices.index <= end)]


def _equal_weight_cagr(test_prices: pd.DataFrame, risk_free_rate: float) -> float:
    """동일가중 Buy & Hold CAGR."""
    import math
    rets = test_prices.pct_change().dropna()
    daily_port = rets.mean(axis=1).values
    pv = float(np.prod(1 + daily_port))
    years = len(daily_port) / 252
    return float(pv ** (1 / max(years, 1e-8)) - 1)


def _fixed_weight_cagr(
    test_prices: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float,
) -> float:
    """고정 비중 constant-mix CAGR."""
    rets = test_prices.pct_change().dropna()
    daily_port = rets.values @ weights
    pv = float(np.prod(1 + daily_port))
    years = len(daily_port) / 252
    return float(pv ** (1 / max(years, 1e-8)) - 1)


def _rollout(agent, env):
    obs, _ = env.reset()
    portfolio_values = [1.0]
    daily_returns = []
    done = False
    while not done:
        action = agent.predict(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        portfolio_values.append(info["portfolio_value"])
        if len(portfolio_values) >= 2:
            prev, curr = portfolio_values[-2], portfolio_values[-1]
            daily_returns.append(curr / prev - 1 if prev > 0 else 0.0)
    return portfolio_values, daily_returns


def _tukey_hsd(groups, data, alpha):
    n_total = sum(len(d) for d in data)
    k = len(groups)
    ss_within = float(sum(np.sum((d - np.mean(d)) ** 2) for d in data))
    df_within = n_total - k
    mse = ss_within / df_within if df_within > 0 else 1e-8

    results = []
    for (i, g1), (j, g2) in combinations(enumerate(groups), 2):
        n1, n2 = len(data[i]), len(data[j])
        mean_diff = float(np.mean(data[i]) - np.mean(data[j]))
        se = float(np.sqrt(mse * (1 / n1 + 1 / n2) / 2)) + 1e-12
        q_stat = abs(mean_diff) / se
        t_stat = q_stat / np.sqrt(2)
        p_approx = float(2 * stats.t.sf(t_stat, df=df_within))
        results.append({
            "group1": g1,
            "group2": g2,
            "mean_diff": round(mean_diff, 4),
            "q_statistic": round(q_stat, 4),
            "p_value_approx": round(p_approx, 6),
            "significant": p_approx < alpha,
        })
    return results
