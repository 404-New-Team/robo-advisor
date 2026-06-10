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
from concurrent.futures import ProcessPoolExecutor
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
    test_method: str            # "one-way-anova"
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

def _run_strategy_fold(args: dict) -> dict:
    """
    단일 폴드의 EW · MVO · DRL(멀티시드) CAGR을 계산해 반환.

    ProcessPoolExecutor 워커로 사용되므로 모듈 레벨에 정의한다.
    병렬 실행 시 PPO verbose=0으로 로그 억제.
    """
    from ..backtest.mvo import MVO, MVOConfig
    from ..envs.portfolio_env import PortfolioEnv
    from ..envs.risk_state import RiskState
    from ..agents.ppo_agent import PPOAgent
    from ..backtest.metrics import compute_metrics

    fold_idx         = args["fold_idx"]
    train_prices     = args["train_prices"]
    test_prices      = args["test_prices"]
    n_seeds          = args["n_seeds"]
    drl_timesteps    = args["drl_timesteps"]
    window_size      = args["window_size"]
    transaction_cost = args["transaction_cost"]
    slippage         = args["slippage"]
    mdd_threshold    = args["mdd_threshold"]
    risk_free_rate   = args["risk_free_rate"]
    trading_days     = args["trading_days"]
    ppo_verbose      = args.get("ppo_verbose", 0)   # 병렬 시 기본 0

    # ── EqualWeight ───────────────────────────────────────────
    ew_cagr = _equal_weight_cagr_with_cost(test_prices, transaction_cost, slippage)

    # ── MVO ───────────────────────────────────────────────────
    mvo = MVO(MVOConfig(target="max_sharpe",
                         risk_free_rate=risk_free_rate,
                         trading_days=trading_days))
    mvo.fit(train_prices)
    mvo_cagr = _fixed_weight_cagr_with_cost(
        test_prices, mvo.get_weights(), transaction_cost, slippage
    )

    # ── DRL 멀티시드 앙상블 ────────────────────────────────────
    agents = []
    for seed_idx in range(n_seeds):
        train_env = PortfolioEnv(
            prices=train_prices,
            risk_state=RiskState(),
            window_size=window_size,
            transaction_cost=transaction_cost,
            slippage=slippage,
            max_drawdown_threshold=mdd_threshold,
        )
        agent = PPOAgent(env=train_env, seed=seed_idx * 42, verbose=ppo_verbose)
        agent.train(
            total_timesteps=drl_timesteps,
            checkpoint_dir=f"checkpoints/anova_fold_{fold_idx:02d}_seed{seed_idx}/",
        )
        agents.append(agent)

    test_env = PortfolioEnv(
        prices=test_prices,
        risk_state=RiskState(),
        window_size=window_size,
        transaction_cost=transaction_cost,
        slippage=slippage,
        max_drawdown_threshold=mdd_threshold,
    )
    pv, dr = _rollout_full(agents, test_env, test_prices)
    perf = compute_metrics(
        daily_returns=dr,
        portfolio_values=pv,
        n_bars=len(test_prices),
        trading_days=trading_days,
        risk_free_rate=risk_free_rate,
    )

    return {
        "fold_idx":    fold_idx,
        "DRL":         float(perf.cagr),
        "MVO":         float(mvo_cagr),
        "EqualWeight": float(ew_cagr),
    }


def collect_strategy_returns(
    prices: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    drl_timesteps: int = 150_000,
    window_size: int = 20,
    transaction_cost: float = 0.00015,
    slippage: float = 0.0005,
    max_drawdown_threshold: float = 0.25,
    risk_free_rate: float = 0.02,
    n_seeds: int = 3,
    n_jobs: int = 1,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    세 전략의 Walk-Forward 폴드별 CAGR 수집.

    n_jobs > 1 이면 ProcessPoolExecutor로 폴드를 병렬 처리한다.
    폴드는 서로 독립적이므로 n_jobs=코어수로 설정하면 거의 선형 단축된다.

    Returns:
        {"DRL": [...], "MVO": [...], "EqualWeight": [...]}
    """
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

    # ── 폴드별 인자 dict 구성 ──────────────────────────────────
    fold_args = []
    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds_dates):
        train_prices = _slice(prices, train_start, train_end)
        test_prices  = _slice(prices, test_start, test_end)

        if len(train_prices) < wf_cfg.min_train_bars:
            if verbose:
                print(f"  Fold {fold_idx:02d} 건너뜀 (훈련 데이터 부족)")
            continue

        fold_args.append({
            "fold_idx":         fold_idx,
            "train_prices":     train_prices,
            "test_prices":      test_prices,
            "n_seeds":          n_seeds,
            "drl_timesteps":    drl_timesteps,
            "window_size":      window_size,
            "transaction_cost": transaction_cost,
            "slippage":         slippage,
            "mdd_threshold":    max_drawdown_threshold,
            "risk_free_rate":   risk_free_rate,
            "trading_days":     wf_cfg.trading_days_per_year,
            # 병렬 시 PPO 로그 억제, 순차 시 verbose 유지
            "ppo_verbose":      0 if n_jobs > 1 else int(verbose),
        })

    n_workers = min(n_jobs, len(fold_args))
    if verbose:
        mode = f"병렬 {n_workers}코어" if n_workers > 1 else "순차"
        print(
            f"[전략 ANOVA] 총 {len(fold_args)}개 폴드 | {mode} | "
            f"DRL seeds={n_seeds} timesteps={drl_timesteps:,}"
        )

    # ── 병렬 or 순차 실행 ─────────────────────────────────────
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            fold_results = list(executor.map(_run_strategy_fold, fold_args))
        if verbose:
            for fr in sorted(fold_results, key=lambda x: x["fold_idx"]):
                print(
                    f"  Fold {fr['fold_idx']:02d} | "
                    f"EW={fr['EqualWeight']:+.2%}  "
                    f"MVO={fr['MVO']:+.2%}  "
                    f"DRL={fr['DRL']:+.2%}"
                )
    else:
        fold_results = []
        for args in fold_args:
            fr = _run_strategy_fold(args)
            fold_results.append(fr)
            if verbose:
                print(
                    f"  Fold {fr['fold_idx']:02d} | "
                    f"EW={fr['EqualWeight']:+.2%}  "
                    f"MVO={fr['MVO']:+.2%}  "
                    f"DRL={fr['DRL']:+.2%}"
                )

    # ── 폴드 순서대로 결과 조립 ────────────────────────────────
    fold_results.sort(key=lambda x: x["fold_idx"])
    results: dict[str, list[float]] = {"DRL": [], "MVO": [], "EqualWeight": []}
    for fr in fold_results:
        results["DRL"].append(fr["DRL"])
        results["MVO"].append(fr["MVO"])
        results["EqualWeight"].append(fr["EqualWeight"])

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
        test_method="one-way-anova",
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


def _equal_weight_cagr_with_cost(
    test_prices: pd.DataFrame,
    transaction_cost: float,
    slippage: float,
) -> float:
    """동일가중 Buy & Hold CAGR — 초기 포트폴리오 구성 1회 진입 비용 포함."""
    rets = test_prices.pct_change().dropna()
    daily_port = rets.mean(axis=1).values
    # 초기 현금→EW 전환 시 턴오버=1.0 (전 자산 매수)
    entry_cost = (transaction_cost + slippage) * 1.0
    pv = (1.0 - entry_cost) * float(np.prod(1 + daily_port))
    years = len(daily_port) / 252
    return float(pv ** (1 / max(years, 1e-8)) - 1)


def _fixed_weight_cagr_with_cost(
    test_prices: pd.DataFrame,
    weights: np.ndarray,
    transaction_cost: float,
    slippage: float,
) -> float:
    """고정 비중 constant-mix CAGR — 초기 포트폴리오 구성 1회 진입 비용 포함."""
    rets = test_prices.pct_change().dropna()
    daily_port = rets.values @ weights
    # 초기 현금→MVO 전환 시 턴오버=sum(|w|)=1.0
    entry_cost = (transaction_cost + slippage) * float(np.sum(np.abs(weights)))
    pv = (1.0 - entry_cost) * float(np.prod(1 + daily_port))
    years = len(daily_port) / 252
    return float(pv ** (1 / max(years, 1e-8)) - 1)


# 하위호환용 별칭 (market_regime_anova.py 등에서 import)
def _equal_weight_cagr(test_prices: pd.DataFrame, risk_free_rate: float) -> float:
    return _equal_weight_cagr_with_cost(test_prices, 0.00015, 0.0005)


def _fixed_weight_cagr(
    test_prices: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float,
) -> float:
    return _fixed_weight_cagr_with_cost(test_prices, weights, 0.00015, 0.0005)


def _rollout_full(
    agents,
    env,
    full_test_prices: pd.DataFrame,
) -> tuple:
    """
    DRL 롤아웃 — 조기 종료(MDD 초과) 시 남은 기간을 EW 수익률로 채워 전체 기간 완주.

    모든 전략이 동일 기간을 완주해야 CAGR 비교가 공정하다.
    agents: PPOAgent 단일 객체 또는 리스트 (리스트이면 앙상블 평균 예측).
    """
    if isinstance(agents, list):
        predict_fn = lambda obs: np.mean([a.predict(obs) for a in agents], axis=0)
    else:
        predict_fn = agents.predict

    obs, _ = env.reset()
    pv_list = [1.0]
    dr_list = []
    last_info: dict = {"step": 0}
    early_stopped = False

    while True:
        action = predict_fn(obs)
        obs, _, terminated, truncated, info = env.step(action)
        last_info = info
        pv_list.append(info["portfolio_value"])
        if len(pv_list) >= 2:
            prev, curr = pv_list[-2], pv_list[-1]
            dr_list.append(curr / prev - 1 if prev > 0 else 0.0)

        if terminated and not truncated:
            early_stopped = True
            break
        if truncated or terminated:
            break

    # 조기 종료 시 남은 기간을 EW 수익률로 채움
    if early_stopped:
        last_step_idx = min(last_info["step"] - 1, len(env.valid_dates) - 1)
        cutoff_date = env.valid_dates[last_step_idx]
        remaining = full_test_prices.loc[full_test_prices.index > cutoff_date]
        if len(remaining) > 1:
            ew_rets = remaining.pct_change().dropna().mean(axis=1).values
            last_pv = pv_list[-1]
            for r in ew_rets:
                last_pv = last_pv * (1.0 + float(r))
                pv_list.append(last_pv)
                dr_list.append(float(r))

    return pv_list, dr_list


def _rollout(agent, env):
    """하위호환용 — 조기 종료 보정 없이 단순 롤아웃."""
    return _rollout_full(agent, env, pd.DataFrame())


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
