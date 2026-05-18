"""
Mean-Variance Optimization (MVO) 모듈 — Markowitz (1952).

포트폴리오 최적화 전략:
  min_variance : 최소 분산 포트폴리오
  max_sharpe   : 최대 Sharpe ratio 포트폴리오 (기본)

효율적 프론티어 계산 및 Walk-Forward 백테스트 포함.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import PerformanceMetrics, compute_metrics
from .walk_forward import WalkForwardConfig, FoldMetrics, WalkForwardResult


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class MVOConfig:
    target: str = "max_sharpe"  # "min_variance" | "max_sharpe"
    risk_free_rate: float = 0.02
    weight_min: float = 0.0     # 개별 자산 최소 비중 (0 = long-only)
    weight_max: float = 1.0     # 개별 자산 최대 비중
    cov_regularization: float = 1e-5  # 공분산 정규화 (특이행렬 방지)
    trading_days: int = 252
    n_frontier_points: int = 50  # 효율적 프론티어 포인트 수


# ─────────────────────────────────────────────────────────────
# MVO 최적화기
# ─────────────────────────────────────────────────────────────

class MVO:
    """Markowitz Mean-Variance 포트폴리오 최적화기."""

    def __init__(self, config: MVOConfig = None):
        self.cfg = config or MVOConfig()
        self.mu_: Optional[np.ndarray] = None   # 연환산 기대수익률
        self.cov_: Optional[np.ndarray] = None  # 연환산 공분산 행렬
        self.n_: int = 0
        self.tickers_: list = []

    # ------------------------------------------------------------------
    # 추정
    # ------------------------------------------------------------------

    def fit(self, prices: pd.DataFrame) -> "MVO":
        """가격 데이터로 기대수익률과 공분산 행렬을 추정한다."""
        rets = prices.pct_change().dropna()
        self.tickers_ = list(prices.columns)
        self.n_ = len(self.tickers_)
        td = self.cfg.trading_days

        self.mu_ = rets.mean().values * td
        raw_cov = rets.cov().values * td
        # Ledoit-Wolf 스타일 대각 정규화로 특이행렬 방지
        self.cov_ = raw_cov + np.eye(self.n_) * self.cfg.cov_regularization
        return self

    # ------------------------------------------------------------------
    # 최적화
    # ------------------------------------------------------------------

    def get_weights(self) -> np.ndarray:
        """설정된 target에 따라 최적 비중 반환."""
        if self.cfg.target == "min_variance":
            return self.min_variance_weights()
        return self.max_sharpe_weights()

    def min_variance_weights(self) -> np.ndarray:
        """분산을 최소화하는 포트폴리오 비중."""
        self._check_fitted()

        def objective(w):
            return float(w @ self.cov_ @ w)

        def grad(w):
            return 2 * self.cov_ @ w

        return self._solve(objective, grad)

    def max_sharpe_weights(self) -> np.ndarray:
        """Sharpe ratio를 최대화하는 포트폴리오 비중."""
        self._check_fitted()
        rf = self.cfg.risk_free_rate

        def objective(w):
            port_ret = float(w @ self.mu_)
            port_vol = float(np.sqrt(w @ self.cov_ @ w + 1e-12))
            return -(port_ret - rf) / port_vol  # 음수: 최소화 = Sharpe 최대화

        def grad(w):
            port_ret = float(w @ self.mu_)
            port_var = float(w @ self.cov_ @ w + 1e-12)
            port_vol = float(np.sqrt(port_var))
            denom = port_vol
            d_ret = self.mu_
            d_vol = (self.cov_ @ w) / denom
            sharpe = (port_ret - rf) / denom
            return -(d_ret * denom - (port_ret - rf) * d_vol) / (denom ** 2)

        return self._solve(objective, grad)

    def portfolio_stats(self, weights: np.ndarray) -> dict:
        """주어진 비중의 기대수익률·변동성·Sharpe 반환."""
        self._check_fitted()
        ret = float(weights @ self.mu_)
        vol = float(np.sqrt(weights @ self.cov_ @ weights))
        sharpe = (ret - self.cfg.risk_free_rate) / max(vol, 1e-10)
        return {"return": ret, "volatility": vol, "sharpe": sharpe}

    # ------------------------------------------------------------------
    # 효율적 프론티어
    # ------------------------------------------------------------------

    def efficient_frontier(self) -> pd.DataFrame:
        """
        효율적 프론티어 포인트 계산.

        Returns:
            DataFrame with columns: return, volatility, sharpe, weights_*
        """
        self._check_fitted()
        n_pts = self.cfg.n_frontier_points

        # 목표 수익률 범위: 최소분산 수익률 ~ 최대 개별자산 수익률
        w_minvar = self.min_variance_weights()
        ret_min = float(w_minvar @ self.mu_)
        ret_max = float(self.mu_.max())

        if ret_min >= ret_max:
            ret_min = float(self.mu_.min())

        target_rets = np.linspace(ret_min, ret_max, n_pts)
        rows = []

        for target_ret in target_rets:
            w = self._solve_for_target_return(target_ret)
            if w is None:
                continue
            stats = self.portfolio_stats(w)
            row = {
                "return": stats["return"],
                "volatility": stats["volatility"],
                "sharpe": stats["sharpe"],
            }
            for i, ticker in enumerate(self.tickers_):
                row[f"w_{ticker}"] = float(w[i])
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _solve(self, objective, grad=None) -> np.ndarray:
        n = self.n_
        x0 = np.ones(n) / n
        bounds = [(self.cfg.weight_min, self.cfg.weight_max)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]

        result = minimize(
            objective,
            x0,
            jac=grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success or np.any(np.isnan(result.x)):
            return x0  # 수렴 실패 시 동일가중 반환

        w = np.clip(result.x, self.cfg.weight_min, self.cfg.weight_max)
        total = w.sum()
        return w / total if total > 1e-8 else x0

    def _solve_for_target_return(self, target_ret: float) -> Optional[np.ndarray]:
        """목표 수익률 달성하는 최소분산 포트폴리오 비중."""
        n = self.n_
        x0 = np.ones(n) / n
        bounds = [(self.cfg.weight_min, self.cfg.weight_max)] * n
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: float(w @ self.mu_) - target_ret},
        ]

        result = minimize(
            lambda w: float(w @ self.cov_ @ w),
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )

        if not result.success or np.any(np.isnan(result.x)):
            return None
        w = np.clip(result.x, 0, None)
        total = w.sum()
        return w / total if total > 1e-8 else None

    def _check_fitted(self) -> None:
        if self.mu_ is None or self.cov_ is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")


# ─────────────────────────────────────────────────────────────
# MVO Walk-Forward 백테스트
# ─────────────────────────────────────────────────────────────

def run_mvo_walk_forward(
    prices: pd.DataFrame,
    wf_config: WalkForwardConfig,
    mvo_config: MVOConfig = None,
    verbose: bool = True,
) -> WalkForwardResult:
    """
    MVO를 Walk-Forward 방식으로 백테스트한다.

    훈련 구간에서 MVO 비중을 결정하고, 테스트 구간에서
    해당 비중을 매일 리밸런싱(constant-mix)하여 수익률을 평가한다.
    """
    mvo_cfg = mvo_config or MVOConfig(
        risk_free_rate=wf_config.risk_free_rate,
        trading_days=wf_config.trading_days_per_year,
    )

    idx = prices.index
    folds_dates = _build_fold_dates(prices, wf_config)

    if not folds_dates:
        raise ValueError("유효한 폴드를 생성할 수 없습니다.")

    fold_metrics: List[FoldMetrics] = []

    for fold_idx, (train_start, train_end, test_start, test_end) in enumerate(folds_dates):
        train_prices = prices.loc[(prices.index >= train_start) & (prices.index <= train_end)]
        test_prices = prices.loc[(prices.index >= test_start) & (prices.index <= test_end)]

        if len(train_prices) < wf_config.min_train_bars:
            if verbose:
                print(f"  Fold {fold_idx:02d} 건너뜀 (훈련 데이터 부족)")
            continue

        # MVO 비중 결정 (훈련 구간 기반)
        mvo = MVO(mvo_cfg)
        mvo.fit(train_prices)
        weights = mvo.get_weights()

        if verbose:
            stats = mvo.portfolio_stats(weights)
            print(f"\n[MVO Fold {fold_idx:02d}] 테스트: {test_start.date()} ~ {test_end.date()}")
            print(f"  비중: { {t: f'{w:.3f}' for t, w in zip(mvo.tickers_, weights)} }")
            print(f"  기대: ret={stats['return']:.2%}  vol={stats['volatility']:.2%}  "
                  f"sharpe={stats['sharpe']:.3f}")

        # 테스트 구간 시뮬레이션 (일별 constant-mix)
        test_rets = test_prices.pct_change().dropna()
        daily_port_rets = test_rets.values @ weights  # 매일 비중 × 자산수익률
        portfolio_values = _rets_to_values(daily_port_rets)

        # 벤치마크: 동일가중
        bm_returns = test_rets.mean(axis=1).values

        perf = compute_metrics(
            daily_returns=daily_port_rets.tolist(),
            portfolio_values=portfolio_values,
            benchmark_returns=bm_returns,
            n_bars=len(test_prices),
            trading_days=wf_config.trading_days_per_year,
            risk_free_rate=wf_config.risk_free_rate,
        )

        if verbose:
            print(f"  결과: CAGR={perf.cagr:+.2%}  MDD={perf.max_drawdown:.2%}  "
                  f"Sharpe={perf.sharpe:.3f}")

        fold_metrics.append(FoldMetrics(
            fold_idx=fold_idx,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            n_train_bars=len(train_prices),
            n_test_bars=len(test_prices),
            metrics=perf,
            portfolio_values=portfolio_values,
            daily_returns=daily_port_rets.tolist(),
        ))

    result = _aggregate_folds(fold_metrics, wf_config)
    if verbose:
        print("\n" + result.summary())
    return result


# ─────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────

def _build_fold_dates(prices: pd.DataFrame, cfg: WalkForwardConfig):
    idx = prices.index
    total_start, total_end = idx[0], idx[-1]
    folds = []
    train_start = total_start

    while True:
        train_end = train_start + pd.DateOffset(months=cfg.train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=cfg.test_months)
        if test_end > total_end:
            break
        te_snap = idx[idx <= train_end][-1] if any(idx <= train_end) else None
        ts_snap = idx[idx <= test_end][-1] if any(idx <= test_end) else None
        if te_snap is None or ts_snap is None:
            break
        folds.append((train_start, te_snap, test_start, ts_snap))
        train_start = train_start + pd.DateOffset(months=cfg.step_months)

    return folds


def _rets_to_values(daily_rets: np.ndarray) -> List[float]:
    pv = [1.0]
    for r in daily_rets:
        pv.append(pv[-1] * (1 + float(r)))
    return pv


def _aggregate_folds(folds: List[FoldMetrics], cfg: WalkForwardConfig) -> WalkForwardResult:
    if not folds:
        return WalkForwardResult(config=cfg, folds=[])

    def _mean(attr):
        return float(np.mean([getattr(f.metrics, attr) for f in folds]))

    def _std(attr):
        vals = [getattr(f.metrics, attr) for f in folds]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    all_values: List[float] = []
    for fm in folds:
        if not all_values:
            all_values.extend(fm.portfolio_values)
        else:
            scale = all_values[-1] / fm.portfolio_values[0] if fm.portfolio_values[0] else 1.0
            all_values.extend(v * scale for v in fm.portfolio_values[1:])

    return WalkForwardResult(
        config=cfg,
        folds=folds,
        mean_cagr=_mean("cagr"),
        std_cagr=_std("cagr"),
        mean_sharpe=_mean("sharpe"),
        std_sharpe=_std("sharpe"),
        mean_max_drawdown=_mean("max_drawdown"),
        std_max_drawdown=_std("max_drawdown"),
        mean_sortino=_mean("sortino"),
        mean_calmar=_mean("calmar"),
        mean_var_95=_mean("var_95"),
        mean_cvar_95=_mean("cvar_95"),
        mean_alpha=_mean("alpha"),
        mean_beta=_mean("beta"),
        mean_information_ratio=_mean("information_ratio"),
        equity_curve=pd.Series(all_values, name="portfolio_value"),
    )
