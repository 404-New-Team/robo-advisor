"""
Walk-Forward 백테스트 모듈.

슬라이딩 윈도우 방식으로 훈련/테스트 구간을 반복하며 모델 성과를 검증한다.

흐름:
  전체 가격 데이터
    → 훈련 윈도우(train_months) + 테스트 윈도우(test_months) 폴드 분할
      → 폴드마다: PPO 학습 → 테스트 구간 추론 → 12개 지표 집계
        → WalkForwardResult (폴드별 + 통합 통계)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..agents.ppo_agent import PPOAgent
from ..envs.portfolio_env import PortfolioEnv, RewardVariant
from ..envs.risk_state import RiskState
from .metrics import PerformanceMetrics, compute_metrics


# ─────────────────────────────────────────────────────────────
# Config / Result 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    train_months: int = 24
    test_months: int = 6
    step_months: int = 6
    min_train_bars: int = 200
    train_timesteps: int = 50_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    window_size: int = 20
    transaction_cost: float = 0.00015
    slippage: float = 0.0005
    max_drawdown_threshold: float = 0.15
    reward_variant: RewardVariant = RewardVariant.R3_FULL
    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252


@dataclass
class FoldMetrics:
    fold_idx: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train_bars: int
    n_test_bars: int
    metrics: PerformanceMetrics
    portfolio_values: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)

    # 자주 쓰는 지표를 속성으로 노출
    @property
    def cagr(self) -> float:
        return self.metrics.cagr

    @property
    def sharpe(self) -> float:
        return self.metrics.sharpe

    @property
    def max_drawdown(self) -> float:
        return self.metrics.max_drawdown


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: List[FoldMetrics]

    mean_cagr: float = 0.0
    std_cagr: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    mean_max_drawdown: float = 0.0
    std_max_drawdown: float = 0.0
    mean_sortino: float = 0.0
    mean_calmar: float = 0.0
    mean_var_95: float = 0.0
    mean_cvar_95: float = 0.0
    mean_alpha: float = 0.0
    mean_beta: float = 0.0
    mean_information_ratio: float = 0.0

    equity_curve: Optional[pd.Series] = None

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Walk-Forward 백테스트  ({len(self.folds)} 폴드)",
            "=" * 60,
            f"  CAGR             : {self.mean_cagr:+.2%}  ± {self.std_cagr:.2%}",
            f"  Sharpe           : {self.mean_sharpe:+.3f}  ± {self.std_sharpe:.3f}",
            f"  Sortino          : {self.mean_sortino:+.3f}",
            f"  Calmar           : {self.mean_calmar:+.3f}",
            f"  Max Drawdown     : {self.mean_max_drawdown:.2%}  ± {self.std_max_drawdown:.2%}",
            f"  VaR 95%          : {self.mean_var_95:.2%}",
            f"  CVaR 95%         : {self.mean_cvar_95:.2%}",
            f"  Alpha (annual)   : {self.mean_alpha:+.4f}",
            f"  Beta             : {self.mean_beta:+.4f}",
            f"  Info Ratio       : {self.mean_information_ratio:+.3f}",
            "-" * 60,
        ]
        for fm in self.folds:
            m = fm.metrics
            lines.append(
                f"  Fold {fm.fold_idx:02d} [{fm.test_start} ~ {fm.test_end}]  "
                f"CAGR={m.cagr:+.2%}  MDD={m.max_drawdown:.2%}  "
                f"Sharpe={m.sharpe:.3f}  IR={m.information_ratio:.3f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Walk-Forward 엔진
# ─────────────────────────────────────────────────────────────

class WalkForwardBacktest:
    """슬라이딩 윈도우 Walk-Forward 백테스트."""

    def __init__(self, prices: pd.DataFrame, config: WalkForwardConfig = None):
        self.prices = prices
        self.cfg = config or WalkForwardConfig()

    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> WalkForwardResult:
        folds_dates = self._build_fold_dates()
        if not folds_dates:
            raise ValueError(
                "유효한 폴드를 생성할 수 없습니다. "
                "데이터 기간 또는 train_months/test_months 설정을 확인하세요."
            )

        fold_metrics: List[FoldMetrics] = []
        for idx, (train_start, train_end, test_start, test_end) in enumerate(folds_dates):
            if verbose:
                print(f"\n[Fold {idx:02d}] 훈련: {train_start.date()} ~ {train_end.date()} "
                      f"| 테스트: {test_start.date()} ~ {test_end.date()}")

            fm = self._run_fold(idx, train_start, train_end, test_start, test_end, verbose)
            if fm is None:
                continue
            fold_metrics.append(fm)
            if verbose:
                m = fm.metrics
                print(f"         CAGR={m.cagr:+.2%}  MDD={m.max_drawdown:.2%}  "
                      f"Sharpe={m.sharpe:.3f}  Sortino={m.sortino:.3f}  "
                      f"VaR={m.var_95:.2%}  CVaR={m.cvar_95:.2%}")

        result = self._aggregate(fold_metrics)
        if verbose:
            print("\n" + result.summary())
        return result

    # ------------------------------------------------------------------

    def _build_fold_dates(self):
        idx = self.prices.index
        total_start = idx[0]
        total_end = idx[-1]

        folds = []
        train_start = total_start

        while True:
            train_end = train_start + pd.DateOffset(months=self.cfg.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.cfg.test_months)

            if test_end > total_end:
                break

            train_end_snap = idx[idx <= train_end][-1] if any(idx <= train_end) else None
            test_end_snap = idx[idx <= test_end][-1] if any(idx <= test_end) else None

            if train_end_snap is None or test_end_snap is None:
                break

            folds.append((train_start, train_end_snap, test_start, test_end_snap))
            train_start = train_start + pd.DateOffset(months=self.cfg.step_months)

        return folds

    def _slice(self, start, end) -> pd.DataFrame:
        return self.prices.loc[(self.prices.index >= start) & (self.prices.index <= end)]

    def _run_fold(
        self,
        fold_idx: int,
        train_start, train_end,
        test_start, test_end,
        verbose: bool,
    ) -> Optional[FoldMetrics]:
        train_prices = self._slice(train_start, train_end)
        test_prices = self._slice(test_start, test_end)

        if len(train_prices) < self.cfg.min_train_bars:
            if verbose:
                print(f"  → 훈련 데이터 부족 ({len(train_prices)}봉), 건너뜀")
            return None

        # ── 훈련 ──────────────────────────────────────────────────
        train_env = self._make_env(train_prices)
        agent = PPOAgent(
            env=train_env,
            learning_rate=self.cfg.learning_rate,
            batch_size=self.cfg.batch_size,
        )
        agent.train(
            total_timesteps=self.cfg.train_timesteps,
            checkpoint_dir=f"checkpoints/fold_{fold_idx:02d}/",
        )

        # ── 테스트 추론 ────────────────────────────────────────────
        test_env = self._make_env(test_prices)
        portfolio_values, daily_returns = self._rollout(agent, test_env)

        # ── 벤치마크: 동일가중 Buy & Hold ─────────────────────────
        n_steps = len(daily_returns)
        bm_raw = test_prices.pct_change().dropna().mean(axis=1).values
        benchmark_returns = bm_raw[:n_steps] if len(bm_raw) >= n_steps else bm_raw

        # ── 12개 성과 지표 ─────────────────────────────────────────
        perf = compute_metrics(
            daily_returns=daily_returns,
            portfolio_values=portfolio_values,
            benchmark_returns=benchmark_returns,
            n_bars=len(test_prices),
            trading_days=self.cfg.trading_days_per_year,
            risk_free_rate=self.cfg.risk_free_rate,
        )

        return FoldMetrics(
            fold_idx=fold_idx,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            n_train_bars=len(train_prices),
            n_test_bars=len(test_prices),
            metrics=perf,
            portfolio_values=portfolio_values,
            daily_returns=daily_returns,
        )

    def _make_env(self, prices: pd.DataFrame) -> PortfolioEnv:
        return PortfolioEnv(
            prices=prices,
            risk_state=RiskState(),
            window_size=self.cfg.window_size,
            transaction_cost=self.cfg.transaction_cost,
            slippage=self.cfg.slippage,
            max_drawdown_threshold=self.cfg.max_drawdown_threshold,
            reward_variant=self.cfg.reward_variant,
        )

    @staticmethod
    def _rollout(agent: PPOAgent, env: PortfolioEnv):
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

    def _aggregate(self, folds: List[FoldMetrics]) -> WalkForwardResult:
        if not folds:
            return WalkForwardResult(config=self.cfg, folds=[])

        def _mean(attr): return float(np.mean([getattr(f.metrics, attr) for f in folds]))
        def _std(attr):
            vals = [getattr(f.metrics, attr) for f in folds]
            return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # 이어붙인 equity curve
        all_values: List[float] = []
        for fm in folds:
            if not all_values:
                all_values.extend(fm.portfolio_values)
            else:
                scale = all_values[-1] / fm.portfolio_values[0] if fm.portfolio_values[0] else 1.0
                all_values.extend(v * scale for v in fm.portfolio_values[1:])

        return WalkForwardResult(
            config=self.cfg,
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
