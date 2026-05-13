"""
Walk-Forward 백테스트 모듈.

슬라이딩 윈도우 방식으로 훈련/테스트 구간을 반복하며 모델 성과를 검증한다.

흐름:
  전체 가격 데이터
    → 훈련 윈도우(train_months) + 테스트 윈도우(test_months) 폴드 분할
      → 폴드마다: PPO 학습 → 테스트 구간 추론 → 지표 집계
        → WalkForwardResult (폴드별 + 통합 통계)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from ..agents.ppo_agent import PPOAgent
from ..envs.portfolio_env import PortfolioEnv, RewardVariant
from ..envs.risk_state import RiskState


# ─────────────────────────────────────────────────────────────
# Config / Result 데이터 클래스
# ─────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    train_months: int = 24          # 훈련 윈도우 길이 (월)
    test_months: int = 6            # 테스트 윈도우 길이 (월)
    step_months: int = 6            # 슬라이딩 간격 (월); 보통 test_months와 동일
    min_train_bars: int = 200       # 유효한 폴드로 인정할 최소 훈련 봉 수
    train_timesteps: int = 50_000   # 폴드당 PPO 학습 스텝
    learning_rate: float = 3e-4
    batch_size: int = 256
    window_size: int = 20           # PortfolioEnv window_size
    transaction_cost: float = 0.00015
    slippage: float = 0.0005
    max_drawdown_threshold: float = 0.15
    reward_variant: RewardVariant = RewardVariant.R3_FULL
    risk_free_rate: float = 0.02    # 연간 무위험 수익률 (Sharpe 계산용)
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
    total_return: float             # 누적 수익률
    cagr: float                     # 연환산 수익률
    sharpe: float                   # Sharpe ratio (일간)
    max_drawdown: float             # 최대 낙폭
    portfolio_values: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    config: WalkForwardConfig
    folds: List[FoldMetrics]

    # 통합 지표 (전 폴드 평균 ± 표준편차)
    mean_cagr: float = 0.0
    std_cagr: float = 0.0
    mean_sharpe: float = 0.0
    std_sharpe: float = 0.0
    mean_max_drawdown: float = 0.0
    std_max_drawdown: float = 0.0

    # 전 폴드 이어붙인 equity curve
    equity_curve: Optional[pd.Series] = None

    def summary(self) -> str:
        lines = [
            "=" * 54,
            f"  Walk-Forward 백테스트 결과  ({len(self.folds)} 폴드)",
            "=" * 54,
            f"  CAGR        : {self.mean_cagr:+.2%}  ± {self.std_cagr:.2%}",
            f"  Sharpe      : {self.mean_sharpe:+.3f}  ± {self.std_sharpe:.3f}",
            f"  Max Drawdown: {self.mean_max_drawdown:.2%}  ± {self.std_max_drawdown:.2%}",
            "-" * 54,
        ]
        for fm in self.folds:
            lines.append(
                f"  Fold {fm.fold_idx:02d} "
                f"[{fm.test_start} ~ {fm.test_end}] "
                f"CAGR={fm.cagr:+.2%}  MDD={fm.max_drawdown:.2%}  "
                f"Sharpe={fm.sharpe:.3f}"
            )
        lines.append("=" * 54)
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
    # 공개 API
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> WalkForwardResult:
        """전체 Walk-Forward 실행 후 WalkForwardResult 반환."""
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
                print(f"         CAGR={fm.cagr:+.2%}  MDD={fm.max_drawdown:.2%}  "
                      f"Sharpe={fm.sharpe:.3f}")

        result = self._aggregate(fold_metrics)
        if verbose:
            print("\n" + result.summary())
        return result

    # ------------------------------------------------------------------
    # 내부 구현
    # ------------------------------------------------------------------

    def _build_fold_dates(self):
        """(train_start, train_end, test_start, test_end) 튜플 목록 반환."""
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

            # 실제 존재하는 날짜로 snap
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

        # ── 훈련 ──────────────────────────────────────────────
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

        # ── 테스트 추론 ────────────────────────────────────────
        test_env = self._make_env(test_prices)
        portfolio_values, daily_returns = self._rollout(agent, test_env)

        # ── 지표 계산 ──────────────────────────────────────────
        total_return, cagr, sharpe, mdd = self._compute_metrics(
            portfolio_values, daily_returns, len(test_prices)
        )

        return FoldMetrics(
            fold_idx=fold_idx,
            train_start=str(train_start.date()),
            train_end=str(train_end.date()),
            test_start=str(test_start.date()),
            test_end=str(test_end.date()),
            n_train_bars=len(train_prices),
            n_test_bars=len(test_prices),
            total_return=total_return,
            cagr=cagr,
            sharpe=sharpe,
            max_drawdown=mdd,
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
                prev = portfolio_values[-2]
                curr = portfolio_values[-1]
                daily_returns.append(curr / prev - 1 if prev > 0 else 0.0)

        return portfolio_values, daily_returns

    def _compute_metrics(
        self,
        portfolio_values: List[float],
        daily_returns: List[float],
        n_bars: int,
    ):
        if not portfolio_values or portfolio_values[0] == 0:
            return 0.0, 0.0, 0.0, 0.0

        total_return = portfolio_values[-1] / portfolio_values[0] - 1

        # CAGR
        years = n_bars / self.cfg.trading_days_per_year
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1 / max(years, 1e-6)) - 1

        # Sharpe
        if len(daily_returns) > 1:
            arr = np.array(daily_returns)
            rf_daily = self.cfg.risk_free_rate / self.cfg.trading_days_per_year
            excess = arr - rf_daily
            sigma = float(np.std(arr, ddof=1))
            sharpe = float(np.mean(excess) / sigma * math.sqrt(self.cfg.trading_days_per_year)) if sigma > 0 else 0.0
        else:
            sharpe = 0.0

        # MDD
        arr_pv = np.array(portfolio_values)
        peak = np.maximum.accumulate(arr_pv)
        drawdown = (peak - arr_pv) / (peak + 1e-8)
        mdd = float(drawdown.max())

        return total_return, cagr, sharpe, mdd

    def _aggregate(self, folds: List[FoldMetrics]) -> WalkForwardResult:
        cfg = self.cfg
        if not folds:
            return WalkForwardResult(config=cfg, folds=[])

        cagrs = [f.cagr for f in folds]
        sharpes = [f.sharpe for f in folds]
        mdds = [f.max_drawdown for f in folds]

        # 이어붙인 equity curve
        all_values: List[float] = []
        for fm in folds:
            if not all_values:
                all_values.extend(fm.portfolio_values)
            else:
                scale = all_values[-1] / fm.portfolio_values[0] if fm.portfolio_values[0] else 1.0
                all_values.extend(v * scale for v in fm.portfolio_values[1:])
        equity_curve = pd.Series(all_values, name="portfolio_value")

        return WalkForwardResult(
            config=cfg,
            folds=folds,
            mean_cagr=float(np.mean(cagrs)),
            std_cagr=float(np.std(cagrs, ddof=1)) if len(cagrs) > 1 else 0.0,
            mean_sharpe=float(np.mean(sharpes)),
            std_sharpe=float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0,
            mean_max_drawdown=float(np.mean(mdds)),
            std_max_drawdown=float(np.std(mdds, ddof=1)) if len(mdds) > 1 else 0.0,
            equity_curve=equity_curve,
        )
