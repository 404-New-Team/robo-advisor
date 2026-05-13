"""
포트폴리오 관리 RL 환경 (Gymnasium 호환).

관측 공간: [시장 피처 (n_assets×11)] + [리스크 태그 (n_tags)] + [현재 비중 (n_assets)]
  시장 피처 11개: ret1d, ret5d, ret20d, vol20d, mom20d + rsi14, macd, macd_signal, bb_upper, bb_lower, bb_position
행동 공간: logit 벡터 → softmax → 포트폴리오 비중 (합=1)
보상 함수 변형 (RewardVariant):
  R1_LOGRET : 로그 수익률만 (baseline)
  R2_SHARPE : 롤링 Sharpe ratio (위험 조정 수익률)
  R3_FULL   : 로그 수익률 - 리스크 집중도 페널티 - 최대낙폭 페널티

리스크 태그는 inject_risk_tags()로 외부 리서치 에이전트가 주입하며,
관측 공간을 통해 에이전트 행동에 자동으로 반영된다.
"""

import enum
from collections import deque

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

from .risk_state import RiskState, RISK_TAG_NAMES
from ..data.preprocessors import compute_features


class RewardVariant(str, enum.Enum):
    R1_LOGRET = "R1_LOGRET"  # 로그 수익률만 (baseline)
    R2_SHARPE = "R2_SHARPE"  # 롤링 Sharpe ratio
    R3_FULL   = "R3_FULL"    # 로그수익률 - 리스크 페널티 - 낙폭 페널티


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        prices: pd.DataFrame,
        risk_state: Optional[RiskState] = None,
        window_size: int = 20,
        transaction_cost: float = 0.001,
        risk_penalty_lambda: float = 0.5,
        drawdown_penalty_mu: float = 1.0,
        reward_variant: RewardVariant = RewardVariant.R3_FULL,
        sharpe_window: int = 60,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.prices = prices
        self.tickers = list(prices.columns)
        self.n_assets = len(self.tickers)
        self.n_tags = len(RISK_TAG_NAMES)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.risk_penalty_lambda = risk_penalty_lambda
        self.drawdown_penalty_mu = drawdown_penalty_mu
        self.reward_variant = RewardVariant(reward_variant)
        self.sharpe_window = sharpe_window
        self.render_mode = render_mode
        self._return_history: deque = deque(maxlen=sharpe_window)

        self.risk_state = risk_state if risk_state is not None else RiskState()

        # 비정상성 제거된 시장 피처 사전 계산
        self.features = compute_features(prices, window_size)
        self.valid_dates = self.features.index

        if len(self.valid_dates) == 0:
            raise ValueError(
                f"피처 계산 결과가 비어 있습니다. prices 데이터({len(prices)}행)를 확인하세요. "
                f"yfinance 다운로드 실패 또는 캐시 손상일 수 있습니다."
            )

        # 관측 공간: 시장특성(n_assets*11, 기본5+기술지표6) + 리스크태그(n_tags) + 현재비중(n_assets)
        n_obs = self.n_assets * 11 + self.n_tags + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )

        # 행동 공간: logit (softmax로 비중 변환)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self._current_step: int = 0
        self._current_weights: np.ndarray = np.ones(self.n_assets) / self.n_assets
        self._portfolio_value: float = 1.0
        self._peak_value: float = 1.0

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step = 0
        self._current_weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._portfolio_value = 1.0
        self._peak_value = 1.0
        self._return_history.clear()
        self.risk_state.reset()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        new_weights = self._softmax(action)

        date = self.valid_dates[self._current_step]
        price_idx = self.prices.index.get_loc(date)
        next_price_idx = price_idx + 1

        if next_price_idx >= len(self.prices):
            return self._get_obs(), 0.0, True, False, self._get_info()

        asset_returns = (
            self.prices.iloc[next_price_idx].values / self.prices.iloc[price_idx].values - 1
        )

        turnover = np.sum(np.abs(new_weights - self._current_weights))
        portfolio_return = float(np.dot(new_weights, asset_returns)) - self.transaction_cost * turnover
        self._return_history.append(portfolio_return)

        reward = self._compute_reward(portfolio_return, new_weights)

        self._portfolio_value *= (1 + portfolio_return)
        self._peak_value = max(self._peak_value, self._portfolio_value)
        self._current_weights = new_weights
        self.risk_state.step_decay()
        self._current_step += 1

        truncated = self._current_step >= len(self.valid_dates) - 1
        return self._get_obs(), reward, False, truncated, self._get_info()

    def inject_risk_tags(self, tags: list) -> None:
        """외부 리서치 에이전트에서 리스크 태그를 주입하는 인터페이스."""
        self.risk_state.update(tags)

    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        step = min(self._current_step, len(self.valid_dates) - 1)
        date = self.valid_dates[step]
        market = self.features.loc[date].values.astype(np.float32)
        risk = self.risk_state.to_array()
        return np.concatenate([market, risk, self._current_weights])

    def _compute_reward(self, portfolio_return: float, weights: np.ndarray) -> float:
        if self.reward_variant == RewardVariant.R1_LOGRET:
            return self._reward_logret(portfolio_return)
        if self.reward_variant == RewardVariant.R2_SHARPE:
            return self._reward_sharpe(portfolio_return)
        return self._reward_full(portfolio_return, weights)

    def _reward_logret(self, portfolio_return: float) -> float:
        """R1: 로그 수익률만 (baseline)."""
        return float(np.log1p(portfolio_return + 1e-8))

    def _reward_sharpe(self, portfolio_return: float) -> float:
        """R2: 롤링 Sharpe ratio — 위험 조정 수익률."""
        log_ret = np.log1p(portfolio_return + 1e-8)
        history = list(self._return_history)
        if len(history) < 2:
            return float(log_ret)
        mu = float(np.mean(history))
        sigma = float(np.std(history, ddof=1)) + 1e-8
        return float(mu / sigma)

    def _reward_full(self, portfolio_return: float, weights: np.ndarray) -> float:
        """R3: 로그 수익률 - 리스크 집중도 페널티 - 낙폭 페널티."""
        log_ret = np.log1p(portfolio_return + 1e-8)

        # Herfindahl 집중도 지수 × 집계 리스크 수준
        aggregate_risk = float(np.mean(self.risk_state.to_array()))
        concentration = float(np.sum(weights ** 2))
        risk_penalty = self.risk_penalty_lambda * aggregate_risk * concentration

        drawdown = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-8)
        drawdown_penalty = self.drawdown_penalty_mu * drawdown

        return float(log_ret - risk_penalty - drawdown_penalty)

    def _get_info(self) -> dict:
        drawdown = (self._peak_value - self._portfolio_value) / (self._peak_value + 1e-8)
        return {
            "portfolio_value": round(self._portfolio_value, 6),
            "drawdown": round(float(drawdown), 6),
            "risk_state": self.risk_state.to_array().tolist(),
            "weights": self._current_weights.tolist(),
            "step": self._current_step,
        }

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return (e_x / e_x.sum()).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            info = self._get_info()
            print(
                f"Step {info['step']:4d} | "
                f"Value: {info['portfolio_value']:.4f} | "
                f"Drawdown: {info['drawdown']:.2%} | "
                f"Risk: {np.mean(info['risk_state']):.3f}"
            )
