"""
포트폴리오 성과 지표 12개 계산 모듈.

지표 목록:
  수익률  : total_return, cagr
  위험    : volatility, max_drawdown, var_95, cvar_95
  위험조정: sharpe, sortino, calmar
  벤치마크: alpha, beta, information_ratio
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


@dataclass
class PerformanceMetrics:
    # ── 수익률 ──────────────────────────────────────────────────
    total_return: float       # 1.  누적 수익률
    cagr: float               # 2.  연환산 수익률 (CAGR)
    # ── 위험 ────────────────────────────────────────────────────
    volatility: float         # 3.  연환산 변동성
    max_drawdown: float       # 4.  최대 낙폭 (MDD)
    var_95: float             # 5.  VaR 95%  (손실 양수 표현)
    cvar_95: float            # 6.  CVaR 95% (Expected Shortfall)
    # ── 위험 조정 수익률 ─────────────────────────────────────────
    sharpe: float             # 7.  Sharpe ratio
    sortino: float            # 8.  Sortino ratio
    calmar: float             # 9.  Calmar ratio (CAGR / MDD)
    # ── 벤치마크 대비 ────────────────────────────────────────────
    alpha: float              # 10. Jensen's Alpha (연환산)
    beta: float               # 11. Beta (시장 민감도)
    information_ratio: float  # 12. Information Ratio

    def as_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "calmar": self.calmar,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
        }

    def summary(self) -> str:
        lines = [
            f"  Total Return : {self.total_return:+.2%}",
            f"  CAGR         : {self.cagr:+.2%}",
            f"  Volatility   : {self.volatility:.2%}",
            f"  Max Drawdown : {self.max_drawdown:.2%}",
            f"  VaR 95%      : {self.var_95:.2%}",
            f"  CVaR 95%     : {self.cvar_95:.2%}",
            f"  Sharpe       : {self.sharpe:+.3f}",
            f"  Sortino      : {self.sortino:+.3f}",
            f"  Calmar       : {self.calmar:+.3f}",
            f"  Alpha        : {self.alpha:+.4f}",
            f"  Beta         : {self.beta:+.4f}",
            f"  Info Ratio   : {self.information_ratio:+.3f}",
        ]
        return "\n".join(lines)


def compute_metrics(
    daily_returns: Union[list, np.ndarray],
    portfolio_values: Optional[Union[list, np.ndarray]] = None,
    benchmark_returns: Optional[Union[list, np.ndarray]] = None,
    n_bars: Optional[int] = None,
    trading_days: int = 252,
    risk_free_rate: float = 0.02,
) -> PerformanceMetrics:
    """
    12개 포트폴리오 성과 지표를 계산한다.

    Args:
        daily_returns    : 일별 수익률 시계열 (소수, 예: 0.01 = 1%)
        portfolio_values : 포트폴리오 가치 시계열 (MDD·total_return 정밀 계산용)
        benchmark_returns: 벤치마크 일별 수익률 (alpha·beta·IR 계산용; None → 0)
        n_bars           : 거래일 수 (CAGR 기간 기준; None → len(daily_returns))
        trading_days     : 연간 거래일 수 (기본 252)
        risk_free_rate   : 연간 무위험 수익률 (기본 0.02)
    """
    r = np.array(daily_returns, dtype=float)
    n = len(r)
    rf_daily = risk_free_rate / trading_days
    bars = n_bars if n_bars is not None else n

    # ── 1. 누적 수익률 ───────────────────────────────────────────
    if portfolio_values is not None and len(portfolio_values) >= 2:
        pv = np.array(portfolio_values, dtype=float)
        total_return = float(pv[-1] / pv[0] - 1) if pv[0] > 0 else 0.0
    elif n > 0:
        total_return = float(np.prod(1 + r) - 1)
    else:
        total_return = 0.0

    # ── 2. CAGR ──────────────────────────────────────────────────
    years = bars / trading_days
    cagr = float((1 + total_return) ** (1 / max(years, 1e-8)) - 1)

    # ── 3. 연환산 변동성 ─────────────────────────────────────────
    volatility = float(np.std(r, ddof=1) * math.sqrt(trading_days)) if n > 1 else 0.0

    # ── 4. 최대 낙폭 (MDD) ───────────────────────────────────────
    if portfolio_values is not None and len(portfolio_values) >= 2:
        pv_for_dd = np.array(portfolio_values, dtype=float)
    elif n > 0:
        pv_for_dd = np.cumprod(1 + r)
    else:
        pv_for_dd = np.array([1.0])
    peak = np.maximum.accumulate(pv_for_dd)
    dd = (peak - pv_for_dd) / (peak + 1e-10)
    max_drawdown = float(dd.max())

    # ── 5. VaR 95% ───────────────────────────────────────────────
    if n > 0:
        var_95 = float(-np.percentile(r, 5))
    else:
        var_95 = 0.0

    # ── 6. CVaR 95% (Expected Shortfall) ────────────────────────
    if n > 0:
        threshold = np.percentile(r, 5)
        tail = r[r <= threshold]
        cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else var_95
    else:
        cvar_95 = 0.0

    # ── 7. Sharpe ratio ──────────────────────────────────────────
    if n > 1:
        excess = r - rf_daily
        sigma = float(np.std(r, ddof=1))
        sharpe = float(np.mean(excess) / sigma * math.sqrt(trading_days)) if sigma > 1e-12 else 0.0
    else:
        sharpe = 0.0

    # ── 8. Sortino ratio ─────────────────────────────────────────
    if n > 1:
        excess = r - rf_daily
        downside = r[r < rf_daily]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
        sortino = float(np.mean(excess) / downside_std * math.sqrt(trading_days)) if downside_std > 1e-12 else 0.0
    else:
        sortino = 0.0

    # ── 9. Calmar ratio ──────────────────────────────────────────
    calmar = float(cagr / max_drawdown) if max_drawdown > 1e-8 else 0.0

    # ── 10–12. Alpha / Beta / Information Ratio ──────────────────
    if benchmark_returns is not None:
        bm = np.array(benchmark_returns, dtype=float)
        min_len = min(n, len(bm))
        if min_len > 1:
            r_a, bm_a = r[:min_len], bm[:min_len]
            bm_var = float(np.var(bm_a, ddof=1))
            if bm_var > 1e-12:
                beta = float(np.cov(r_a, bm_a, ddof=1)[0, 1] / bm_var)
            else:
                beta = 0.0
            # Jensen's Alpha: 일간 초과수익의 연환산
            alpha = float((np.mean(r_a) - beta * np.mean(bm_a)) * trading_days)
            active = r_a - bm_a
            active_std = float(np.std(active, ddof=1))
            information_ratio = (
                float(np.mean(active) / active_std * math.sqrt(trading_days))
                if active_std > 1e-12 else 0.0
            )
        else:
            alpha, beta, information_ratio = 0.0, 0.0, 0.0
    else:
        alpha, beta, information_ratio = 0.0, 0.0, 0.0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        volatility=volatility,
        max_drawdown=max_drawdown,
        var_95=var_95,
        cvar_95=cvar_95,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        alpha=alpha,
        beta=beta,
        information_ratio=information_ratio,
    )
