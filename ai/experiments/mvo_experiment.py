"""
MVO (Mean-Variance Optimization) 실험 스크립트.

사용법:
  cd ai
  python experiments/mvo_experiment.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.backtest.mvo import MVO, MVOConfig, run_mvo_walk_forward
from src.backtest.walk_forward import WalkForwardConfig

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
RESULT_DIR = Path(__file__).parent / "results"


def _numpy_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(
        tickers=env_cfg["tickers"],
        start="2019-01-01",
        end="2026-05-01",
    )
    print(f"데이터 shape: {prices.shape}  기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    # ── 전체 기간 MVO 최적화 ─────────────────────────────────────
    print("\n[전체 기간 MVO 최적화]")
    for target in ("max_sharpe", "min_variance"):
        mvo = MVO(MVOConfig(target=target))
        mvo.fit(prices)
        w = mvo.get_weights()
        stats = mvo.portfolio_stats(w)
        print(f"\n  target={target}")
        print(f"  비중: { {t: round(float(v), 4) for t, v in zip(mvo.tickers_, w)} }")
        print(f"  기대수익률: {stats['return']:.2%}  변동성: {stats['volatility']:.2%}  "
              f"Sharpe: {stats['sharpe']:.3f}")

    # ── 효율적 프론티어 ──────────────────────────────────────────
    print("\n[효율적 프론티어 계산]")
    mvo_full = MVO(MVOConfig(n_frontier_points=30))
    mvo_full.fit(prices)
    frontier = mvo_full.efficient_frontier()
    print(frontier[["return", "volatility", "sharpe"]].to_string(index=False))

    # ── Walk-Forward 백테스트 ────────────────────────────────────
    print("\n[MVO Walk-Forward 백테스트]")
    wf_cfg = WalkForwardConfig(
        train_months=24,
        test_months=6,
        step_months=6,
        window_size=env_cfg["window_size"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0005),
    )
    mvo_cfg = MVOConfig(target="max_sharpe")
    result = run_mvo_walk_forward(prices, wf_cfg, mvo_cfg, verbose=True)

    # ── 결과 저장 ────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    frontier_path = RESULT_DIR / "mvo_frontier.json"
    frontier_records = frontier[["return", "volatility", "sharpe"]].to_dict(orient="records")
    with open(frontier_path, "w", encoding="utf-8") as f:
        json.dump(frontier_records, f, ensure_ascii=False, indent=2, default=_numpy_default)

    wf_path = RESULT_DIR / "mvo_walk_forward_result.json"
    out = {
        "summary": {
            "n_folds": len(result.folds),
            "mean_cagr": result.mean_cagr,
            "std_cagr": result.std_cagr,
            "mean_sharpe": result.mean_sharpe,
            "mean_sortino": result.mean_sortino,
            "mean_max_drawdown": result.mean_max_drawdown,
            "mean_var_95": result.mean_var_95,
            "mean_cvar_95": result.mean_cvar_95,
            "mean_alpha": result.mean_alpha,
            "mean_beta": result.mean_beta,
            "mean_information_ratio": result.mean_information_ratio,
        },
        "folds": [
            {"fold_idx": fm.fold_idx, "test_start": fm.test_start,
             "test_end": fm.test_end, **fm.metrics.as_dict()}
            for fm in result.folds
        ],
    }
    with open(wf_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=_numpy_default)

    print(f"\n효율적 프론티어 저장: {frontier_path}")
    print(f"Walk-Forward 결과 저장: {wf_path}")


if __name__ == "__main__":
    main()
