"""
Walk-Forward 백테스트 실행 스크립트.

사용법:
  cd ai
  python experiments/walk_forward_experiment.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.backtest.walk_forward import WalkForwardBacktest, WalkForwardConfig

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
        end="2024-12-31",
    )
    print(f"데이터 shape: {prices.shape}  기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    wf_cfg = WalkForwardConfig(
        train_months=24,
        test_months=6,
        step_months=6,
        train_timesteps=30_000,
        window_size=env_cfg["window_size"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0005),
        max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.15),
    )

    backtest = WalkForwardBacktest(prices=prices, config=wf_cfg)
    result = backtest.run(verbose=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "config": {
            "train_months": wf_cfg.train_months,
            "test_months": wf_cfg.test_months,
            "step_months": wf_cfg.step_months,
            "train_timesteps": wf_cfg.train_timesteps,
        },
        "summary": {
            "n_folds": len(result.folds),
            "mean_cagr": result.mean_cagr,
            "std_cagr": result.std_cagr,
            "mean_sharpe": result.mean_sharpe,
            "std_sharpe": result.std_sharpe,
            "mean_max_drawdown": result.mean_max_drawdown,
            "std_max_drawdown": result.std_max_drawdown,
        },
        "folds": [
            {
                "fold_idx": fm.fold_idx,
                "test_start": fm.test_start,
                "test_end": fm.test_end,
                "n_train_bars": fm.n_train_bars,
                "n_test_bars": fm.n_test_bars,
                **fm.metrics.as_dict(),
            }
            for fm in result.folds
        ],
    }

    save_path = RESULT_DIR / "walk_forward_result.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=_numpy_default)

    print(f"\n결과 저장 완료: {save_path}")


if __name__ == "__main__":
    main()
