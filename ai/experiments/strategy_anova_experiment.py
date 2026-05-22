"""
전략 ANOVA 실험 스크립트.

DRL(PPO) vs MVO vs 동일가중(Equal-Weight) 세 전략을
Walk-Forward 폴드별 CAGR로 One-way ANOVA 비교.

사용법:
  cd ai
  python experiments/strategy_anova_experiment.py
  python experiments/strategy_anova_experiment.py --folds 6 --drl_timesteps 50000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.research.strategy_anova import (
    collect_strategy_returns,
    run_strategy_anova,
    report_strategy_anova,
)

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
RESULT_DIR  = Path(__file__).parent / "results"


def _numpy_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Strategy ANOVA: DRL vs MVO vs EqualWeight")
    parser.add_argument("--start",          type=str, default="2019-01-01",  help="데이터 시작일")
    parser.add_argument("--end",            type=str, default="2026-05-01",  help="데이터 종료일")
    parser.add_argument("--train_months",   type=int, default=24,  help="훈련 기간 (월)")
    parser.add_argument("--test_months",    type=int, default=6,   help="테스트 기간 (월)")
    parser.add_argument("--step_months",    type=int, default=6,   help="슬라이딩 스텝 (월)")
    parser.add_argument("--drl_timesteps",  type=int, default=30_000, help="폴드당 DRL 학습 스텝")
    parser.add_argument("--alpha",          type=float, default=0.05, help="ANOVA 유의 수준")
    parser.add_argument("--save_path",      type=str, default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    tickers = env_cfg["tickers"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(tickers, start=args.start, end=args.end)
    print(f"  shape={prices.shape}  기간={prices.index[0].date()} ~ {prices.index[-1].date()}")

    # ── 전략별 폴드 CAGR 수집 ──────────────────────────────────────
    print("\n[전략 CAGR 수집] DRL / MVO / EqualWeight")
    returns_by_strategy = collect_strategy_returns(
        prices=prices,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        drl_timesteps=args.drl_timesteps,
        window_size=env_cfg["window_size"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0005),
        max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.15),
        risk_free_rate=cfg.get("backtest", {}).get("risk_free_rate", 0.02),
        verbose=True,
    )

    # ── ANOVA 검정 ────────────────────────────────────────────────
    result = run_strategy_anova(
        returns_by_strategy=returns_by_strategy,
        alpha=args.alpha,
        metric_name="fold_cagr",
    )

    # ── 결과 출력 ─────────────────────────────────────────────────
    save_path = args.save_path or str(RESULT_DIR / "strategy_anova_result.json")
    report_strategy_anova(result, save_path=save_path)

    # ── 원시 데이터도 함께 저장 ────────────────────────────────────
    raw_path = RESULT_DIR / "strategy_anova_raw_returns.json"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(returns_by_strategy, f, ensure_ascii=False, indent=2, default=_numpy_default)
    print(f"  원시 수익률 저장: {raw_path}")


if __name__ == "__main__":
    main()
