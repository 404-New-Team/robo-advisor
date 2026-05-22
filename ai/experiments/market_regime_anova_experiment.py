"""
시장 국면별 성과 비교 실험 — ANOVA 검증 3.

Two-way ANOVA: 전략(DRL / MVO / EqualWeight) × 국면(Bull / Bear / Sideways)

사용법:
  cd ai
  python experiments/market_regime_anova_experiment.py
  python experiments/market_regime_anova_experiment.py --drl_timesteps 5000
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.research.market_regime_anova import (
    collect_regime_returns,
    run_twoway_anova,
    report_twoway_anova,
)

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
RESULT_DIR  = Path(__file__).parent / "results"


def _numpy_default(obj):
    if isinstance(obj, float) and (obj != obj):   # NaN
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(description="Two-way ANOVA: 전략 × 시장 국면")
    parser.add_argument("--start",          type=str,   default="2019-01-01")
    parser.add_argument("--end",            type=str,   default="2026-05-01")
    parser.add_argument("--train_months",   type=int,   default=24)
    parser.add_argument("--test_months",    type=int,   default=6)
    parser.add_argument("--step_months",    type=int,   default=6)
    parser.add_argument("--drl_timesteps",  type=int,   default=30_000)
    parser.add_argument("--threshold_bull", type=float, default=0.10,
                        help="Bull 국면 최소 CAGR (연환산)")
    parser.add_argument("--threshold_bear", type=float, default=-0.10,
                        help="Bear 국면 최대 CAGR (연환산)")
    parser.add_argument("--alpha",          type=float, default=0.05)
    parser.add_argument("--save_path",      type=str,   default=None)
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env_cfg = cfg["environment"]
    tickers = env_cfg["tickers"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(tickers, start=args.start, end=args.end)
    print(f"  shape={prices.shape}  기간={prices.index[0].date()} ~ {prices.index[-1].date()}")

    # ── 폴드별 전략 CAGR + 국면 레이블 수집 ──────────────────────
    print("\n[국면별 전략 수익률 수집]")
    records = collect_regime_returns(
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
        threshold_bull=args.threshold_bull,
        threshold_bear=args.threshold_bear,
        verbose=True,
    )

    if len(records) < 6:
        print(f"\n[경고] 관측치가 {len(records)}개로 부족합니다. ANOVA 신뢰도가 낮을 수 있습니다.")

    # ── Two-way ANOVA ──────────────────────────────────────────
    result = run_twoway_anova(
        records=records,
        alpha=args.alpha,
        metric_name="fold_cagr",
    )

    # ── 결과 출력 + 저장 ───────────────────────────────────────
    save_path = args.save_path or str(RESULT_DIR / "market_regime_anova_result.json")
    report_twoway_anova(result, save_path=save_path)

    # ── 원시 데이터 저장 ───────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULT_DIR / "market_regime_anova_records.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=_numpy_default)
    print(f"  원시 데이터 저장: {raw_path}")


if __name__ == "__main__":
    main()
