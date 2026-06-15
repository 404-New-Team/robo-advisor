"""
벤치마크 전략 Walk-Forward 사전 계산 스크립트.

DRL과 동일한 기간/폴드 구조로 MVO와 Equal Weight를 평가하여 캐시 파일을 저장한다.
API가 이 파일을 읽어 대시보드에 표시하므로, DRL과 공정하게 비교된다.

저장 결과:
  experiments/results/mvo_walk_forward_result.json
  experiments/results/ew_walk_forward_result.json

사용법:
  cd ai
  python experiments/run_benchmark_eval.py
  python experiments/run_benchmark_eval.py --start 2017-01-01 --end 2025-12-31
"""

import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.backtest.metrics import compute_metrics
from src.backtest.mvo import MVO, MVOConfig, run_mvo_walk_forward, _build_fold_dates
from src.backtest.walk_forward import WalkForwardConfig

import argparse

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
RESULT_DIR = Path(__file__).parent / "results"


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        return default if f != f else f
    except Exception:
        return default


def _wf_result_to_cache(result) -> dict:
    folds = []
    for fm in result.folds:
        m = fm.metrics
        folds.append({
            "test_start": fm.test_start,
            "test_end":   fm.test_end,
            **m.as_dict(),
        })
    return {
        "summary": {
            "mean_cagr":               result.mean_cagr,
            "mean_sharpe":             result.mean_sharpe,
            "mean_sortino":            result.mean_sortino,
            "mean_calmar":             result.mean_calmar,
            "mean_max_drawdown":       result.mean_max_drawdown,
            "mean_volatility":         float(np.mean([f.metrics.volatility for f in result.folds])) if result.folds else 0.0,
            "mean_var_95":             result.mean_var_95,
            "mean_cvar_95":            result.mean_cvar_95,
            "mean_alpha":              result.mean_alpha,
            "mean_beta":               result.mean_beta,
            "mean_information_ratio":  result.mean_information_ratio,
        },
        "folds": folds,
    }


def run_equal_weight(prices, cfg: WalkForwardConfig) -> dict:
    n = prices.shape[1]
    weights = np.ones(n) / n
    folds_dates = _build_fold_dates(prices, cfg)

    fold_data, fold_metrics_list = [], []
    for _, (_, _, test_start, test_end) in enumerate(folds_dates):
        test_prices = prices.loc[(prices.index >= test_start) & (prices.index <= test_end)]
        test_rets = test_prices.pct_change().dropna()
        chunk = (test_rets.values * weights).sum(axis=1)
        if len(chunk) < 2:
            continue
        cpv = np.insert(np.cumprod(1 + chunk), 0, 1.0)
        cm = compute_metrics(daily_returns=chunk.tolist(), portfolio_values=cpv.tolist())
        fold_metrics_list.append(cm)
        fold_data.append({
            "test_start": str(test_start.date() if hasattr(test_start, "date") else test_start),
            "test_end":   str(test_end.date() if hasattr(test_end, "date") else test_end),
            **cm.as_dict(),
        })

    def _mean(attr):
        vals = [getattr(m, attr, 0.0) for m in fold_metrics_list]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "summary": {
            "mean_cagr":              _mean("cagr"),
            "mean_sharpe":            _mean("sharpe"),
            "mean_sortino":           _mean("sortino"),
            "mean_calmar":            _mean("calmar"),
            "mean_max_drawdown":      _mean("max_drawdown"),
            "mean_volatility":        _mean("volatility"),
            "mean_var_95":            _mean("var_95"),
            "mean_cvar_95":           _mean("cvar_95"),
            "mean_alpha":             _mean("alpha"),
            "mean_beta":              _mean("beta"),
            "mean_information_ratio": _mean("information_ratio"),
        },
        "folds": fold_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",         type=str, default="2017-01-01")
    parser.add_argument("--end",           type=str, default="2025-12-31")
    parser.add_argument("--train_months",  type=int, default=24)
    parser.add_argument("--test_months",   type=int, default=6)
    parser.add_argument("--step_months",   type=int, default=6)
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg_yaml = yaml.safe_load(f)
    env_cfg = cfg_yaml["environment"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(
        tickers=env_cfg["tickers"],
        start=args.start,
        end=args.end,
    )
    print(f"데이터 shape: {prices.shape}  기간: {prices.index[0].date()} ~ {prices.index[-1].date()}")

    wf_cfg = WalkForwardConfig(
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    # ── MVO ──────────────────────────────────────────────────────────────────
    print(f"\n[1/2] MVO Walk-Forward 계산 중 ({args.train_months}mo train / {args.test_months}mo test)...")
    mvo_result = run_mvo_walk_forward(prices, wf_cfg, MVOConfig(), verbose=True)
    mvo_cache = _wf_result_to_cache(mvo_result)
    mvo_path = RESULT_DIR / "mvo_walk_forward_result.json"
    with open(mvo_path, "w", encoding="utf-8") as f:
        json.dump(mvo_cache, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {mvo_path}")
    print(f"  Sharpe={mvo_cache['summary']['mean_sharpe']:.4f}  CAGR={mvo_cache['summary']['mean_cagr']:.4f}")

    # ── Equal Weight ─────────────────────────────────────────────────────────
    print(f"\n[2/2] Equal Weight Walk-Forward 계산 중...")
    ew_cache = run_equal_weight(prices, wf_cfg)
    ew_path = RESULT_DIR / "ew_walk_forward_result.json"
    with open(ew_path, "w", encoding="utf-8") as f:
        json.dump(ew_cache, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료: {ew_path}")
    print(f"  Sharpe={ew_cache['summary']['mean_sharpe']:.4f}  CAGR={ew_cache['summary']['mean_cagr']:.4f}")

    print("\n=== 전략 비교 요약 ===")
    print(f"  {'전략':<14} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8}")
    print(f"  {'-'*40}")
    for name, cache in [("MVO", mvo_cache), ("Equal Weight", ew_cache)]:
        s = cache["summary"]
        print(f"  {name:<14} {s['mean_sharpe']:>8.4f} {s['mean_cagr']:>8.4f} {s['mean_max_drawdown']:>8.4f}")
    print("\n완료. API 재시작 없이 대시보드에 즉시 반영됩니다.")


if __name__ == "__main__":
    main()
