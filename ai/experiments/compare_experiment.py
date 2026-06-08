"""
DRL / MVO / EqualWeight 세 전략을 동일 폴드 날짜로 비교하는 실험.

결과는 experiments/results/comparison_tm{train}_tt{test}.json 에 저장되며,
/ai/backtest API가 이 파일을 우선 읽어 대시보드에 표시한다.

사용법:
  cd ai
  python experiments/compare_experiment.py
  python experiments/compare_experiment.py --train_months 48 --test_months 12
  python experiments/compare_experiment.py --train_months 48 --test_months 12 --n_envs 4
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.market_data import fetch_prices
from src.research.strategy_anova import collect_strategy_returns

CONFIG_PATH = Path(__file__).parent.parent / "src" / "config" / "settings.yaml"
RESULT_DIR  = Path(__file__).parent / "results"


def _numpy_default(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def _make_summary(cagr_list: list) -> dict:
    arr = [v for v in cagr_list if v is not None]
    if not arr:
        return {}
    return {
        "mean_cagr":             float(np.mean(arr)),
        "std_cagr":              float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "mean_sharpe":           0.0,   # fold-level sharpe는 별도 계산 필요 시 확장
        "mean_max_drawdown":     0.0,
        "mean_sortino":          0.0,
        "mean_calmar":           0.0,
        "mean_volatility":       0.0,
        "mean_var_95":           0.0,
        "mean_cvar_95":          0.0,
        "mean_alpha":            0.0,
        "mean_beta":             0.0,
        "mean_information_ratio":0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="DRL vs MVO vs EW 동일 폴드 비교")
    parser.add_argument("--start",         type=str, default="2019-01-01")
    parser.add_argument("--end",           type=str, default="2024-12-31")
    parser.add_argument("--train_months",  type=int, default=24)
    parser.add_argument("--test_months",   type=int, default=6)
    parser.add_argument("--step_months",   type=int, default=6)
    parser.add_argument("--drl_timesteps", type=int, default=150_000)
    parser.add_argument("--n_seeds",       type=int, default=1)
    parser.add_argument("--n_envs",        type=int, default=4)
    parser.add_argument("--n_jobs",        type=int, default=1)
    args = parser.parse_args()

    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg["environment"]

    print("시장 데이터 로딩 중...")
    prices = fetch_prices(env_cfg["tickers"], start=args.start, end=args.end)
    print(f"  shape={prices.shape}  {prices.index[0].date()} ~ {prices.index[-1].date()}")

    print(f"\n[전략 비교] train={args.train_months}mo / test={args.test_months}mo / step={args.step_months}mo")
    print(f"           DRL: {args.drl_timesteps:,} steps × {args.n_seeds} seed(s) × {args.n_envs} envs")

    returns = collect_strategy_returns(
        prices=prices,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.step_months,
        drl_timesteps=args.drl_timesteps,
        window_size=env_cfg["window_size"],
        transaction_cost=env_cfg["transaction_cost"],
        slippage=env_cfg.get("slippage", 0.0005),
        max_drawdown_threshold=env_cfg.get("max_drawdown_threshold", 0.25),
        risk_free_rate=cfg.get("backtest", {}).get("risk_free_rate", 0.02),
        n_seeds=args.n_seeds,
        n_jobs=args.n_jobs,
        verbose=True,
    )

    # ── 결과를 API 호환 형식으로 변환 ─────────────────────────────────────
    def _to_api_format(strategy_key: str) -> dict:
        cagr_list = returns.get(strategy_key, [])
        folds_meta = returns.get("_folds_meta", [])  # fold 날짜 메타 (있으면)
        folds = []
        for i, cagr in enumerate(cagr_list):
            meta = folds_meta[i] if i < len(folds_meta) else {}
            folds.append({
                "test_start": meta.get("test_start", ""),
                "test_end":   meta.get("test_end", ""),
                "cagr":       float(cagr) if cagr is not None else None,
                "total_return": float(cagr) if cagr is not None else None,
                "sharpe":     None,
            })
        return {
            "summary": _make_summary(cagr_list),
            "folds":   folds,
        }

    output = {
        "config": {
            "train_months":  args.train_months,
            "test_months":   args.test_months,
            "step_months":   args.step_months,
            "drl_timesteps": args.drl_timesteps,
            "n_seeds":       args.n_seeds,
            "n_envs":        args.n_envs,
            "start":         args.start,
            "end":           args.end,
        },
        "DRL":         _to_api_format("DRL"),
        "MVO":         _to_api_format("MVO"),
        "EqualWeight": _to_api_format("EqualWeight"),
        "_raw_cagr":   {k: returns[k] for k in ["DRL", "MVO", "EqualWeight"] if k in returns},
    }

    # 설정별 고유 파일명으로 저장
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"comparison_tm{args.train_months}_tt{args.test_months}.json"
    save_path = RESULT_DIR / fname
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=_numpy_default)
    print(f"\n비교 결과 저장: {save_path}")

    # 요약 출력
    print("\n" + "="*55)
    print(f"  전략 비교 요약  (train={args.train_months}mo / test={args.test_months}mo)")
    print("="*55)
    for key in ["DRL", "MVO", "EqualWeight"]:
        cagrs = returns.get(key, [])
        if cagrs:
            mean = np.mean(cagrs)
            std  = np.std(cagrs, ddof=1) if len(cagrs) > 1 else 0.0
            print(f"  {key:12s}: CAGR {mean:+.2%}  ± {std:.2%}  (n={len(cagrs)})")
    print("="*55)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
