"""
실험 2-5 비교 분석: 보상 함수 Lambda 탐색 실험 결과 비교.

가설: Lambda가 클수록 MDD↓ / CAGR↓ 트레이드오프 곡선이 존재한다.

사용법:
  cd ai
  python experiments/compare_exp_lambda.py
  python experiments/compare_exp_lambda.py --timesteps 300000
"""

import argparse
import json
from pathlib import Path

RESULT_DIR = Path(__file__).parent / "results"
TRAIN_MONTHS, TEST_MONTHS = 24, 6
N_SEEDS = 1

LAMBDA_LIST = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
_DEFAULT_LAMBDA = 0.1


def _lambda_tag(lam: float) -> str:
    return "l" + str(lam).replace(".", "p")


def _load(timesteps: int, lam: float) -> dict | None:
    base = f"walk_forward_tm{TRAIN_MONTHS}_tt{TEST_MONTHS}_ts{timesteps}_s{N_SEEDS}"
    if lam != _DEFAULT_LAMBDA:
        base += f"_{_lambda_tag(lam)}"
    path = RESULT_DIR / (base + ".json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summary_row(lam: float, data: dict) -> dict:
    s = data["summary"]
    return {
        "lambda":      lam,
        "label":       _lambda_tag(lam),
        "mean_cagr":   round(s["mean_cagr"],         4),
        "std_cagr":    round(s["std_cagr"],          4),
        "mean_sharpe": round(s["mean_sharpe"],       4),
        "std_sharpe":  round(s["std_sharpe"],        4),
        "mean_mdd":    round(s["mean_max_drawdown"], 4),
        "std_mdd":     round(s["std_max_drawdown"],  4),
        "n_folds":     s["n_folds"],
    }


def print_summary_table(rows: list[dict], timesteps: int) -> None:
    print("\n" + "=" * 80)
    print(f"  실험 2-5: 보상 함수 Lambda 탐색  (timesteps={timesteps:,}, n_seeds={N_SEEDS})")
    print("=" * 80)
    header = (
        f"{'lambda':>10} {'CAGR(μ)':>10} {'CAGR(σ)':>9} "
        f"{'Sharpe(μ)':>10} {'Sharpe(σ)':>10} {'MDD(μ)':>8} {'MDD(σ)':>8}"
    )
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['label']:>10} "
            f"{r['mean_cagr']:>10.4f} "
            f"{r['std_cagr']:>9.4f} "
            f"{r['mean_sharpe']:>10.4f} "
            f"{r['std_sharpe']:>10.4f} "
            f"{r['mean_mdd']:>8.4f} "
            f"{r['std_mdd']:>8.4f}"
        )
    print("=" * 80)

    if len(rows) >= 2:
        best_sharpe = max(rows, key=lambda r: r["mean_sharpe"])
        best_cagr   = max(rows, key=lambda r: r["mean_cagr"])
        best_mdd    = min(rows, key=lambda r: r["mean_mdd"])
        print(f"\n  최고 Sharpe: lambda={best_sharpe['label']}  "
              f"Sharpe={best_sharpe['mean_sharpe']:.4f}  std={best_sharpe['std_sharpe']:.4f}")
        print(f"  최고 CAGR  : lambda={best_cagr['label']}  "
              f"CAGR={best_cagr['mean_cagr']:.4f}")
        print(f"  최저 MDD   : lambda={best_mdd['label']}  "
              f"MDD={best_mdd['mean_mdd']:.4f}")


def print_fold_table(all_data: dict[float, dict]) -> None:
    fold_keys: dict[str, dict] = {}
    for lam, data in all_data.items():
        for fold in data["folds"]:
            key = fold["test_start"]
            if key not in fold_keys:
                fold_keys[key] = {"test_end": fold["test_end"]}
            fold_keys[key][lam] = fold

    lam_list = sorted(all_data.keys())
    header_vals = "  ".join(f"{_lambda_tag(lam):>9}" for lam in lam_list)

    for metric, label in [("sharpe", "Sharpe"), ("cagr", "CAGR")]:
        print(f"\n{'폴드':<6} {'기간':<24}  {header_vals}  ({label})")
        print("-" * (36 + 12 * len(lam_list)))
        for i, (start, entry) in enumerate(sorted(fold_keys.items())):
            period = f"{start} ~ {entry['test_end']}"
            values = []
            for lam in lam_list:
                if lam in entry:
                    values.append(f"{entry[lam][metric]:>9.4f}")
                else:
                    values.append(f"{'—':>9}")
            print(f"  {i:<4} {period:<24}  {'  '.join(values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=225_000)
    args = parser.parse_args()

    all_data: dict[float, dict] = {}
    for lam in LAMBDA_LIST:
        data = _load(args.timesteps, lam)
        if data is None:
            print(f"[경고] lambda={_lambda_tag(lam)}, ts={args.timesteps:,} 결과 파일 없음 — 건너뜀")
        else:
            all_data[lam] = data

    if not all_data:
        print("결과 파일이 없습니다. run_exp_lambda.sh를 먼저 실행하세요.")
        return

    rows = [_summary_row(lam, data) for lam, data in sorted(all_data.items())]
    print_summary_table(rows, args.timesteps)
    print_fold_table(all_data)


if __name__ == "__main__":
    main()
