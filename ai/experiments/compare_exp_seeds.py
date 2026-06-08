"""
실험 2-3 비교 분석: 멀티 시드 앙상블 실험 결과 비교.

가설: 시드 수가 늘어날수록 std_sharpe(분산)가 줄어들고 성과가 안정화된다.

사용법:
  cd ai
  python experiments/compare_exp_seeds.py
  python experiments/compare_exp_seeds.py --timesteps 300000
  python experiments/compare_exp_seeds.py --timesteps 300000 --save
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULT_DIR = Path(__file__).parent / "results"
SEEDS_LIST = [1, 3, 5]
TRAIN_MONTHS, TEST_MONTHS = 24, 6


def _load(timesteps: int, n_seeds: int) -> dict | None:
    path = RESULT_DIR / f"walk_forward_tm{TRAIN_MONTHS}_tt{TEST_MONTHS}_ts{timesteps}_s{n_seeds}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summary_row(n_seeds: int, data: dict) -> dict:
    s = data["summary"]
    return {
        "n_seeds":       n_seeds,
        "mean_cagr":     round(s["mean_cagr"], 4),
        "std_cagr":      round(s["std_cagr"], 4),
        "mean_sharpe":   round(s["mean_sharpe"], 4),
        "std_sharpe":    round(s["std_sharpe"], 4),
        "mean_mdd":      round(s["mean_max_drawdown"], 4),
        "std_mdd":       round(s["std_max_drawdown"], 4),
        "n_folds":       s["n_folds"],
    }


def print_summary_table(rows: list[dict], timesteps: int) -> None:
    print("\n" + "=" * 78)
    print(f"  실험 2-3: 멀티 시드 앙상블 실험  (timesteps={timesteps:,})")
    print("=" * 78)
    header = f"{'n_seeds':>8} {'CAGR(μ)':>10} {'CAGR(σ)':>9} {'Sharpe(μ)':>10} {'Sharpe(σ)':>10} {'MDD(μ)':>8} {'MDD(σ)':>8}"
    print(header)
    print("-" * 78)
    for r in rows:
        print(
            f"{r['n_seeds']:>8} "
            f"{r['mean_cagr']:>10.4f} "
            f"{r['std_cagr']:>9.4f} "
            f"{r['mean_sharpe']:>10.4f} "
            f"{r['std_sharpe']:>10.4f} "
            f"{r['mean_mdd']:>8.4f} "
            f"{r['std_mdd']:>8.4f}"
        )
    print("=" * 78)

    # 분산 감소 여부 평가
    if len(rows) >= 2:
        print("\n  [분산 감소 분석]")
        base = rows[0]
        for r in rows[1:]:
            sharpe_diff = r["std_sharpe"] - base["std_sharpe"]
            direction = "감소 ✅" if sharpe_diff < 0 else "증가 ❌"
            print(
                f"  n_seeds={r['n_seeds']} vs {base['n_seeds']}: "
                f"std_sharpe {base['std_sharpe']:.4f} → {r['std_sharpe']:.4f} "
                f"({sharpe_diff:+.4f}, {direction})"
            )


def print_fold_table(all_data: dict[int, dict]) -> None:
    fold_keys: dict[str, dict] = {}
    for n_seeds, data in all_data.items():
        for fold in data["folds"]:
            key = fold["test_start"]
            if key not in fold_keys:
                fold_keys[key] = {"test_end": fold["test_end"]}
            fold_keys[key][n_seeds] = fold

    seeds_list = sorted(all_data.keys())
    header_vals = "  ".join(f"{'s='+str(s):>9}" for s in seeds_list)

    for metric, label in [("sharpe", "Sharpe"), ("cagr", "CAGR")]:
        print(f"\n{'폴드':<6} {'기간':<24}  {header_vals}  ({label})")
        print("-" * (36 + 12 * len(seeds_list)))
        for i, (start, entry) in enumerate(sorted(fold_keys.items())):
            period = f"{start} ~ {entry['test_end']}"
            values = []
            for s in seeds_list:
                if s in entry:
                    values.append(f"{entry[s][metric]:>9.4f}")
                else:
                    values.append(f"{'—':>9}")
            print(f"  {i:<4} {period:<24}  {'  '.join(values)}")


def save_outputs(rows: list[dict], all_data: dict[int, dict], timesteps: int, out_dir: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    csv_path = out_dir / f"exp23_seeds_ts{timesteps}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장: {csv_path}")

    seeds_vals = [r["n_seeds"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Sharpe(μ ± σ) — 핵심 지표
    ax = axes[0]
    mu  = [r["mean_sharpe"] for r in rows]
    std = [r["std_sharpe"]  for r in rows]
    ax.bar([str(s) for s in seeds_vals], mu, yerr=std, capsize=5,
           color=["#3498db", "#2ecc71", "#e74c3c"][:len(rows)])
    ax.set_title("Mean Sharpe (±1 std)", fontsize=10)
    ax.set_xlabel("n_seeds")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    # std_sharpe — 분산 감소 시각화
    ax = axes[1]
    ax.plot([str(s) for s in seeds_vals], std, marker="o", linewidth=2, color="#e67e22")
    ax.set_title("std_sharpe (낮을수록 안정적)", fontsize=10)
    ax.set_xlabel("n_seeds")
    ax.set_ylabel("std Sharpe")
    ax.grid(True, linestyle="--", alpha=0.4)

    # CAGR(μ ± σ)
    ax = axes[2]
    mu_c  = [r["mean_cagr"] for r in rows]
    std_c = [r["std_cagr"]  for r in rows]
    ax.bar([str(s) for s in seeds_vals], mu_c, yerr=std_c, capsize=5,
           color=["#3498db", "#2ecc71", "#e74c3c"][:len(rows)])
    ax.set_title("Mean CAGR (±1 std)", fontsize=10)
    ax.set_xlabel("n_seeds")
    ax.set_ylabel("CAGR")
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(f"실험 2-3: 멀티 시드 앙상블 실험 (ts={timesteps:,})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = out_dir / f"exp23_seeds_ts{timesteps}_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"그래프 저장: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000, help="비교할 훈련 스텝")
    parser.add_argument("--save", action="store_true", help="결과를 CSV·PNG로 저장")
    args = parser.parse_args()

    all_data: dict[int, dict] = {}
    for s in SEEDS_LIST:
        data = _load(args.timesteps, s)
        if data is None:
            print(f"[경고] n_seeds={s}, ts={args.timesteps:,} 결과 파일 없음 — 건너뜀")
        else:
            all_data[s] = data

    if not all_data:
        print("결과 파일이 하나도 없습니다. run_exp_seeds.sh를 먼저 실행하세요.")
        return

    rows = [_summary_row(s, data) for s, data in sorted(all_data.items())]
    print_summary_table(rows, args.timesteps)
    print_fold_table(all_data)

    if len(rows) > 1:
        best = max(rows, key=lambda r: r["mean_sharpe"])
        print(f"\n최고 Sharpe: n_seeds={best['n_seeds']}  Sharpe={best['mean_sharpe']:.4f}  std={best['std_sharpe']:.4f}")

    if args.save:
        save_outputs(rows, all_data, args.timesteps, RESULT_DIR / "exp23_seeds")


if __name__ == "__main__":
    main()
