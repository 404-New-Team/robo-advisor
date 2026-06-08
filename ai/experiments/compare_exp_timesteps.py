"""
실험 2-2 비교 분석: 훈련 스텝 증가 실험 결과 비교.

사용법:
  cd ai
  python experiments/compare_exp_timesteps.py
  python experiments/compare_exp_timesteps.py --save  # PNG/CSV 저장
"""

import argparse
import json
from pathlib import Path

import numpy as np

RESULT_DIR = Path(__file__).parent / "results"
TIMESTEPS = [30_000, 150_000, 200_000, 300_000, 500_000]
TRAIN_MONTHS, TEST_MONTHS = 24, 6


def _load(ts: int) -> dict | None:
    path = RESULT_DIR / f"walk_forward_tm{TRAIN_MONTHS}_tt{TEST_MONTHS}_ts{ts}_s1.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summary_row(ts: int, data: dict) -> dict:
    s = data["summary"]
    return {
        "timesteps": ts,
        "mean_cagr":         round(s["mean_cagr"], 4),
        "std_cagr":          round(s["std_cagr"], 4),
        "mean_sharpe":       round(s["mean_sharpe"], 4),
        "std_sharpe":        round(s["std_sharpe"], 4),
        "mean_mdd":          round(s["mean_max_drawdown"], 4),
        "std_mdd":           round(s["std_max_drawdown"], 4),
        "n_folds":           s["n_folds"],
    }


def print_summary_table(rows: list[dict]) -> None:
    header = f"{'timesteps':>10} {'CAGR(μ)':>10} {'CAGR(σ)':>9} {'Sharpe(μ)':>10} {'Sharpe(σ)':>10} {'MDD(μ)':>8} {'MDD(σ)':>8} {'folds':>6}"
    print("\n" + "=" * 75)
    print("  실험 2-2: 훈련 스텝 증가 실험 요약")
    print("=" * 75)
    print(header)
    print("-" * 75)
    for r in rows:
        print(
            f"{r['timesteps']:>10,} "
            f"{r['mean_cagr']:>10.4f} "
            f"{r['std_cagr']:>9.4f} "
            f"{r['mean_sharpe']:>10.4f} "
            f"{r['std_sharpe']:>10.4f} "
            f"{r['mean_mdd']:>8.4f} "
            f"{r['std_mdd']:>8.4f} "
            f"{r['n_folds']:>6}"
        )
    print("=" * 75)


def print_fold_table(all_data: dict[int, dict]) -> None:
    # 공통 폴드 기간 수집
    fold_keys: dict[str, dict] = {}  # test_start → {ts: metrics}
    for ts, data in all_data.items():
        for fold in data["folds"]:
            key = fold["test_start"]
            if key not in fold_keys:
                fold_keys[key] = {"test_end": fold["test_end"]}
            fold_keys[key][ts] = fold

    ts_list = sorted(all_data.keys())
    ts_header = "  ".join(f"{ts:>9,}" for ts in ts_list)
    print(f"\n{'폴드':<6} {'기간':<24}  " + ts_header + "  (Sharpe)")
    print("-" * (36 + 13 * len(ts_list)))
    for i, (start, entry) in enumerate(sorted(fold_keys.items())):
        period = f"{start} ~ {entry['test_end']}"
        values = []
        for ts in ts_list:
            if ts in entry:
                values.append(f"{entry[ts]['sharpe']:>9.4f}")
            else:
                values.append(f"{'—':>9}")
        print(f"  {i:<4} {period:<24}  {'  '.join(values)}")

    print()
    print(f"{'':6} {'':24}  " + ts_header + "  (CAGR)")
    print("-" * (36 + 13 * len(ts_list)))
    for i, (start, entry) in enumerate(sorted(fold_keys.items())):
        period = f"{start} ~ {entry['test_end']}"
        values = []
        for ts in ts_list:
            if ts in entry:
                values.append(f"{entry[ts]['cagr']:>9.4f}")
            else:
                values.append(f"{'—':>9}")
        print(f"  {i:<4} {period:<24}  {'  '.join(values)}")


def save_outputs(rows: list[dict], all_data: dict[int, dict], out_dir: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    csv_path = out_dir / "exp22_timesteps_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV 저장: {csv_path}")

    # 꺾은선 그래프: Sharpe / CAGR / MDD vs timesteps
    ts_vals = [r["timesteps"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("mean_sharpe", "std_sharpe", "Mean Sharpe Ratio"),
        ("mean_cagr",   "std_cagr",   "Mean CAGR"),
        ("mean_mdd",    "std_mdd",    "Mean MDD (낮을수록 좋음)"),
    ]
    for ax, (mu_key, std_key, title) in zip(axes, metrics):
        mu  = [r[mu_key]  for r in rows]
        std = [r[std_key] for r in rows]
        ax.errorbar(ts_vals, mu, yerr=std, marker="o", capsize=4, linewidth=1.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training Timesteps")
        ax.set_xscale("log")
        ax.set_xticks(ts_vals)
        ax.set_xticklabels([f"{t//1000}K" for t in ts_vals])
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("실험 2-2: 훈련 스텝 증가 실험", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = out_dir / "exp22_timesteps_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"그래프 저장: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="결과를 CSV·PNG로 저장")
    args = parser.parse_args()

    all_data: dict[int, dict] = {}
    for ts in TIMESTEPS:
        data = _load(ts)
        if data is None:
            print(f"[경고] ts={ts:,} 결과 파일 없음 — 건너뜀")
        else:
            all_data[ts] = data

    if not all_data:
        print("결과 파일이 하나도 없습니다. run_exp_timesteps.sh를 먼저 실행하세요.")
        return

    rows = [_summary_row(ts, data) for ts, data in sorted(all_data.items())]
    print_summary_table(rows)
    print_fold_table(all_data)

    if len(rows) > 1:
        best = max(rows, key=lambda r: r["mean_sharpe"])
        print(f"\n최고 Sharpe: ts={best['timesteps']:,}  Sharpe={best['mean_sharpe']:.4f}  CAGR={best['mean_cagr']:.4f}  MDD={best['mean_mdd']:.4f}")

    if args.save:
        save_outputs(rows, all_data, RESULT_DIR / "exp22_timesteps")


if __name__ == "__main__":
    main()
