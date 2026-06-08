"""
실험 2-4 비교 분석: PPO Learning Rate 튜닝 실험 결과 비교.

가설: 학습률에 따라 수렴 속도와 최종 성과가 달라진다.
      너무 작으면 과소적합, 너무 크면 불안정 / 발산.

사용법:
  cd ai
  python experiments/compare_exp_lr.py
  python experiments/compare_exp_lr.py --timesteps 300000
  python experiments/compare_exp_lr.py --timesteps 300000 --save
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

RESULT_DIR = Path(__file__).parent / "results"
TRAIN_MONTHS, TEST_MONTHS = 24, 6
N_SEEDS = 1

LR_LIST = [1e-4, 3e-4, 1e-3]
_DEFAULT_LR = 3e-4


def _lr_tag(lr: float) -> str:
    s = f"{lr:.0e}"
    s = re.sub(r"e([+-])0*(\d+)", lambda m: f"e{m.group(1)}{m.group(2)}", s)
    return f"lr{s}"


def _load(timesteps: int, lr: float) -> dict | None:
    base = f"walk_forward_tm{TRAIN_MONTHS}_tt{TEST_MONTHS}_ts{timesteps}_s{N_SEEDS}"
    if lr != _DEFAULT_LR:
        base += f"_{_lr_tag(lr)}"
    path = RESULT_DIR / (base + ".json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _summary_row(lr: float, data: dict) -> dict:
    s = data["summary"]
    return {
        "learning_rate": lr,
        "lr_label":      _lr_tag(lr),
        "mean_cagr":     round(s["mean_cagr"],         4),
        "std_cagr":      round(s["std_cagr"],          4),
        "mean_sharpe":   round(s["mean_sharpe"],       4),
        "std_sharpe":    round(s["std_sharpe"],        4),
        "mean_mdd":      round(s["mean_max_drawdown"], 4),
        "std_mdd":       round(s["std_max_drawdown"],  4),
        "n_folds":       s["n_folds"],
    }


def print_summary_table(rows: list[dict], timesteps: int) -> None:
    print("\n" + "=" * 80)
    print(f"  실험 2-4: PPO Learning Rate 튜닝  (timesteps={timesteps:,}, n_seeds={N_SEEDS})")
    print("=" * 80)
    header = (
        f"{'lr':>10} {'CAGR(μ)':>10} {'CAGR(σ)':>9} "
        f"{'Sharpe(μ)':>10} {'Sharpe(σ)':>10} {'MDD(μ)':>8} {'MDD(σ)':>8}"
    )
    print(header)
    print("-" * 80)
    for r in rows:
        print(
            f"{r['lr_label']:>10} "
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
        print(f"\n  최고 Sharpe: lr={best_sharpe['lr_label']}  "
              f"Sharpe={best_sharpe['mean_sharpe']:.4f}  std={best_sharpe['std_sharpe']:.4f}")
        print(f"  최고 CAGR  : lr={best_cagr['lr_label']}  "
              f"CAGR={best_cagr['mean_cagr']:.4f}")

    # 기준(3e-4) 대비 변화
    base = next((r for r in rows if r["learning_rate"] == _DEFAULT_LR), None)
    if base and len(rows) > 1:
        print("\n  [기준 lr=3e-4 대비 Sharpe 변화]")
        for r in rows:
            if r["learning_rate"] == _DEFAULT_LR:
                continue
            diff = r["mean_sharpe"] - base["mean_sharpe"]
            sign = "+" if diff >= 0 else ""
            mark = "✅" if diff > 0 else "❌"
            print(
                f"  lr={r['lr_label']:>6}: "
                f"{base['mean_sharpe']:.4f} → {r['mean_sharpe']:.4f} "
                f"({sign}{diff:.4f}) {mark}"
            )


def print_fold_table(all_data: dict[float, dict]) -> None:
    fold_keys: dict[str, dict] = {}
    for lr, data in all_data.items():
        for fold in data["folds"]:
            key = fold["test_start"]
            if key not in fold_keys:
                fold_keys[key] = {"test_end": fold["test_end"]}
            fold_keys[key][lr] = fold

    lr_list = sorted(all_data.keys())
    header_vals = "  ".join(f"{_lr_tag(lr):>9}" for lr in lr_list)

    for metric, label in [("sharpe", "Sharpe"), ("cagr", "CAGR")]:
        print(f"\n{'폴드':<6} {'기간':<24}  {header_vals}  ({label})")
        print("-" * (36 + 12 * len(lr_list)))
        for i, (start, entry) in enumerate(sorted(fold_keys.items())):
            period = f"{start} ~ {entry['test_end']}"
            values = []
            for lr in lr_list:
                if lr in entry:
                    values.append(f"{entry[lr][metric]:>9.4f}")
                else:
                    values.append(f"{'—':>9}")
            print(f"  {i:<4} {period:<24}  {'  '.join(values)}")


def save_outputs(rows: list[dict], all_data: dict[float, dict], timesteps: int, out_dir: Path) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"exp24_lr_ts{timesteps}_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "learning_rate"])
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items() if k != "learning_rate"} for r in rows])
    print(f"CSV 저장: {csv_path}")

    labels = [r["lr_label"] for r in rows]
    colors = ["#3498db", "#2ecc71", "#e74c3c"][: len(rows)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (mu_k, std_k, title, ylabel) in zip(axes, [
        ("mean_sharpe", "std_sharpe", "Mean Sharpe Ratio (±1 std)", "Sharpe Ratio"),
        ("mean_cagr",   "std_cagr",   "Mean CAGR (±1 std)",         "CAGR"),
        ("mean_mdd",    "std_mdd",    "Mean MDD (낮을수록 좋음)",    "MDD"),
    ]):
        mu  = [r[mu_k]  for r in rows]
        std = [r[std_k] for r in rows]
        ax.bar(labels, mu, yerr=std, capsize=5, color=colors)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

    fig.suptitle(f"실험 2-4: PPO Learning Rate 튜닝 (ts={timesteps:,})", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plot_path = out_dir / f"exp24_lr_ts{timesteps}_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"그래프 저장: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=150_000, help="비교할 훈련 스텝")
    parser.add_argument("--save", action="store_true", help="결과를 CSV·PNG로 저장")
    args = parser.parse_args()

    all_data: dict[float, dict] = {}
    for lr in LR_LIST:
        data = _load(args.timesteps, lr)
        if data is None:
            print(f"[경고] lr={_lr_tag(lr)}, ts={args.timesteps:,} 결과 파일 없음 — 건너뜀")
        else:
            all_data[lr] = data

    if not all_data:
        print("결과 파일이 하나도 없습니다. run_exp_lr.sh를 먼저 실행하세요.")
        return

    rows = [_summary_row(lr, data) for lr, data in sorted(all_data.items())]
    print_summary_table(rows, args.timesteps)
    print_fold_table(all_data)

    if args.save:
        save_outputs(rows, all_data, args.timesteps, RESULT_DIR / "exp24_lr")


if __name__ == "__main__":
    main()
