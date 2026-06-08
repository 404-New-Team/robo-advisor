"""
Walk-Forward 실험 결과 비교 요약.
experiments/results/ 에서 walk_forward_tm*.json 파일을 모두 읽어 비교표를 출력한다.

사용법:
  python experiments/summarize_results.py
  python experiments/summarize_results.py --ts 150000
"""

import argparse
import json
from pathlib import Path

RESULT_DIR = Path(__file__).parent / "results"


def load_results(ts_filter: int = None) -> list[dict]:
    pattern = "walk_forward_tm*.json"
    rows = []
    for path in sorted(RESULT_DIR.glob(pattern)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        cfg = data.get("config", {})
        smr = data.get("summary", {})

        if ts_filter and cfg.get("train_timesteps") != ts_filter:
            continue

        rows.append({
            "file":      path.name,
            "train":     cfg.get("train_months", "?"),
            "test":      cfg.get("test_months",  "?"),
            "timesteps": cfg.get("train_timesteps", "?"),
            "n_envs":    cfg.get("n_envs", "?"),
            "folds":     smr.get("n_folds", "?"),
            "cagr":      smr.get("mean_cagr",         0.0),
            "std_cagr":  smr.get("std_cagr",          0.0),
            "sharpe":    smr.get("mean_sharpe",        0.0),
            "mdd":       smr.get("mean_max_drawdown",  0.0),
        })

    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("결과 파일 없음.")
        return

    # 헤더
    print()
    print("=" * 85)
    print(f"  {'train':>6} {'test':>5} {'steps':>8} {'폴드':>5} │"
          f" {'CAGR':>9} {'±std':>8} {'Sharpe':>8} {'MDD':>8}")
    print("-" * 85)

    # CAGR 기준 내림차순 정렬
    for r in sorted(rows, key=lambda x: x["cagr"], reverse=True):
        print(f"  {r['train']:>6}mo {r['test']:>4}mo {r['timesteps']:>8,} {r['folds']:>5}개 │"
              f" {r['cagr']:>+8.2%} {r['std_cagr']:>7.2%} {r['sharpe']:>8.3f} {r['mdd']:>7.2%}")

    print("=" * 85)
    print()


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 결과 비교 요약")
    parser.add_argument("--ts", type=int, default=None, help="timesteps 필터 (예: 150000)")
    args = parser.parse_args()

    rows = load_results(ts_filter=args.ts)
    print(f"\n[Walk-Forward 결과 비교]  ({len(rows)}개 파일)")
    print_table(rows)


if __name__ == "__main__":
    main()
