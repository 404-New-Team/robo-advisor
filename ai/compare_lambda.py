import json
from pathlib import Path

result_dir = Path("/app/experiments/results")
lambdas = [0.5, 1.0, 2.0, 3.0, 5.0]
tm, tt, ts, ns = 24, 6, 225000, 1

print()
print(f"  {'Lambda':>8} | {'CAGR':>9} {'+-std':>8} {'Sharpe':>8} {'MDD':>8} {'폴드':>5}")
print(f"  {'-'*8}-+-{'-'*9}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*5}")

for lam in lambdas:
    lstr = str(lam).replace(".", "p")
    fname = f"walk_forward_tm{tm}_tt{tt}_ts{ts}_s{ns}_l{lstr}.json"
    fpath = result_dir / fname
    if not fpath.exists():
        print(f"  {lam:>8} |  (결과 없음)")
        continue
    data = json.loads(fpath.read_text())
    s = data["summary"]
    print(f"  {lam:>8} | {s['mean_cagr']:>+8.2%} {s['std_cagr']:>7.2%} {s['mean_sharpe']:>8.3f} {s['mean_max_drawdown']:>7.2%} {s['n_folds']:>5}개")

print()
