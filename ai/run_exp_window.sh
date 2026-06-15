#!/bin/bash
# 관측 윈도우 크기 N 탐색 실험
#
# 2-6-1. N = 10
# 2-6-2. N = 20
# 2-6-3. N = 30
# 2-6-4. N = 40

IMAGE="robo-advisor-ai:latest"
START="2017-01-01"
END="2025-12-31"
TIMESTEPS=225000
N_ENVS=4
N_SEEDS=1
LAMBDA=0.1

# 윈도우 실험은 train/test 고정
TRAIN_MONTHS=24
TEST_MONTHS=6

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/experiments/results"

DOCKER_RUN="docker run --rm \
  -v $SCRIPT_DIR/checkpoints:/app/checkpoints \
  -v $RESULTS_DIR:/app/experiments/results \
  -v $SCRIPT_DIR/.cache:/app/.cache \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-} \
  $IMAGE"

WINDOWS=(10 20 30 40)
DEFAULT_WINDOW=20

echo "======================================================"
echo "  관측 윈도우 N 탐색 실험 (${#WINDOWS[@]}가지)"
echo "  기간: $START ~ $END"
echo "  train=${TRAIN_MONTHS}mo / test=${TEST_MONTHS}mo"
echo "  timesteps: $TIMESTEPS  |  lambda: $LAMBDA"
echo "======================================================"

TOTAL=${#WINDOWS[@]}
IDX=1

for N in "${WINDOWS[@]}"; do
  # 기본값(20)이면 _w 태그 없음, 비기본값이면 _w{N} 태그 추가
  if [ "$N" -eq "$DEFAULT_WINDOW" ]; then
    FNAME="walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_s${N_SEEDS}.json"
  else
    FNAME="walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_s${N_SEEDS}_w${N}.json"
  fi

  echo ""
  echo "[$IDX/$TOTAL] window_size=${N}  →  $FNAME"
  echo "------------------------------------------------------"

  $DOCKER_RUN python experiments/walk_forward_experiment.py \
    --start $START \
    --end $END \
    --train_months $TRAIN_MONTHS \
    --test_months $TEST_MONTHS \
    --drl_timesteps $TIMESTEPS \
    --n_envs $N_ENVS \
    --n_seeds $N_SEEDS \
    --risk_penalty_lambda $LAMBDA \
    --window_size $N

  if [ $? -eq 0 ]; then
    echo "  ✓ 완료: $FNAME"
  else
    echo "  ✗ 실패: window_size=${N}"
  fi

  IDX=$((IDX + 1))
done

echo ""
echo "======================================================"
echo "  전체 완료. 관측 윈도우 실험 결과 비교:"
echo "======================================================"

$DOCKER_RUN python - <<PYEOF
import json
from pathlib import Path

result_dir = Path("/app/experiments/results")
windows     = [10, 20, 30, 40]
default_n   = 20
tm, tt, ts, ns = ${TRAIN_MONTHS}, ${TEST_MONTHS}, ${TIMESTEPS}, ${N_SEEDS}

def fname(n):
    base = f"walk_forward_tm{tm}_tt{tt}_ts{ts}_s{ns}"
    if n != default_n:
        base += f"_w{n}"
    return base + ".json"

print()
print(f"  {'N':>6} │ {'CAGR':>9} {'±std':>8} {'Sharpe':>8} {'MDD':>8} {'폴드':>5}")
print(f"  {'-'*6}-+-{'-'*9}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*5}")

for n in windows:
    fpath = result_dir / fname(n)
    if not fpath.exists():
        print(f"  {n:>6} │  (결과 없음)")
        continue
    data = json.loads(fpath.read_text())
    s = data["summary"]
    print(f"  {n:>6} │ {s['mean_cagr']:>+8.2%} {s['std_cagr']:>7.2%} "
          f"{s['mean_sharpe']:>8.3f} {s['mean_max_drawdown']:>7.2%} {s['n_folds']:>5}개")

print()
PYEOF
