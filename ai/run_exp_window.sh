#!/bin/bash
# 관측 윈도우 크기 N 탐색 실험
#
# 2-6-1. N = 20
# 2-6-2. N = 30
# 2-6-3. N = 40
# 2-6-4. N = 60

IMAGE="robo-advisor-ai:latest"
START="2017-01-01"
END="2025-12-31"
TIMESTEPS=150000
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

WINDOWS=(20 30 40 60)

echo "======================================================"
echo "  관측 윈도우 N 탐색 실험 (${#WINDOWS[@]}가지)"
echo "  기간: $START ~ $END"
echo "  train=${TRAIN_MONTHS}mo / test=${TEST_MONTHS}mo"
echo "  timesteps: $TIMESTEPS  |  lambda: $LAMBDA"
echo "======================================================"

TOTAL=${#WINDOWS[@]}
IDX=1

LAMBDA_STR="${LAMBDA/./p}"

for N in "${WINDOWS[@]}"; do
  FNAME="walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_l${LAMBDA_STR}_n${N}_s${N_SEEDS}.json"

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
windows = [20, 30, 40, 60]
lambda_str = "${LAMBDA_STR}"

print()
print(f"  {'N':>6} │ {'CAGR':>9} {'±std':>8} {'Sharpe':>8} {'MDD':>8} {'폴드':>5}")
print(f"  {'-'*6}-+-{'-'*9}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*5}")

for n in windows:
    fname = f"walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_l{lambda_str}_n{n}_s${N_SEEDS}.json"
    fpath = result_dir / fname
    if not fpath.exists():
        print(f"  {n:>6} │  (결과 없음)")
        continue
    data = json.loads(fpath.read_text())
    s = data["summary"]
    print(f"  {n:>6} │ {s['mean_cagr']:>+8.2%} {s['std_cagr']:>7.2%} "
          f"{s['mean_sharpe']:>8.3f} {s['mean_max_drawdown']:>7.2%} {s['n_folds']:>5}개")

print()
PYEOF
