#!/bin/bash
# 보상 함수 Lambda 탐색 실험
# 가설: Lambda가 클수록 MDD↓ / CAGR↓ 트레이드오프 곡선이 존재한다
#
# 2-5-1. lambda = 0.5
# 2-5-2. lambda = 1.0
# 2-5-3. lambda = 2.0
# 2-5-4. lambda = 3.0
# 2-5-5. lambda = 5.0

IMAGE="robo-advisor-ai:latest"
START="2017-01-01"
END="2025-12-31"
TIMESTEPS=225000
N_ENVS=4
N_SEEDS=1

# Lambda 실험은 설정 고정 (가장 신뢰도 높은 폴드 수 기준)
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

LAMBDAS=(0.5 1.0 2.0 3.0 5.0)

echo "======================================================"
echo "  Lambda 탐색 실험 (${#LAMBDAS[@]}가지)"
echo "  기간: $START ~ $END"
echo "  train=${TRAIN_MONTHS}mo / test=${TEST_MONTHS}mo"
echo "  timesteps: $TIMESTEPS  |  n_envs: $N_ENVS"
echo "======================================================"

TOTAL=${#LAMBDAS[@]}
IDX=1

for LAMBDA in "${LAMBDAS[@]}"; do
  LAMBDA_STR="${LAMBDA/./p}"
  FNAME="walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_s${N_SEEDS}_l${LAMBDA_STR}.json"

  echo ""
  echo "[$IDX/$TOTAL] lambda=${LAMBDA}  →  $FNAME"
  echo "------------------------------------------------------"

  $DOCKER_RUN python experiments/walk_forward_experiment.py \
    --start $START \
    --end $END \
    --train_months $TRAIN_MONTHS \
    --test_months $TEST_MONTHS \
    --drl_timesteps $TIMESTEPS \
    --n_envs $N_ENVS \
    --n_seeds $N_SEEDS \
    --risk_penalty_lambda $LAMBDA

  if [ $? -eq 0 ]; then
    echo "  ✓ 완료: $FNAME"
  else
    echo "  ✗ 실패: lambda=${LAMBDA}"
  fi

  IDX=$((IDX + 1))
done

echo ""
echo "======================================================"
echo "  전체 완료. Lambda 실험 결과 비교:"
echo "======================================================"

$DOCKER_RUN python - <<PYEOF
import json
from pathlib import Path

result_dir = Path("/app/experiments/results")
lambdas = [0.5, 1.0, 2.0, 3.0, 5.0]
tm, tt, ts, ns = ${TRAIN_MONTHS}, ${TEST_MONTHS}, ${TIMESTEPS}, ${N_SEEDS}

print()
print(f"  {'Lambda':>8} │ {'CAGR':>9} {'±std':>8} {'Sharpe':>8} {'MDD':>8} {'폴드':>5}")
print(f"  {'-'*8}-+-{'-'*9}-{'-'*8}-{'-'*8}-{'-'*8}-{'-'*5}")

for lam in lambdas:
    lstr = str(lam).replace(".", "p")
    fname = f"walk_forward_tm{tm}_tt{tt}_ts{ts}_s{ns}_l{lstr}.json"
    fpath = result_dir / fname
    if not fpath.exists():
        print(f"  {lam:>8} │  (결과 없음)")
        continue
    data = json.loads(fpath.read_text())
    s = data["summary"]
    print(f"  {lam:>8} │ {s['mean_cagr']:>+8.2%} {s['std_cagr']:>7.2%} "
          f"{s['mean_sharpe']:>8.3f} {s['mean_max_drawdown']:>7.2%} {s['n_folds']:>5}개")

print()
PYEOF
