#!/bin/bash
# Walk-Forward 실험 - 6가지 train/test 설정을 순차 실행
# 기간: 2017~2025  |  timesteps: 150,000  |  n_envs: 4

IMAGE="robo-advisor-ai:latest"
START="2017-01-01"
END="2025-12-31"
TIMESTEPS=150000
N_ENVS=4
N_SEEDS=1
LAMBDA=0.1
WINDOW=20

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/experiments/results"

DOCKER_RUN="docker run --rm \
  -v $SCRIPT_DIR/checkpoints:/app/checkpoints \
  -v $RESULTS_DIR:/app/experiments/results \
  -v $SCRIPT_DIR/.cache:/app/.cache \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-} \
  $IMAGE"

# train_months / test_months 조합
CONFIGS=(
  "24 6"
  "48 12"
  "48 6"
  "36 6"
  "24 3"
  "12 6"
)

echo "======================================================"
echo "  Walk-Forward 실험 (6가지 설정)"
echo "  기간: $START ~ $END"
echo "  timesteps: $TIMESTEPS  |  n_envs: $N_ENVS"
echo "======================================================"

TOTAL=${#CONFIGS[@]}
IDX=1

for CONFIG in "${CONFIGS[@]}"; do
  TRAIN=$(echo $CONFIG | awk '{print $1}')
  TEST=$(echo $CONFIG | awk '{print $2}')
  LAMBDA_STR="${LAMBDA/./p}"
  FNAME="walk_forward_tm${TRAIN}_tt${TEST}_ts${TIMESTEPS}_l${LAMBDA_STR}_n${WINDOW}_s${N_SEEDS}.json"

  echo ""
  echo "[$IDX/$TOTAL] train=${TRAIN}개월 / test=${TEST}개월  →  $FNAME"
  echo "------------------------------------------------------"

  $DOCKER_RUN python experiments/walk_forward_experiment.py \
    --start $START \
    --end $END \
    --train_months $TRAIN \
    --test_months $TEST \
    --drl_timesteps $TIMESTEPS \
    --n_envs $N_ENVS \
    --n_seeds $N_SEEDS \
    --risk_penalty_lambda $LAMBDA \
    --window_size $WINDOW

  if [ $? -eq 0 ]; then
    echo "  ✓ 완료: $FNAME"
  else
    echo "  ✗ 실패: train=${TRAIN} / test=${TEST}"
  fi

  IDX=$((IDX + 1))
done

echo ""
echo "======================================================"
echo "  전체 완료. 결과 파일 목록:"
ls -lh "$RESULTS_DIR"/walk_forward_tm*_ts${TIMESTEPS}_l*_n${WINDOW}_s${N_SEEDS}.json 2>/dev/null || echo "  (결과 없음)"
echo "======================================================"

echo ""
echo "  성능 비교표:"
$DOCKER_RUN python experiments/summarize_results.py --ts $TIMESTEPS
