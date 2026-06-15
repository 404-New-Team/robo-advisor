#!/bin/bash
# 실험 2-4: PPO Learning Rate 튜닝 실험
# 모든 조건을 동일하게 유지하고 learning_rate만 변경하여 비교한다.
#
# 파일명 규칙:
#   lr=3e-4 (기본값) → walk_forward_tm24_tt6_ts150000_s1.json
#   lr=1e-4          → walk_forward_tm24_tt6_ts150000_s1_lr1e-4.json
#   lr=1e-3          → walk_forward_tm24_tt6_ts150000_s1_lr1e-3.json
#
# 사용법:
#   cd ai
#   bash experiments/run_exp_lr.sh
#   bash experiments/run_exp_lr.sh --timesteps 300000   # 다른 스텝 사용
#   bash experiments/run_exp_lr.sh --skip 1e-4          # 특정 LR 건너뜀

set -e
cd "$(dirname "$0")/.."

IMAGE="robo-advisor-ai:latest"
DOCKER_RUN="docker run --rm \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/experiments/results:/app/experiments/results \
  -v $(pwd)/.cache:/app/.cache \
  -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-} \
  $IMAGE"

# ── 공통 설정 (모든 실험 동일) ─────────────────────────────────────────────
START="2017-01-01"
END="2025-12-31"
TRAIN_MONTHS=24
TEST_MONTHS=6
STEP_MONTHS=6
N_SEEDS=1
N_ENVS=4
TIMESTEPS=225000

# 테스트할 learning rate 목록 (기본값 3e-4 포함)
LR_LIST=("1e-4" "3e-4" "1e-3")

# 인자 파싱
SKIP=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        --skip) shift; while [[ $# -gt 0 && $1 != --* ]]; do SKIP+=("$1"); shift; done ;;
        *) shift ;;
    esac
done

should_skip() {
    for s in "${SKIP[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    return 1
}

# lr 값에 따라 파일명 suffix 결정 (3e-4는 기본값이라 태그 없음)
lr_suffix() {
    local lr="$1"
    if [[ "$lr" == "3e-4" ]]; then
        echo ""
    else
        echo "_lr${lr}"
    fi
}

echo "====================================================="
echo "  실험 2-4: PPO Learning Rate 튜닝"
echo "  기간: $START ~ $END"
echo "  train=${TRAIN_MONTHS}mo / test=${TEST_MONTHS}mo / step=${STEP_MONTHS}mo"
echo "  timesteps=$TIMESTEPS / n_seeds=$N_SEEDS / n_envs=$N_ENVS"
echo "====================================================="
echo ""

TOTAL=${#LR_LIST[@]}
IDX=0

for LR in "${LR_LIST[@]}"; do
    IDX=$((IDX + 1))

    if should_skip "$LR"; then
        echo "[$IDX/$TOTAL] lr=$LR — 건너뜀 (--skip 지정)"
        echo ""
        continue
    fi

    SUFFIX=$(lr_suffix "$LR")
    OUTFILE="experiments/results/walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_s${N_SEEDS}${SUFFIX}.json"

    if [[ -f "$OUTFILE" ]]; then
        echo "[$IDX/$TOTAL] lr=$LR — 이미 존재: $OUTFILE (건너뜀)"
        echo "  재실행하려면 파일을 삭제 후 다시 실행하세요."
        echo ""
        continue
    fi

    echo "[$IDX/$TOTAL] lr=$LR 시작..."
    START_TIME=$(date +%s)

    $DOCKER_RUN python experiments/walk_forward_experiment.py \
        --start          "$START" \
        --end            "$END" \
        --train_months   $TRAIN_MONTHS \
        --test_months    $TEST_MONTHS \
        --step_months    $STEP_MONTHS \
        --drl_timesteps  $TIMESTEPS \
        --n_seeds        $N_SEEDS \
        --n_envs         $N_ENVS \
        --learning_rate  $LR

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "  완료: ${ELAPSED}초 소요 → $OUTFILE"
    echo ""
done

echo "====================================================="
echo "  모든 실험 완료. 비교 분석 실행 중..."
echo "====================================================="

$DOCKER_RUN python experiments/compare_exp_lr.py --timesteps $TIMESTEPS
