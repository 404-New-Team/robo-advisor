#!/bin/bash
# 실험 2-2: 훈련 스텝 증가 실험
# 모든 조건을 동일하게 유지하고 drl_timesteps만 변경하여 공정하게 비교한다.
#
# 사용법:
#   cd ai
#   bash experiments/run_exp_timesteps.sh
#   bash experiments/run_exp_timesteps.sh --skip 30000 150000   # 특정 스텝 건너뜀

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

TIMESTEPS=(30000 150000 200000 225000 300000 500000)

# --skip 인자 파싱
SKIP=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip) shift; while [[ $# -gt 0 && $1 != --* ]]; do SKIP+=("$1"); shift; done ;;
        *) shift ;;
    esac
done

should_skip() {
    for s in "${SKIP[@]}"; do [[ "$s" == "$1" ]] && return 0; done
    return 1
}

echo "====================================================="
echo "  실험 2-2: 훈련 스텝 증가 실험"
echo "  기간: $START ~ $END"
echo "  train=${TRAIN_MONTHS}mo / test=${TEST_MONTHS}mo / step=${STEP_MONTHS}mo"
echo "  n_seeds=$N_SEEDS / n_envs=$N_ENVS"
echo "====================================================="
echo ""

TOTAL=${#TIMESTEPS[@]}
IDX=0

for TS in "${TIMESTEPS[@]}"; do
    IDX=$((IDX + 1))

    if should_skip "$TS"; then
        echo "[$IDX/$TOTAL] ts=$TS — 건너뜀 (--skip 지정)"
        echo ""
        continue
    fi

    OUTFILE="experiments/results/walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TS}_s${N_SEEDS}.json"

    if [[ -f "$OUTFILE" ]]; then
        echo "[$IDX/$TOTAL] ts=$TS — 이미 존재: $OUTFILE (건너뜀)"
        echo "  재실행하려면 파일을 삭제 후 다시 실행하세요."
        echo ""
        continue
    fi

    echo "[$IDX/$TOTAL] ts=$TS 시작..."
    START_TIME=$(date +%s)

    $DOCKER_RUN python experiments/walk_forward_experiment.py \
        --start      "$START" \
        --end        "$END" \
        --train_months  $TRAIN_MONTHS \
        --test_months   $TEST_MONTHS \
        --step_months   $STEP_MONTHS \
        --drl_timesteps $TS \
        --n_seeds       $N_SEEDS \
        --n_envs        $N_ENVS

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "  완료: ${ELAPSED}초 소요 → $OUTFILE"
    echo ""
done

echo "====================================================="
echo "  모든 실험 완료. 비교 분석 실행 중..."
echo "====================================================="

$DOCKER_RUN python experiments/compare_exp_timesteps.py
