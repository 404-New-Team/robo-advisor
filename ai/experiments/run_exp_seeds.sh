#!/bin/bash
# 실험 2-3: 멀티 시드 앙상블 실험
# 모든 조건을 동일하게 유지하고 n_seeds만 변경하여 공정하게 비교한다.
#
# 사용법:
#   cd ai
#   bash experiments/run_exp_seeds.sh
#   bash experiments/run_exp_seeds.sh --timesteps 300000   # 사용할 훈련 스텝 지정
#   bash experiments/run_exp_seeds.sh --skip 1             # 특정 시드 수 건너뜀

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
N_ENVS=4
TIMESTEPS=225000

SEEDS_LIST=(1 3 5)

# n_seeds별 고정 시드 값
# n_seeds=1: [3]
# n_seeds=3: [3 41 25]
# n_seeds=5: [3 41 25 73 16]
declare -A SEEDS_MAP
SEEDS_MAP[1]="3"
SEEDS_MAP[3]="3 41 25"
SEEDS_MAP[5]="3 41 25 73 16"

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

echo "====================================================="
echo "  실험 2-3: 멀티 시드 앙상블 실험"
echo "  기간: $START ~ $END"
echo "  train=${TRAIN_MONTHS}mo / test=${TEST_MONTHS}mo / step=${STEP_MONTHS}mo"
echo "  timesteps=$TIMESTEPS / n_envs=$N_ENVS"
echo "  시드: n=1:[3]  n=3:[3,41,25]  n=5:[3,41,25,73,16]"
echo "====================================================="
echo ""

TOTAL=${#SEEDS_LIST[@]}
IDX=0

for N_SEEDS in "${SEEDS_LIST[@]}"; do
    IDX=$((IDX + 1))

    if should_skip "$N_SEEDS"; then
        echo "[$IDX/$TOTAL] n_seeds=$N_SEEDS — 건너뜀 (--skip 지정)"
        echo ""
        continue
    fi

    OUTFILE="experiments/results/walk_forward_tm${TRAIN_MONTHS}_tt${TEST_MONTHS}_ts${TIMESTEPS}_s${N_SEEDS}.json"

    if [[ -f "$OUTFILE" ]]; then
        echo "[$IDX/$TOTAL] n_seeds=$N_SEEDS — 이미 존재: $OUTFILE (건너뜀)"
        echo "  재실행하려면 파일을 삭제 후 다시 실행하세요."
        echo ""
        continue
    fi

    SEED_VALS="${SEEDS_MAP[$N_SEEDS]}"
    echo "[$IDX/$TOTAL] n_seeds=$N_SEEDS (seeds: $SEED_VALS) 시작..."
    START_TIME=$(date +%s)

    $DOCKER_RUN python experiments/walk_forward_experiment.py \
        --start         "$START" \
        --end           "$END" \
        --train_months  $TRAIN_MONTHS \
        --test_months   $TEST_MONTHS \
        --step_months   $STEP_MONTHS \
        --drl_timesteps $TIMESTEPS \
        --n_envs        $N_ENVS \
        --seeds         $SEED_VALS

    END_TIME=$(date +%s)
    ELAPSED=$(( END_TIME - START_TIME ))
    echo "  완료: ${ELAPSED}초 소요 → $OUTFILE"
    echo ""
done

echo "====================================================="
echo "  모든 실험 완료. 비교 분석 실행 중..."
echo "====================================================="

$DOCKER_RUN python experiments/compare_exp_seeds.py --timesteps $TIMESTEPS
