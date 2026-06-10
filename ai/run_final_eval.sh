#!/bin/bash
# Walk-Forward 최종 성능 평가
# 실험에서 검증된 최적 하이퍼파라미터로 평가하고 대시보드 연동 파일을 저장한다.
#
# 저장 결과:
#   experiments/results/walk_forward_tm24_tt6_ts225000_s1_lr1e-4_l0p5_w30.json
#   experiments/results/walk_forward_result.json   (대시보드 API 연동용, 위 파일을 복사)
#
# 사용법:
#   cd ai
#   bash run_final_eval.sh
#   bash run_final_eval.sh --timesteps 300000

set -e
cd "$(dirname "$0")"

IMAGE="robo-advisor-ai:latest"

# 인자 파싱
TIMESTEPS=225000
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================="
echo "  Walk-Forward 최종 성능 평가"
echo "  train=24mo / test=6mo / step=6mo"
echo "  learning_rate = 1e-4"
echo "  lambda        = 0.5"
echo "  window_size   = 30"
echo "  timesteps     = $TIMESTEPS"
echo "  n_seeds       = 5  (seeds: 3 41 25 73 16)"
echo "  데이터 기간   = 2017-01-01 ~ 2025-12-31"
echo "============================================="
echo ""

docker run --rm \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/experiments/results:/app/experiments/results" \
  -v "$(pwd)/src/.cache:/app/src/.cache" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  "$IMAGE" \
  python experiments/walk_forward_experiment.py \
    --start               2017-01-01 \
    --end                 2025-12-31 \
    --train_months        24 \
    --test_months         6 \
    --step_months         6 \
    --drl_timesteps       "$TIMESTEPS" \
    --n_envs              4 \
    --learning_rate       1e-4 \
    --risk_penalty_lambda 0.5 \
    --window_size         30 \
    --seeds               3 41 25 73 16 \
    --final

echo ""
echo "============================================="
echo "  평가 완료"
echo "  대시보드 파일: experiments/results/walk_forward_result.json"
echo "============================================="
