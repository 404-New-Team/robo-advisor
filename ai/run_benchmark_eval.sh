#!/bin/bash
# MVO / Equal Weight 벤치마크 사전 계산
# DRL과 동일한 기간/폴드 구조로 평가하여 대시보드 비교를 공정하게 만든다.
#
# 저장 결과:
#   experiments/results/mvo_walk_forward_result.json
#   experiments/results/ew_walk_forward_result.json
#
# 사용법:
#   cd ai
#   bash run_benchmark_eval.sh

set -e
cd "$(dirname "$0")"

IMAGE="robo-advisor-ai:latest"

echo "============================================="
echo "  벤치마크 Walk-Forward 사전 계산"
echo "  기간: 2017-01-01 ~ 2025-12-31"
echo "  train=24mo / test=6mo / step=6mo"
echo "  전략: MVO, Equal Weight"
echo "============================================="
echo ""

docker run --rm \
  -v "$(pwd)/experiments/results:/app/experiments/results" \
  -v "$(pwd)/src/.cache:/app/src/.cache" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  "$IMAGE" \
  python experiments/run_benchmark_eval.py \
    --start        2017-01-01 \
    --end          2025-12-31 \
    --train_months 24 \
    --test_months  6 \
    --step_months  6

echo ""
echo "============================================="
echo "  완료. 대시보드에 즉시 반영됩니다."
echo "============================================="
