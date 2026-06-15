#!/bin/bash
# 최종 PPO 모델 훈련 (production)
# 실험에서 검증된 최적 하이퍼파라미터로 전체 데이터를 학습하여 저장한다.
#
# 저장 결과:
#   checkpoints/production/seed_3/final_model.zip   (+ seed 41, 25, 73, 16)
#   checkpoints/portfolio_ppo_best.zip               (API 로딩용, seed=3 복사)
#
# 사용법:
#   cd ai
#   bash train_final_model.sh
#   bash train_final_model.sh --timesteps 300000

set -e
cd "$(dirname "$0")"

IMAGE="robo-advisor-ai:latest"

# 인자 파싱 (timesteps 오버라이드 가능)
TIMESTEPS=225000
while [[ $# -gt 0 ]]; do
    case $1 in
        --timesteps) TIMESTEPS="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================="
echo "  최종 PPO 모델 훈련"
echo "  learning_rate = 1e-4"
echo "  lambda        = 0.5"
echo "  window_size   = 30"
echo "  timesteps     = $TIMESTEPS"
echo "  seeds         = 3 41 25 73 16"
echo "  데이터 기간   = 2017-01-01 ~ 2025-12-31"
echo "============================================="
echo ""

docker run --rm \
  -v "$(pwd)/checkpoints:/app/checkpoints" \
  -v "$(pwd)/src/.cache:/app/src/.cache" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
  "$IMAGE" \
  python experiments/train_final_model.py \
    --start         2017-01-01 \
    --end           2025-12-31 \
    --timesteps     "$TIMESTEPS" \
    --learning_rate 1e-4 \
    --lambda_val    0.5 \
    --window_size   30 \
    --n_envs        4 \
    --seeds         3 41 25 73 16

echo ""
echo "============================================="
echo "  훈련 완료"
echo "  API 체크포인트: checkpoints/portfolio_ppo_best.zip"
echo "============================================="
