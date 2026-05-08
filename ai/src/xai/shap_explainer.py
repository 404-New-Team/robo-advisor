"""
SHAP 기반 RL 의사결정 해석기.

규제 요건(XAI 의무화) 대응: 포트폴리오 조정 결정의 근거를 특성 중요도로 설명.
KernelExplainer를 사용하므로 모델 구조에 무관하게 적용 가능.
"""

import numpy as np
import shap
from typing import Optional


class SHAPExplainer:
    def __init__(self, agent, feature_names: list):
        self.agent = agent
        self.feature_names = feature_names
        self._explainer: Optional[shap.KernelExplainer] = None

    def fit(self, background_obs: np.ndarray, n_background: int = 100) -> None:
        """학습 완료 후 1회 실행 — 배경 데이터로 KernelExplainer 초기화."""
        background = shap.sample(background_obs, min(n_background, len(background_obs)))

        def predict_fn(obs_batch: np.ndarray) -> np.ndarray:
            return np.array([self.agent.predict(obs, deterministic=True) for obs in obs_batch])

        self._explainer = shap.KernelExplainer(predict_fn, background)

    def explain(self, obs: np.ndarray) -> dict:
        if self._explainer is None:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        shap_values = self._explainer.shap_values(obs.reshape(1, -1), silent=True)

        # multi-output(자산별 action) 이면 절댓값 평균
        if isinstance(shap_values, list):
            importance = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
        else:
            importance = np.abs(shap_values[0])

        return {
            "feature_names": self.feature_names,
            "shap_importance": importance,
            "base_value": self._explainer.expected_value,
        }

    def top_k_features(self, obs: np.ndarray, k: int = 5) -> list:
        """상위 k개 결정 요인 반환 — 의사결정 근거 보고용."""
        result = self.explain(obs)
        importance = result["shap_importance"]
        top_idx = importance.argsort()[::-1][:k]
        return [(self.feature_names[i], round(float(importance[i]), 4)) for i in top_idx]
