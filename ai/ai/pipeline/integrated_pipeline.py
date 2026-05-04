"""
통합 파이프라인: 리서치 에이전트 → 리스크 태그 → RL 환경 → Safe-Guard → SHAP 해석

end-to-end 흐름:
  뉴스 텍스트
    → RiskDetector (Claude API)         # 뉴스에서 리스크 이벤트 탐지
      → RiskTag 목록
        → PortfolioEnv.inject_risk_tags()  # 관측 공간에 반영
          → PPOAgent.predict()             # 포트폴리오 비중 결정
            → SafeGuardMonitor.validate()  # 규제 제약 검증 및 조정
              → PortfolioEnv.step()        # 실행
                → SHAPExplainer           # 의사결정 설명 (선택)
"""

from pathlib import Path
import numpy as np
import yaml

from ..envs.portfolio_env import PortfolioEnv
from ..envs.risk_state import RiskState
from ..research.risk_detector import RiskDetector
from ..agents.ppo_agent import PPOAgent
from ..safeguard.monitor import SafeGuardMonitor, SafeGuardConfig
from ..xai.shap_explainer import SHAPExplainer
from ..data.market_data import fetch_prices


class IntegratedPipeline:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        env_cfg = self.cfg["environment"]
        risk_cfg = self.cfg["risk"]

        self.risk_state = RiskState(
            tag_names=risk_cfg["tag_names"],
            decay_rate=risk_cfg["decay_rate"],
        )

        prices = fetch_prices(
            tickers=env_cfg["tickers"],
            start="2020-01-01",
            end="2024-12-31",
        )

        self.env = PortfolioEnv(
            prices=prices,
            risk_state=self.risk_state,
            window_size=env_cfg["window_size"],
            transaction_cost=env_cfg["transaction_cost"],
            risk_penalty_lambda=self.cfg["reward"]["risk_penalty_lambda"],
            drawdown_penalty_mu=self.cfg["reward"]["drawdown_penalty_mu"],
        )

        train_cfg = self.cfg["training"]
        self.agent = PPOAgent(
            env=self.env,
            learning_rate=train_cfg["learning_rate"],
            batch_size=train_cfg["batch_size"],
        )

        research_cfg = self.cfg["research"]
        self.risk_detector = RiskDetector(
            model=research_cfg["model"],
            max_tokens=research_cfg["max_tokens"],
        )

        self.safeguard = SafeGuardMonitor(SafeGuardConfig())
        self.explainer = SHAPExplainer(self.agent, self._build_feature_names())

    # ------------------------------------------------------------------

    def train(self) -> None:
        """PPO 에이전트 학습."""
        self.agent.train(total_timesteps=self.cfg["training"]["total_timesteps"])

    def run_inference(self, news_texts: list = None, explain_every: int = 10) -> dict:
        """
        학습된 에이전트로 1 에피소드 실행.
        news_texts가 주어지면 리스크 태그를 탐지하여 주입한다.
        """
        obs, _ = self.env.reset()

        # 뉴스 → 리스크 태그 주입
        if news_texts:
            tags = self.risk_detector.detect(news_texts)
            self.env.inject_risk_tags(tags)
            print(f"[리서치 에이전트] 탐지된 태그: {tags}")
            print(f"[리스크 상태]     {self.risk_state}")

        done = False
        step = 0
        history = []

        while not done:
            raw_action = self.agent.predict(obs)
            weights = self.env._softmax(raw_action)

            # Safe-Guard 검증 및 조정
            info_snapshot = self.env._get_info()
            validated_weights = self.safeguard.validate(weights, info_snapshot)
            action = self.safeguard.weights_to_action(validated_weights)

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            history.append({
                "step": step,
                "reward": round(reward, 6),
                "portfolio_value": info["portfolio_value"],
                "drawdown": info["drawdown"],
                "weights": info["weights"],
                "risk_state": info["risk_state"],
            })

            # SHAP 설명 출력 (explainer가 초기화된 경우)
            if step % explain_every == 0 and self.explainer._explainer is not None:
                top = self.explainer.top_k_features(obs, k=3)
                print(f"[SHAP Step {step:4d}] 주요 결정 요인: {top}")

            step += 1

        print(f"\n[Safe-Guard 위반 요약] {self.safeguard.summary()}")
        return {"history": history, "final_info": info}

    def setup_explainer(self, n_background: int = 100) -> None:
        """학습 완료 후 호출 — 배경 데이터로 SHAP explainer 초기화."""
        obs, _ = self.env.reset()
        background = []
        for _ in range(n_background):
            background.append(obs)
            action = self.env.action_space.sample()
            obs, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                obs, _ = self.env.reset()
        self.explainer.fit(np.array(background))

    # ------------------------------------------------------------------

    def _build_feature_names(self) -> list:
        tickers = self.cfg["environment"]["tickers"]
        suffixes = ["ret1d", "ret5d", "ret20d", "vol20d", "mom20d"]
        market = [f"{t}_{s}" for t in tickers for s in suffixes]
        risk = self.cfg["risk"]["tag_names"]
        weights = [f"w_{t}" for t in tickers]
        return market + risk + weights
