from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


class PPOAgent:
    def __init__(self, env, learning_rate: float = 3e-4, batch_size: int = 256):
        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=2048,
            gamma=0.99,
            verbose=1,
        )

    def train(self, total_timesteps: int, checkpoint_dir: str = "checkpoints/", callbacks: list = None) -> None:
        cb_list = [
            CheckpointCallback(
                save_freq=50_000,
                save_path=checkpoint_dir,
                name_prefix="portfolio_ppo",
            )
        ]
        if callbacks:
            cb_list.extend(callbacks)
        self.model.learn(total_timesteps=total_timesteps, callback=cb_list)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env=None):
        agent = cls.__new__(cls)
        agent.model = PPO.load(path, env=env)
        return agent

    @property
    def policy(self):
        return self.model.policy
