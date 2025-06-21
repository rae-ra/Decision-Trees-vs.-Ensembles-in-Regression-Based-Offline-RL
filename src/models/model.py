from __future__ import annotations

import abc
import json
import os

import numpy as np


# ------------------- rollout eval for one RTG ------------------------
def _rollout_many(model, env, rtg0: float, episodes: int,
                  T: int, ref_span: float) -> float:
    scores = []
    for ep in range(episodes):
        obs = env.reset(seed=model.config.get("seed", 0))
        if isinstance(obs, tuple):         # gymnasium safety
            obs, _ = obs
        rtg, t, done, ep_ret = rtg0, 0, False, 0.0

        while not done and t < T:
            phi = np.hstack([obs, rtg, t]).reshape(1, -1)
            act = np.clip(model.predict(phi)[0],
                          env.action_space.low, env.action_space.high)
            obs, rew, done, *_ = env.step(act)
            ep_ret += rew
            rtg  = np.clip(rtg - rew / ref_span, 0.0, 1.0)
            t += 1

        # ─────── diagnostic: print feature importances once ───────
        if ep == 0:   # only the first episode so logs stay short
            try:
                fi = (model.model.feature_importances_
                      if not hasattr(model.model, "estimators_")
                      else model.model.estimators_[0].feature_importances_)
                print("RTG importance =", fi[-2], "  timestep importance =", fi[-1])
            except Exception:
                pass
        # ───────────────────────────────────────────────────────────
        score = env.get_normalized_score(ep_ret)
        scores.append(score)

    return float(np.mean(scores))


# ------------------- base class for all models ------------------------
class Model(abc.ABC):
    """
    Abstract base for offline RL regressors.
    Must implement:
      • fit(X, y)
      • predict(X)
      • _save_impl(dir_path)
      • _load_impl(dir_path)

    Includes:
      • save/load logic with config
      • evaluate() = RTG-specific rollouts
    """

    def __init__(self,model_code:str,  config: dict):
        self.model_code = model_code
        self.config = config
        self._last_rtg_used = 1.0

    # ------------------- training interface ---------------------------
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        ...

    @abc.abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    # ------------------- persistence: save/load -----------------------
    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._save_impl(dir_path)  # delegate to subclass
        with open(os.path.join(dir_path, "hyperparams.json"), "w") as f:
            json.dump(self.config, f, indent=2)

    @classmethod
    def load(cls, dir_path: str, model_code: str | None = None, config: dict | None = None) -> "Model":
        if config is None:
            with open(os.path.join(dir_path, "hyperparams.json")) as f:
                config = json.load(f)
        obj = cls(model_code, config)
        obj._load_impl(dir_path)
        return obj

    @abc.abstractmethod
    def _save_impl(self, dir_path: str) -> None:
        ...

    @abc.abstractmethod
    def _load_impl(self, dir_path: str) -> None:
        ...

    def get_model_code(self) -> str:
        return self.model_code

    def _print_result_line(self, env, rtg_val: float, score: float):
        model_code = self.get_model_code()
        env_name = getattr(env, "short_name", env.unwrapped.spec.id.split("-")[0])
        seed = self.config.get("seed", 0)
        print(f"CODE: {model_code}, ENV: {env_name}, SEED: {seed}, RTG {rtg_val:.3f}, RESULT {score * 100:.2f}")

    def evaluate(self, env, episodes: int, target_rtg: float) -> float:
        """
        Evaluate this model using a fixed (already scaled) target RTG.
        Returns mean normalized score.
        """
        ref_min = env.ref_min_score
        ref_max = env.ref_max_score
        ref_span = ref_max - ref_min
        T = env.spec.max_episode_steps

        score = _rollout_many(self, env, target_rtg, episodes, T, ref_span)
        self._print_result_line(env, target_rtg, score)
        return score
