# src/models/xgb_model.py

import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from src.models.model import Model


#: If you choose to save each booster for a multi‐output model separately,
#: this template will be used (one JSON per output dimension).
BOOSTER_FILENAME_TEMPLATE = "booster_{:02d}.json"
#: For single‐output XGB, we'll write exactly one JSON file.
SINGLE_MODEL_FILENAME = "model.json"
#: Name of the pickled scaler (when scale_obs=True).
SCALER_FILENAME = "scaler.pkl"


class XGBModel(Model):
    """
    Single‐output XGBoost model (family “XGB”), with an optional StandardScaler on the “obs” portion.

    Configuration dictionary (config) must contain:
      {
        "xgb": {
            "n_estimators": <int>,
            "max_depth": <int>,         # etc. (any valid XGBRegressor kwargs)
            "scale_obs": <bool>         # if True, we will scale the first (n_features - 2) dims of X
        }
      }

    Behavior:
      - If scale_obs=False (default), we build one XGBRegressor(**xgb_args).
      - If scale_obs=True, we split each input X into “obs” = X[:, :obs_dim] and “tail” = X[:, obs_dim:];
        we fit a StandardScaler on “obs” during train, then concatenate (obs_scaled || tail) → XGBRegressor.
      - We always produce exactly one JSON file named “model.json” (no pickling of sklearn wrappers).

    Persistence under <dir_path>:
      - model.json       (XGBoost’s own JSON‐exported booster)
      - scaler.pkl       (if scale_obs=True)
      - hyperparams.json (saved by the base Model._save())
    """

    def __init__(self, model_code: str, config: Dict[str, Any]) -> None:
        """
        Initialize a single‐output XGBoost regressor.

        Args:
            model_code:  Unique identifier string (used by Model base class).
            config:      Dict containing a single key "xgb" whose value is a dict of hyperparameters:
                         - Any XGBRegressor arguments (e.g. "n_estimators", "max_depth", etc.)
                         - "scale_obs": bool (optional; defaults to False).
        """
        super().__init__(model_code, config)
        # Copy so that popping "scale_obs" does NOT mutate task.hyperparams
        hp = dict(config.get("xgb", {}))
        self.scale_obs = bool(hp.pop("scale_obs", False))

        # Prepare a StandardScaler if needed:
        self.scaler: Optional[StandardScaler] = StandardScaler() if self.scale_obs else None

        # The remaining keys in hp go directly into XGBRegressor:
        xgb_args = {k: v for k, v in hp.items() if k != "scale_obs"}
        self.model = XGBRegressor(**xgb_args)

    def _split_X(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Split input X into “obs” and “tail” based on last 2 reserved columns.

        Assumes: obs_dim = X.shape[1] - 2.

        Returns:
            obs:  X[:, :obs_dim]
            tail: X[:, obs_dim:]
        """
        obs_dim = X.shape[1] - 2
        obs = X[:, :obs_dim]
        tail = X[:, obs_dim:]
        return obs, tail

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the XGBRegressor, with optional obs‐scaling.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target (single output), shape (n_samples,) or (n_samples, 1).
        """
        if self.scale_obs:
            obs, tail = self._split_X(X)
            obs_scaled = self.scaler.fit_transform(obs)
            X_scaled = np.hstack([obs_scaled, tail])
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with the trained XGBRegressor, applying the same obs‐scaling if enabled.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples,) or (n_samples, 1).
        """
        if self.scale_obs:
            obs, tail = self._split_X(X)
            obs_scaled = self.scaler.transform(obs)
            X_scaled = np.hstack([obs_scaled, tail])
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X)

    def _save_impl(self, dir_path: str) -> None:
        """
        Persist the single‐output XGBoost model (model.json) and optional scaler.

        Args:
            dir_path: Target directory.
        """
        os.makedirs(dir_path, exist_ok=True)

        # 1) Save the XGBRegressor to JSON:
        model_path = os.path.join(dir_path, SINGLE_MODEL_FILENAME)
        # XGBRegressor.save_model → writes JSON if filename ends with .json
        self.model.save_model(model_path)

        # 2) If we used a StandardScaler, pickle it:
        if self.scale_obs:
            scaler_path = os.path.join(dir_path, SCALER_FILENAME)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

    def _load_impl(self, dir_path: str) -> None:
        """
        Load the single‐output XGBoost model (model.json) and optional scaler.

        Args:
            dir_path: Directory from which to load.
        """
        hp = self.config.get("xgb", {})
        xgb_args = {k: v for k, v in hp.items() if k != "scale_obs"}

        # 1) If scale_obs, load the pickled scaler:
        if self.scale_obs:
            scaler_path = os.path.join(dir_path, SCALER_FILENAME)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # 2) Recreate a fresh XGBRegressor and load JSON:
        self.model = XGBRegressor(**xgb_args)
        model_path = os.path.join(dir_path, SINGLE_MODEL_FILENAME)
        self.model.load_model(model_path)


class MXGBModel(Model):
    """
    Multi‐output XGBoost model (family “M-XGB”), with an optional StandardScaler on the “obs” portion.

    Configuration dictionary (config) must contain:
      {
        "xgb": {
            "n_estimators": <int>,
            "max_depth": <int>,          # etc. (any valid XGBRegressor kwargs)
            "scale_obs": <bool>          # if True, we will scale the first (n_features - 2) dims of X
        }
      }

    Behavior:
      - If scale_obs=False (default), wraps a single XGBRegressor in MultiOutputRegressor.
      - If scale_obs=True, it will scale the “obs” submatrix before training each sub‐regressor.
      - During save/load, we serialize each underlying booster for each output dimension as
        “booster_00.json”, “booster_01.json”, … in the target directory. We also pickle the
        StandardScaler (if scale_obs=True).

    Persistence under <dir_path>:
      - booster_00.json, booster_01.json, … (one JSON per output dimension)
      - scaler.pkl            (if scale_obs=True)
      - hyperparams.json      (saved by the base Model._save())
    """

    def __init__(self, model_code: str, config: Dict[str, Any]) -> None:
        """
        Initialize a multi‐output XGBoost ensemble.

        Args:
            model_code:  Unique identifier string (used by Model base class).
            config:      Dict containing a single key "xgb" whose value is a dict of hyperparameters:
                         - Any XGBRegressor arguments (e.g. "n_estimators", "max_depth", etc.)
                         - "scale_obs": bool (optional; defaults to False).
        """
        super().__init__(model_code, config)
        # Copy so that popping "scale_obs" does NOT mutate task.hyperparams
        hp = dict(config.get("xgb", {}))
        self.scale_obs = bool(hp.pop("scale_obs", False))

        # StandardScaler (for obs) if needed:
        self.scaler: Optional[StandardScaler] = StandardScaler() if self.scale_obs else None

        # Remaining hp keys go into each XGBRegressor:
        xgb_args = {k: v for k, v in hp.items() if k != "scale_obs"}
        base_reg = XGBRegressor(**xgb_args)
        self.model = MultiOutputRegressor(base_reg, n_jobs=-1)

    def _split_X(self, X: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Split input X into “obs” and “tail” based on last 2 reserved columns.

        Assumes: obs_dim = X.shape[1] - 2.

        Returns:
            obs:  X[:, :obs_dim]
            tail: X[:, obs_dim:]
        """
        obs_dim = X.shape[1] - 2
        obs = X[:, :obs_dim]
        tail = X[:, obs_dim:]
        return obs, tail

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the MultiOutput XGBoost ensemble, with optional obs‐scaling.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Targets, shape (n_samples, n_outputs).
        """
        if self.scale_obs:
            obs, tail = self._split_X(X)
            obs_scaled = self.scaler.fit_transform(obs)
            X_scaled = np.hstack([obs_scaled, tail])
            self.model.fit(X_scaled, y)
        else:
            self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict with the trained multi‐output ensemble, applying the same obs‐scaling if enabled.

        Args:
            X: Input features, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples, n_outputs).
        """
        if self.scale_obs:
            obs, tail = self._split_X(X)
            obs_scaled = self.scaler.transform(obs)
            X_scaled = np.hstack([obs_scaled, tail])
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X)

    def _save_impl(self, dir_path: str) -> None:
        """
        Persist each booster JSON and optional scaler.

        Args:
            dir_path: Target directory.
        """
        os.makedirs(dir_path, exist_ok=True)

        # 1) Save each fitted estimator separately:
        for idx, estimator in enumerate(self.model.estimators_):
            filename = BOOSTER_FILENAME_TEMPLATE.format(idx)
            path = os.path.join(dir_path, filename)
            estimator.save_model(path)

        # 2) Pickle the StandardScaler if used:
        if self.scale_obs:
            scaler_path = os.path.join(dir_path, SCALER_FILENAME)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

    def _load_impl(self, dir_path: str) -> None:
        """
        Load the StandardScaler and each booster JSON to rebuild the MultiOutputRegressor.

        Args:
            dir_path: Directory from which to load.
        """
        hp = self.config.get("xgb", {})
        xgb_args = {k: v for k, v in hp.items() if k != "scale_obs"}

        # 1) If scale_obs, load the pickled scaler:
        if self.scale_obs:
            scaler_path = os.path.join(dir_path, SCALER_FILENAME)
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

        # 2) Find how many “booster_XX.json” files exist:
        all_files = sorted(f for f in os.listdir(dir_path) if f.startswith("booster_") and f.endswith(".json"))
        num_outputs = len(all_files)

        # 3) Reconstruct the MultiOutputRegressor:
        base_reg = XGBRegressor(**xgb_args)
        self.model = MultiOutputRegressor(base_reg, n_jobs=-1)
        self.model.estimators_ = []

        # 4) For each output dimension, load its JSON into a fresh XGBRegressor:
        for idx in range(num_outputs):
            filename = BOOSTER_FILENAME_TEMPLATE.format(idx)
            path = os.path.join(dir_path, filename)
            reg = XGBRegressor(**xgb_args)
            reg.load_model(path)
            self.model.estimators_.append(reg)
