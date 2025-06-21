# src/models/tree_model.py
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.models.model import Model


class MultiOutputDecisionTreeModel(Model):
    """
    Fast multi-output DecisionTreeRegressor with one tree per action dimension.
    Much faster than MultiOutputRegressor - no parallelization overhead.
    Each tree is smaller and more interpretable per joint/action.
    Artefacts:
        /<dir>/tree.pkl
        /<dir>/hyperparams.json
    """
    TREE_FILE = "tree.pkl"
    SCALER_FILE = "scaler.pkl"

    def __init__(self, model_code, config):
        super().__init__(model_code, config)
        self.scale_obs = config.get("scale_obs", False)
        self.scaler = StandardScaler() if self.scale_obs else None
        self.trees = []  # List of individual trees
        self.n_outputs = None

        # Store config for creating trees during fit
        self.tree_config = {
            'max_depth': config.get("max_depth", None),
            'max_leaf_nodes': config.get("max_leaf_nodes", None),
            'min_samples_leaf': config.get("min_samples_leaf", 1),
            'ccp_alpha': config.get("ccp_alpha", 0.0),
            'criterion': config.get("criterion", "squared_error"),
            'random_state': config.get("seed", 0)
        }

    def fit(self, X, y):
        # Handle both 1D and 2D y
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.n_outputs = y.shape[1]

        # Scale observations if needed
        if self.scale_obs:
            obs_dim = X.shape[1] - 2
            X = np.hstack([self.scaler.fit_transform(X[:, :obs_dim]), X[:, obs_dim:]])

        # Create and train one tree per output dimension
        self.trees = []
        for i in range(self.n_outputs):
            tree = DecisionTreeRegressor(**self.tree_config)
            tree.fit(X, y[:, i])  # Train on just this output dimension
            self.trees.append(tree)

    def predict(self, X):
        # Scale observations if needed
        if self.scale_obs:
            obs_dim = X.shape[1] - 2
            X = np.hstack([self.scaler.transform(X[:, :obs_dim]), X[:, obs_dim:]])

        # Fast prediction: stack all tree predictions
        if self.n_outputs == 1:
            return self.trees[0].predict(X).reshape(-1, 1)

        # Vectorized prediction for multi-output
        predictions = np.column_stack([tree.predict(X) for tree in self.trees])
        return predictions

    # persistence
    def _save_impl(self, dir_path: str):
        with open(os.path.join(dir_path, self.TREE_FILE), "wb") as f:
            pickle.dump((self.trees, self.n_outputs), f)
        if self.scale_obs:
            with open(os.path.join(dir_path, self.SCALER_FILE), "wb") as f:
                pickle.dump(self.scaler, f)

    def _load_impl(self, dir_path: str):
        # rebuild scaler
        if self.scale_obs:
            with open(os.path.join(dir_path, self.SCALER_FILE), "rb") as f:
                self.scaler = pickle.load(f)
        with open(os.path.join(dir_path, self.TREE_FILE), "rb") as f:
            self.trees, self.n_outputs = pickle.load(f)

    @property
    def feature_importances_(self):
        """Return feature importances from the first tree for diagnostic code"""
        if self.trees and len(self.trees) > 0:
            return self.trees[0].feature_importances_
        return None

    @property
    def estimators_(self):
        """Compatibility property - returns list of trees for diagnostic code"""
        return self.trees

    def get_tree_stats(self):
        """Get interpretability stats for each tree/joint"""
        if not self.trees:
            return None

        stats = []
        for i, tree in enumerate(self.trees):
            stats.append({
                'action_dim': i,
                'depth': tree.get_depth(),
                'n_leaves': tree.get_n_leaves(),
                'n_nodes': tree.tree_.node_count,
                'feature_importances': tree.feature_importances_
            })
        return stats

class DecisionTreeModel(Model):
    """
    One interpretable DecisionTreeRegressor.
    Artefacts:
        /<dir>/tree.pkl
        /<dir>/hyperparams.json
    """
    TREE_FILE = "tree.pkl"
    SCALER_FILE = "scaler.pkl"

    def __init__(self, model_code, config):
        super().__init__(model_code, config)
        self.scale_obs = config.get("scale_obs", False)
        self.scaler = StandardScaler() if self.scale_obs else None
        configured_max_depth = config.get("max_depth", None)
        configured_max_leaf_nodes = config.get("max_leaf_nodes", None)
        base = DecisionTreeRegressor(
            max_depth=configured_max_depth,
            max_leaf_nodes=configured_max_leaf_nodes,
            min_samples_leaf=config.get("min_samples_leaf", 1),
            ccp_alpha=config.get("ccp_alpha", 0.0),
            criterion=config.get("criterion", "squared_error")  # ,
            # random_state=config.get("random_state", 0),
        )
        self.model = base

    def fit(self, X, y):
        if self.scale_obs:
            obs_dim = X.shape[1] - 2
            X = np.hstack([self.scaler.fit_transform(X[:, :obs_dim]), X[:, obs_dim:]])
        self.model.fit(X, y)

    def predict(self, X):
        if self.scale_obs:
            obs_dim = X.shape[1] - 2
            X = np.hstack([self.scaler.transform(X[:, :obs_dim]), X[:, obs_dim:]])
        return self.model.predict(X)

    # persistence
    def _save_impl(self, dir_path: str):
        with open(os.path.join(dir_path, self.TREE_FILE), "wb") as f:
            pickle.dump(self.model, f)
        if self.scale_obs:
            with open(os.path.join(dir_path, self.SCALER_FILE), "wb") as f:
                pickle.dump(self.scaler, f)

    def _load_impl(self, dir_path: str):
        # rebuild scaler
        if self.scale_obs:
            with open(os.path.join(dir_path, self.SCALER_FILE), "rb") as f:
                self.scaler = pickle.load(f)
        with open(os.path.join(dir_path, self.TREE_FILE), "rb") as f:
            self.model = pickle.load(f)
