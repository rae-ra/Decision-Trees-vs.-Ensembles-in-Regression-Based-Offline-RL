# src/models/optimal_tree_model.py
import os
import pickle
import numpy as np
from src.models.model import Model
from pystreed import STreeDRegressor  # correct regressor


class OptimalTreeModel(Model):
    """
    Multi-output optimal regression tree via PySTreeD (one regressor per action dimension).
    Config hyperparams (defaults in parentheses):
      optimization_task         ("cost-complex-regression")
      max_depth                 (3)
      max_num_nodes             (None)
      min_leaf_node_size        (1)
      cost_complexity           (0.01)
      time_limit                (600)
      random_seed               (27)
      continuous_binarize_strategy ("quantile")
      n_thresholds              (5)
      n_categories              (5)
      # plus any of the other use_* or caching flags if needed
    """

    TREE_FILE = "optimal_trees.pkl"

    def __init__(self, model_code: str, config: dict):
        super().__init__(model_code, config)
        # stash kwargs for each per-output regressor
        kw = config
        self._reg_kwargs = dict(
            optimization_task=kw.get("optimization_task", "cost-complex-regression"),
            max_depth=kw.get("max_depth", 3),
            max_num_nodes=kw.get("max_num_nodes", None),
            min_leaf_node_size=kw.get("min_leaf_node_size", 1),
            cost_complexity=kw.get("cost_complexity", 0.01),
            time_limit=kw.get("time_limit", 600),
            random_seed=kw.get("random_seed", 27),
            continuous_binarize_strategy=kw.get("continuous_binarize_strategy", "quantile"),
            n_thresholds=kw.get("n_thresholds", 5),
            n_categories=kw.get("n_categories", 5),
            # you can add other flags here, e.g. use_branch_caching=...
        )
        # will be filled in fit()
        self.models: list[STreeDRegressor] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: (n_samples, n_features)
        y: (n_samples, n_outputs)
        """
        n_outputs = y.shape[1]
        self.models = []
        for j in range(n_outputs):
            tree = STreeDRegressor(**self._reg_kwargs)
            # each regressor fits a 1D target vector
            tree.fit(X, y[:, j])
            self.models.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns (n_samples, n_outputs)
        """
        # collect each model's predictions, shape (n_samples,)
        preds = [m.predict(X) for m in self.models]
        # stack to shape (n_samples, n_outputs)
        return np.vstack(preds).T

    def _save_impl(self, dir_path: str) -> None:
        # pickle the entire list of regressors
        pass
        #TODO
        # with open(os.path.join(dir_path, self.TREE_FILE), "wb") as f:
        #     for model in self.models:
        #         model.

    def _load_impl(self, dir_path: str) -> None:
        with open(os.path.join(dir_path, self.TREE_FILE), "rb") as f:
            self.models = pickle.load(f)
