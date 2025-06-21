# tests/test_exp_db.py

import unittest
import tempfile
import json
from pathlib import Path

from src.utils.exp_db import ExperimentDB


class TestExperimentDB(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._create_mock_score()

    def tearDown(self):
        self.tmp.cleanup()

    def _create_mock_score(self):
        path = self.root / "CART-D0L32-0-1-0" / "Ho-M" / "seed-0" / "max"
        path.mkdir(parents=True)
        score = {
            "mean_score": 0.723,
            "episode_scores": [0.723],
            "rtg_config": {"name": "max"},
            "model_code": "CART-D0L32-0-1-0",
            "model_type": "cart",
            "hyperparams": {
                "max_depth": 0,
                "max_leaf_nodes": 32,
                "scale_obs": True,
                "min_samples_leaf": 1,
                "ccp_alpha": 0.0
            },
            "env": "Ho-M",
            "seed": 0
        }
        with open(path / "score.json", "w") as f:
            json.dump(score, f)

    def test_dataframe_contains_expected_fields(self):
        db = ExperimentDB(root=self.root)
        df = db.df()

        self.assertEqual(len(df), 1)
        row = df.iloc[0]

        self.assertEqual(row["model_code"], "CART-D0L32-0-1-0")
        self.assertEqual(row["model_type"], "cart")
        self.assertEqual(row["env_short"], "Ho-M")
        self.assertEqual(row["seed"], 0)
        self.assertEqual(row["rtg_name"], "max")
        self.assertAlmostEqual(row["mean_score"], 72.3, places=3)

    def test_best_returns_identical_row(self):
        db = ExperimentDB(root=self.root)
        best = db.best()

        self.assertEqual(len(best), 1)
        self.assertEqual(best.iloc[0]["mean_score"], db.df().iloc[0]["mean_score"])

    def test_pivot_scores_has_correct_shape(self):
        db = ExperimentDB(root=self.root)
        pivot = db.pivot_scores()

        self.assertEqual(pivot.shape, (1, 1))
        self.assertIn("Ho-M", pivot.columns)
        self.assertIn("CART-D0L32-0-1-0", pivot.index)


if __name__ == "__main__":
    unittest.main()
