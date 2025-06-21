# src/utils/exp_db.py
from __future__ import annotations
from pathlib import Path
import json, functools
from typing import Dict, List, Optional

import pandas as pd

from src.models.rtg_strategies import rtg_slug
from src.utils.fs_layout import MODELS_ROOT, SCORE_JSON
from src.utils.model_code import parse_model_code
from src.utils.fs_layout import parse_seed_dir


class ExperimentDB:
    """
    Lightweight, read-only accessor for everything under `models/`.

    Build once → O(number_of_score_files); thereafter all ops are pure pandas.
    """

    def __init__(self, root: Path | str | None = None, *, eager: bool = True):
        self.root = Path(root) if root else MODELS_ROOT
        self._df: Optional[pd.DataFrame] = None
        if eager:
            self._load()

    # ───────────────────── public API ──────────────────────
    def df(self, reload: bool = False) -> pd.DataFrame:
        if reload or self._df is None:
            self._load()
        return self._df

    def best(self, by: str = "mean_score") -> pd.DataFrame:
        """
        Best RTG *per model_code / env / seed*.  (One row per trained model.)
        """
        df = self.df()
        idx = df.groupby(["model_code", "env", "seed"], sort=False)[by].idxmax()
        return df.loc[idx].reset_index(drop=True)

    # ───────────── convenience exports ─────────────
    def to_csv(self, path: Path | str, **kw):
        self.df().to_csv(path, index=False, **kw)

    def to_latex(self, path: Path | str, **kw):
        self.df().to_latex(path, index=False, escape=False, **kw)

    def pivot_scores(self) -> pd.DataFrame:
        best = self.best()
        return (best.pivot_table(index="model_code",
                                 columns="env_short",
                                 values="mean_score")
                      .sort_index())

    # ──────────────────── internal ────────────────────
    def _load(self):
        recs: List[Dict] = []
        for f in self.root.glob("*/*/seed-*/**/score.json"):
            try:
                recs.append(self._parse_score(f))
            except Exception as e:
                print(f"⚠️  Skip {f}: {e}")

        df = pd.DataFrame.from_records(recs)
        df.sort_values(["model_code", "env", "seed"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        self._df = df

    @staticmethod
    def _parse_score(path: Path) -> Dict:
        data = json.loads(path.read_text())
        model_code, env_short, seed = parse_seed_dir(path.parent.parent)
        meta = parse_model_code(model_code)
        slug = rtg_slug(data["rtg_config"])
        rec = {
            "model_code": model_code,
            "model_type": meta["type"],          # cart, m-cart, …
            "env_short": env_short,              # HC-ME …
            "env": data.get("env", env_short),   # long name if stored
            "seed": seed,
            "mean_score": data["mean_score"] * 100.0,      # → %
            "rtg_name": slug,        # e.g. fixed-0p80
        }
        # flatten hyper-params
        rec.update({f"hp_{k}": v for k, v in meta["hyperparams"].items()})
        return rec