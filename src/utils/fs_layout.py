# src/utils/fs_layout.py
"""
Filesystem layout helpers  – v3  (RTG lives under the seed folder)

models/
  <MODEL_CODE>/                # CART-D0L32-0-0-1   |  XGB-0001
    <ENV>/                     # HC-M, hopper-medium-expert …
      seed-<SEED>/             # seed-0, seed-42 …
        <RTG_SLUG>/            # max, fixed-1p00, …
          score.json

Inside seed-* you ALSO keep
  tree.pkl | model.bin
  hyperparams.json
  run_meta.json   (future)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from src.models.rtg_strategies import rtg_slug
from src.utils.model_code import derive_model_code

SCORE_JSON = "score.json"

MODELS_ROOT = Path("models")


# ──────────────── helpers ────────────────

def seed_dir(model_code: str, env: str, seed: int) -> Path:
    return MODELS_ROOT / model_code / env / f"seed-{seed}"


def env_dir(model_code: str, env: str) -> Path:
    return MODELS_ROOT / model_code / env


def eval_dir(model_code: str, env: str, seed: int, rtg_cfg: Dict) -> Path:
    return seed_dir(model_code, env, seed) / rtg_slug(rtg_cfg)


def model_file(model_code: str, env: str, seed: int, ptype: str) -> Path:
    fname = "tree.pkl" if "tree" in ptype else "model.bin"
    return seed_dir(model_code, env, seed) / fname


def parse_seed_dir(p: Path) -> Tuple[str, str, int]:
    """…/models/CARTD0…/HC-M/seed-0  →  (model_code, env, seed)"""
    model_code, env, seed_dir = p.parts[-3:]
    seed = int(seed_dir.replace("seed-", ""))
    return model_code, env, seed


def get_score_file_path(task, env_name, seed, cfg):
    edir = eval_dir(derive_model_code(task.type, task.hyperparams), env_name, seed, cfg)
    score_file = edir / SCORE_JSON
    return score_file


def load_existing_score(score_file):
    """
    Loads and checks the validity of a score.json file.
    Returns the parsed result if valid, else None.
    """
    if score_file.exists():
        try:
            with open(score_file) as f:
                result = json.load(f)
            assert "mean_score" in result and "model_type" in result
            return result
        except Exception as e:
            print(f"⚠️ Corrupt or invalid score.json at {score_file}, overwriting. ({e})")
    return None


def load_env_best_scores(model_dir: Path) -> Dict[str, float]:
    """
    Read best scores per environment from env_best.json files in a given model_dir.
    Returns a dict mapping env_short -> score (multiplied by 100).
    """
    results = {}
    if not model_dir.is_dir():
        return results

    for env_path in model_dir.iterdir():
        if not env_path.is_dir():
            continue
        f = env_path / "env_best.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
            env_short = env_path.name
            score = data["mean"] * 100
            results[env_short] = score
        except Exception as e:
            print(f"⚠️ Failed to read {f}: {e}")
    return results
