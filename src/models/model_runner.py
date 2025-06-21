#src/models/model_runner.py
import os, json, gc
import time
from typing import List

import numpy as np

from src.utils.task_spec import TaskSpec
from src.utils.data_utils import load_dataset
from src.utils.fs_layout import model_file, eval_dir, seed_dir, env_dir, load_existing_score, get_score_file_path
from src.utils.model_code import derive_model_code
from src.models.rtg_strategies import make_rtg_strategy

from typing import Dict
from src.models.model import Model
from src.utils.time_format import format_duration

EPISODES = 10


def make_model(model_type: str, hyperparams: Dict) -> Model:
    """
    Create a model instance given a model type and hyperparameters.
    """
    model_code = derive_model_code(model_type, hyperparams)
    if model_type.lower() == "xgb":
        from src.models.xgb_model import XGBModel
        return XGBModel(model_code, {"xgb": hyperparams})

    if model_type.lower() in ("m-xgb", "m_xgb"):
        from src.models.xgb_model import MXGBModel
        return MXGBModel(model_code, {"xgb": hyperparams})

    if model_type in ("cart", "tree"):
        from src.models.tree_model import DecisionTreeModel
        return DecisionTreeModel(model_code, hyperparams)

    if model_type in ("m-cart", "m-tree"):
        from src.models.tree_model import MultiOutputDecisionTreeModel
        return MultiOutputDecisionTreeModel(model_code, hyperparams)

    if model_type == "optimal":
        from src.models.optimal_tree_model import OptimalTreeModel
        return OptimalTreeModel(model_code, hyperparams)

    raise ValueError(f"Unknown model type: {model_type}")


def _store_env_score(task: TaskSpec, env_key: str):
    model_id = derive_model_code(task.type, task.hyperparams)
    base = env_dir(model_id, env_key)

    all_best = {}
    for seed_path in base.glob("seed-*"):
        try:
            with open(seed_path / "best.json") as f:
                data = json.load(f)
                all_best[int(data["seed"])] = data["mean_score"]
        except Exception:
            continue

    if all_best:
        env_summary = {
            "mean": float(np.mean(list(all_best.values()))),
            "std": float(np.std(list(all_best.values()))),
            "seed_scores": all_best
        }
        with open(base / "env_best.json", "w") as f:
            json.dump(env_summary, f, indent=2)


def _store_score(task: TaskSpec, env_key: str, seed: int,
                 rtg_cfg: Dict, mean_score: float,
                 episode_scores: List[float] = None):
    """
    Store the evaluation result for a specific (seed, RTG config) pair.
    Includes full hyperparameters, model type, and input metadata for full traceability.
    If file exists, checks schema before overwriting.
    """

    model_id = derive_model_code(task.type, task.hyperparams)
    edir = eval_dir(model_id, env_key, seed, rtg_cfg)
    os.makedirs(edir, exist_ok=True)

    score_file = edir / "score.json"
    result = {
        "mean_score": mean_score,
        "episode_scores": episode_scores if episode_scores else [mean_score],
        "rtg_config": rtg_cfg,
        "model_code": model_id,
        "model_type": task.type,
        "hyperparams": task.hyperparams,
        "env": env_key,
        "seed": seed,
    }

    # Check format of existing file
    load_existing_score(score_file)

    with open(score_file, "w") as f:
        json.dump(result, f, indent=2)


def _store_seed_score(task: TaskSpec, env_key: str, seed: int):
    model_id = derive_model_code(task.type, task.hyperparams)
    sdir = seed_dir(model_id, env_key, seed)
    best_score = -float("inf")
    best_data = None

    for path in sdir.glob("*/score.json"):
        with open(path) as f:
            data = json.load(f)
            score = data.get("mean_score", -float("inf"))
            if score > best_score:
                best_score = score
                best_data = data

    if best_data:
        with open(sdir / "best.json", "w") as f:
            json.dump(best_data, f, indent=2)


def _result_exists(task: TaskSpec, env_key: str, seed: int,
                   rtg_cfg: Dict) -> bool:
    model_id = derive_model_code(task.type, task.hyperparams)
    # print(f">>> [DEBUG] _result_exists for model_code = {model_id}, env={env_key}, seed={seed}, rtg={rtg_cfg}")
    edir = eval_dir(model_id, env_key, seed, rtg_cfg)
    # print(f">>> [DEBUG] checking path {edir}/score.json exists?")
    return (edir / "score.json").exists()


def _obtain_model(task: TaskSpec, seed: int, env_key: str,
                  X, y) -> Model:
    model_id = derive_model_code(task.type, task.hyperparams)
    path = model_file(model_id, env_key, seed, task.type)
    os.makedirs(path.parent, exist_ok=True)

    if task.load_model and path.exists():
        model = make_model(task.type, task.hyperparams)
        model = model.load(str(path.parent))
        print("✅ Model loaded successfully.")
        return model
    # Train from scratch
    print("🔨 Starting model training...")
    t0 = time.time()
    model = make_model(task.type, task.hyperparams)
    model.config["seed"] = seed  # make sure it’s reproducible
    model.fit(X, y)
    t1 = time.time()
    print(f"✅ Model trained in {format_duration(t1 - t0)}")

    if task.save_model:
        model.save(str(path.parent))

    return model


def _run_task(task: TaskSpec):
    env_name = task.name
    X, y, env = load_dataset(*env_name.split("-", 1))
    data = env.get_dataset()
    ref_min, ref_max = env.ref_min_score, env.ref_max_score

    rtg_values = [make_rtg_strategy(cfg).compute(data, ref_min, ref_max)
                  for cfg in task.rtg_strategies]
    rtg_pairs = list(zip(task.rtg_strategies, rtg_values))
    #model = None
    for seed in task.seeds:
        model = None #TODO REMOVE ULTRA SPEED
        for cfg, rtg in rtg_pairs:
            # print(f">>> [DEBUG] about to check RTG={cfg}, task.hyperparams={task.hyperparams}")

            score_file = get_score_file_path(task, env_name, seed, cfg)
            if task.skip_if_result and _result_exists(task, env_name, seed, cfg):
                prev_result = load_existing_score(score_file)
                if prev_result:
                    score = prev_result["mean_score"]
                    print(f"⏩ Skipping existing result → ENV: {env_name}, MODEL: {prev_result['model_code']}, SEED: {seed}, RTG: {rtg:.3f} | Score: {score:.4f}")
                else:
                    print(f"⏩ Skipping result with invalid/corrupt file → {score_file}")
                continue
            if model is None:
                model = _obtain_model(task, seed, env_name, X, y)
            print(f"\n🧪 Starting evaluation → ENV: {env_name}, MODEL: {model.get_model_code()}, SEED: {seed}, RTG: {rtg:.3f}")

            t0 = time.time()
            score = model.evaluate(env, episodes=EPISODES, target_rtg=rtg)
            dt = time.time() - t0
            print(f"✅ Done  |  Score: {score:.4f}  |  Time: {format_duration(dt)}\n")

            _store_score(task, env_name, seed, cfg, score)
            gc.collect()
        _store_seed_score(task, env_name, seed)

    _store_env_score(task, env_name)


class ModelRunner:
    def __init__(self, tasks: List[TaskSpec]):
        self.tasks = tasks

    def run_all(self):
        for task in self.tasks:
            _run_task(task)
