# src/utils/yaml_loader.py
# ==============================================================
"""
Turn a YAML experiment spec into a list of TaskSpec objects.

Standard Specification
----------------------

Each YAML experiment file must define a list under the `tasks:` key.
There are three supported formats for task definitions:

1. Flat model definition (type + hyperparams)
   - Example:
     - type: m-cart
       hyperparams: {max_depth: 6, ...}
       envs: [HC-M, HO-M]
       seeds: [0, 1]
       rtgs: [{name: max}, {name: fixed, fixed_norm: 1.0}]
       save_model: true
       load_model: true
       skip_if_result: true

2. Code-based shorthand (tree code or xgb code)
   - Example:
     - code: M-CART-D6L64-1-10-2
       envs: [HC-M]
       seeds: [0]
       rtgs: [{name: fixed, fixed_norm: 1.0}]
       save_model: true

3. Grid definition (bulk generation)
   - Example:
     - grid:
         models:
           - code: M-CART-D6L64-1-10-2
           - type: m-cart
             hyperparams: {max_depth: 4, max_leaf_nodes: 32}
         envs: [HC-M, HO-M]
         seeds: [0, 1]
         rtgs: [{name: max}, {name: fixed, fixed_norm: 1.0}]

Notes
-----
- `envs`, `seeds`, and `rtgs` are required for all tasks.
- Grid expands into all combinations of model × env × seed × RTG.
- Tasks are converted into TaskSpec objects with full metadata.
- Each model code is parsed into type + hyperparameters automatically.
- Field names use snake_case internally, but YAML is tolerant.
"""


from __future__ import annotations
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.utils.model_code import parse_model_code, short2long
from src.utils.task_spec import TaskSpec


# ───────────────────────────── helpers ────────────────────────────────
def _as_list(x: Any) -> List[Any]:
    """Treat scalar as singleton list so product() works uniformly."""
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _expand_type_block(block: Dict) -> List[TaskSpec]:
    hp = deepcopy(block["hyperparams"])
    seeds = _as_list(block["seeds"])
    rtgs = _as_list(block["rtgs"])
    envs = _as_list(block["envs"])
    type_ = block["type"]

    return [
        TaskSpec(
            name=short2long.get(env, env),
            type=type_,
            hyperparams=hp,
            seeds=seeds,
            rtg_strategies=rtgs,
            save_model=block.get("save_model", False),
            load_model=block.get("load_model", False),
            skip_if_result=block.get("skip_if_result", True)
        )
        for env in envs
    ]


def _expand_code_block(block: Dict) -> List[TaskSpec]:
    parsed = parse_model_code(block["code"])
    hp = parsed["hyperparams"]
    type_ = parsed["type"]

    block = {**block, "type": type_, "hyperparams": hp}
    return _expand_type_block(block)


def _expand_grid_block(grid: Dict) -> List[TaskSpec]:
    all_tasks: List[TaskSpec] = []
    envs = _as_list(grid["envs"])
    seeds = _as_list(grid["seeds"])
    rtgs = _as_list(grid["rtgs"])

    for model in grid["models"]:
        if "code" in model:
            parsed = parse_model_code(model["code"])
            hp = parsed["hyperparams"]
            type_ = parsed["type"]
        else:
            hp = model["hyperparams"]
            type_ = model["type"]

        for env in envs:
            all_tasks.append(TaskSpec(
                name=short2long.get(env, env),
                type=type_,
                hyperparams=deepcopy(hp),
                seeds=seeds,
                rtg_strategies=rtgs,
                save_model=model.get("save_model", False),
                load_model=model.get("load_model", False),
                skip_if_result=model.get("skip_if_result", True)
            ))

    return all_tasks


# ---------------------------------------------------------------------
def load_yaml_tasks(path: Path) -> List[TaskSpec]:
    raw = yaml.safe_load(path.read_text())
    all_tasks: List[TaskSpec] = []

    for block in raw.get("tasks", []):
        if "grid" in block:
            all_tasks.extend(_expand_grid_block(block["grid"]))
        elif "code" in block:
            all_tasks.extend(_expand_code_block(block))
        elif "type" in block:
            all_tasks.extend(_expand_type_block(block))
        else:
            raise ValueError(f"Invalid task entry: {block}")

    return all_tasks
