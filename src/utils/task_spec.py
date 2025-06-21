from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TaskSpec:
    type: str  # e.g. "m-cart", "xgb"
    hyperparams: Dict[str, Any]  # Model config
    seeds: List[int]  # Seeds to run
    rtg_strategies: List[Dict]  # List of RTG strategy configs

    # Flags
    save_model: bool = False
    load_model: bool = False
    skip_if_result: bool = True

    # Task name (e.g. "hopper-medium-expert")
    name: str = ""
