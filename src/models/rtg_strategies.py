# src/models/rtg_strategies.py
import abc
import numpy as np
from typing import Any, Dict


class RTGStrategy(abc.ABC):
    @abc.abstractmethod
    def compute(self, data: Dict[str, Any], ref_min: float, ref_max: float) -> float:
        """
        Given a dataset dict with keys "rewards", "terminals", "timeouts",
        return the initial normalized return-to-go in [0,1].
        """
        ...


class MaxReturnStrategy(RTGStrategy):
    def compute(self, data, ref_min, ref_max):
        r = data["rewards"]
        d = (data.get("terminals", np.zeros_like(r, bool)) |
             data.get("timeouts", np.zeros_like(r, bool)))
        ends = np.where(d)[0]  # last step of every trajectory
        cs = np.concatenate(([0.0], np.cumsum(r)))  # pad with 0 for diff
        traj_returns = cs[ends + 1] - cs[np.concatenate(([0], ends[:-1] + 1))]
        raw = traj_returns.max()
        return (raw - ref_min) / (ref_max - ref_min)


class PercentileReturnStrategy(RTGStrategy):
    def __init__(self, percentile: float):
        self.p = percentile

    def compute(self, data, ref_min, ref_max):
        r = data["rewards"]
        d = (data.get("terminals", np.zeros_like(r, bool)) |
             data.get("timeouts", np.zeros_like(r, bool)))
        ends = np.where(d)[0]
        cs = np.concatenate(([0.0], np.cumsum(r)))
        traj_returns = cs[ends + 1] - cs[np.concatenate(([0], ends[:-1] + 1))]
        raw = np.percentile(traj_returns, self.p)
        return (raw - ref_min) / (ref_max - ref_min)


class FixedReturnStrategy(RTGStrategy):
    def __init__(self, fixed_norm: float):
        self.fixed = fixed_norm

    def compute(self, data, ref_min, ref_max):
        return float(self.fixed)


# registry
_STRATS = {
    "max": MaxReturnStrategy,
    "percentile": PercentileReturnStrategy,
    "fixed": FixedReturnStrategy,
}


def make_rtg_strategy(cfg: Dict[str, Any]) -> RTGStrategy:
    """
    cfg should have keys:
      - "name": one of "max","percentile","fixed"
      - any extra params needed by that strategy (e.g. "percentile": 90)
    """
    name = cfg["name"]
    cls = _STRATS.get(name)
    if cls is None:
        raise ValueError(f"Unknown rtg strategy {name}")
    # drop the name
    params = {k: v for k, v in cfg.items() if k != "name"}
    return cls(**params)


def rtg_slug(cfg: Dict[str, Any]) -> str:
    """
    Canonical string representation of an RTG strategy cfg.
      {"name": "max"}               -> "max"
      {"name": "percentile", p: 90} -> "perc-90"
      {"name": "fixed", fixed_norm:0.15} -> "fixed-0p15"
    """
    name = cfg["name"]
    if name == "fixed":
        return f"fixed-{cfg['fixed_norm']:.2f}".replace(".", "p")
    if name == "percentile":
        return f"perc-{int(cfg['percentile'])}"
    return name

def parse_rtg_slug(slug: str) -> Dict[str, Any]:
    """
    Invert the string 'slug' back into an RTG‐strategy config dict, exactly the inverse of rtg_slug(...).

      - "max"        -> {"name": "max"}
      - "perc-90"    -> {"name": "percentile", "percentile": 90}
      - "fixed-0p15" -> {"name": "fixed", "fixed_norm": 0.15}

    Raises:
        ValueError if the slug doesn’t match one of these patterns.
    """
    if slug == "max":
        return {"name": "max"}
    if slug.startswith("perc-"):
        # e.g. "perc-90" → percentile = 90
        percent_str = slug.split("-", 1)[1]
        return {"name": "percentile", "percentile": int(percent_str)}
    if slug.startswith("fixed-"):
        # e.g. "fixed-0p15" → fixed_norm = 0.15
        val_str = slug.split("-", 1)[1]          # "0p15"
        fixed_norm = float(val_str.replace("p", "."))
        return {"name": "fixed", "fixed_norm": fixed_norm}

    raise ValueError(f"Unknown RTG slug: {slug}")
