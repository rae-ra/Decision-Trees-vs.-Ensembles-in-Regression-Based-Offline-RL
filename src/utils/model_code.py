#src/utils/model_code.py
from __future__ import annotations
from typing import Dict
import re

_ALPHA = {0: 0.0, 1: 1e-3, 2: 1e-2}
_ALPHA_INV = {v: k for k, v in _ALPHA.items()}

short2long = {
    "HC-M": "halfcheetah-medium",
    "HC-MR": "halfcheetah-medium-replay",
    "HC-ME": "halfcheetah-medium-expert",
    "Ho-M": "hopper-medium",
    "Ho-MR": "hopper-medium-replay",
    "Ho-ME": "hopper-medium-expert",
    "W-M": "walker2d-medium",
    "W-MR": "walker2d-medium-replay",
    "W-ME": "walker2d-medium-expert",
}

# ───────────────────────── ENCODERS ─────────────────────────
def _tree_code(hp: Dict, prefix: str) -> str:
    d = hp.get("max_depth", 0)
    d = 0 if d == None else d
    l = hp.get("max_leaf_nodes", 0)
    l = 0 if l == None else l
    s = int(bool(hp.get("scale_obs", False)))
    ml = hp.get("min_samples_leaf", 1)
    a = _ALPHA_INV.get(hp.get("ccp_alpha", 0.0), 0)
    return f"{prefix}-D{d}L{l}-{s}-{ml}-{a}"

def _xgb_code(hp: Dict, prefix: str) -> str:
    s = int(bool(hp.get("scale_obs", False)))
    n = hp.get("n_estimators", 0)
    return f"{prefix}-{s}-{n:04d}"

def cart_code(hp: Dict) -> str:   return _tree_code(hp, "CART")
def m_cart_code(hp: Dict) -> str: return _tree_code(hp, "M-CART")
def opti_code(hp: Dict) -> str:
    d = hp.get("max_depth", 0)
    l = hp.get("max_leaf_nodes", 0)
    ml = hp.get("min_samples_leaf", 1)
    return f"OPTI-D{d}L{l}-{ml}"
def xgb_code(hp: Dict) -> str:    return _xgb_code(hp, "XGB")
def m_xgb_code(hp: Dict) -> str:  return _xgb_code(hp, "M-XGB")

def derive_model_code(ptype: str, hp: Dict) -> str:
    ptype = ptype.lower()
    if ptype in {"cart", "tree"}:        return cart_code(hp)
    if ptype in {"m-cart", "m-tree"}:    return m_cart_code(hp)
    if ptype in {"optimal", "greedy"}:   return opti_code(hp)
    if ptype == "xgb":                   return xgb_code(hp)
    if ptype == "m-xgb":                 return m_xgb_code(hp)
    return "UNKNOWN"

# ───────────────────────── DECODERS ─────────────────────────
def parse_tree_code(code: str) -> Dict:
    match = re.match(r"(M-)?CART-D(\d+)L(\d+)-([01])-(\d+)-(\d)", code)
    if not match:
        raise ValueError(f"Invalid tree code: {code}")
    m, d, l, s, ml, a = match.groups()
    model_type = "m-cart" if m else "cart"
    return {
        "type": model_type,
        "hyperparams": {
            "max_depth": None if int(d) == 0 else int(d),
            "max_leaf_nodes": None if int(l) == 0 else int(l),
            "scale_obs": bool(int(s)),
            "min_samples_leaf": int(ml),
            "ccp_alpha": _ALPHA[int(a)]
        }
    }

def parse_opti_code(code: str) -> Dict:
    match = re.match(r"OPTI-D(\d+)L(\d+)-(\d+)", code)
    if not match:
        raise ValueError(f"Invalid opti code: {code}")
    d, l, ml = match.groups()
    return {
        "type": "optimal",
        "hyperparams": {
            "max_depth": int(d),
            "max_leaf_nodes": int(l),
            "min_samples_leaf": int(ml)
        }
    }

def parse_xgb_code(code: str) -> Dict:
    match = re.match(r"(M-)?XGB-([01])-(\d+)", code)
    if not match:
        raise ValueError(f"Invalid XGB code: {code}")
    m, s, n = match.groups()
    model_type = "m-xgb" if m else "xgb"
    return {
        "type": model_type,
        "hyperparams": {
            "scale_obs": bool(int(s)),
            "n_estimators": int(n)
        }
    }

def parse_model_code(code: str) -> Dict:
    if "CART" in code: return parse_tree_code(code)
    if code.startswith("OPTI"): return parse_opti_code(code)
    if "XGB" in code: return parse_xgb_code(code)
    raise ValueError(f"Unknown model code: {code}")
