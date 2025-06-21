# src/runners/collect_results.py
import csv
import json
from pathlib import Path

from src.utils.model_code import parse_model_code, short2long
from src.utils.fs_layout import MODELS_ROOT as ROOT, load_env_best_scores

# Output directory for summary files
OUT_DIR = Path("results")


def main():
    OUT_DIR.mkdir(exist_ok=True)

    # Canonical task name mappings — standardized for readability in output

    long2short = {v: k for k, v in short2long.items()}
    ORDERED_LONG = list(long2short)   # column order in CSV
    ORDERED_SHORT = list(short2long)  # printing order

    family_results = {}       # e.g. {"CART": {"D0L32-0-0-1": {"Ho-M": 85.2}}}
    best_per_env_all = {}     # e.g. {"CART": {"Ho-M": (score, code)}}

    # Walk through all model folders inside models/
    for code_dir in ROOT.iterdir():
        if not code_dir.is_dir():
            continue

        code_str = code_dir.name

        # Try to parse the model name using official encoding pattern
        try:
            parsed = parse_model_code(code_str)
            model_type = parsed["type"]                     # e.g. "m-cart"
            family = model_type.upper().replace("_", "-")   # e.g. "M-CART"
        except Exception as e:
            print(f"⚠️ Skipping unrecognized model dir '{code_str}': {e}")
            continue

        print(f"📦 Found model: {code_str}  →  Family: {family}")

        # Attempt to load all env_best.json scores from subdirs
        data = load_env_best_scores(code_dir)
        print(data)
        if not data:
            print(f"⚠️ No valid env_best.json found in {code_str}")
            continue

        RESULTS = family_results.setdefault(family, {})
        best_per_env = best_per_env_all.setdefault(family, {})

        for env_long, mean_score in data.items():
            # Register this score under the model
            env_short_k = long2short[env_long]
            RESULTS.setdefault(code_str, {})[env_short_k] = mean_score

            # Attempt to retrieve model hyperparameters for parsimony sorting
            try:
                hpinfo = parse_model_code(code_str)["hyperparams"]
                leaves = hpinfo.get("max_leaf_nodes") or float("inf")
                depth = hpinfo.get("max_depth") or float("inf")
                mins = hpinfo.get("min_samples_leaf", 0)
            except Exception as e:
                print(f"⚠️ Could not decode config for {code_str}: {e}")
                leaves, depth, mins = float("inf"), float("inf"), 0

            # Score is the leading term; smaller trees win ties
            metric = (mean_score, -leaves, -depth, -mins)

            # Retain best model for this env if it's superior
            if env_short_k not in best_per_env or metric > best_per_env[env_short_k][0]:
                best_per_env[env_short_k] = (metric, mean_score, code_str)

    # For each model family, write JSON, CSV, and print summary
    for family, RESULTS in family_results.items():
        OUT_JSON = OUT_DIR / f"{family}_RESULTS.json"
        OUT_CSV = OUT_DIR / f"{family}_RESULTS.csv"
        best_per_env = best_per_env_all[family]

        with open(OUT_JSON, "w") as f:
            json.dump(RESULTS, f, indent=2)

        header = ["Model"] + ORDERED_LONG
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for code, scores in RESULTS.items():
                row = [code] + [scores.get(long2short[col], "--") for col in ORDERED_LONG]
                writer.writerow(row)

        # Print best model per environment for this family
        print(f"\n=== BEST {family} MODEL PER ENV ===")
        print("{:<8} | {:>6} | {}".format("Env", "Score", "Model"))
        print("-" * 40)
        for env_short in ORDERED_SHORT:
            if env_short in best_per_env:
                _, sc, code = best_per_env[env_short]
                print(f"{env_short:<8} | {sc:>6.1f} | {code}")
            else:
                print(f"{env_short:<8} |   --   |  (no result)")


if __name__ == "__main__":
    main()
