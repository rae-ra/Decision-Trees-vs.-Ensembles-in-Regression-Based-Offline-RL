# ─────────────────────────────────────────────────────────────
# src/runners/plot_results.py
#
# Usage examples
#   python main.py plot --task families
#   python main.py plot --task rtg      --models CART-D30L128-0-5-0 --env Ho-ME
#   python main.py plot --task models   --models CART-D30L32-0-5-0,M-CART-D25L32-0-1-0 --env Ho-M

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.rtg_strategies import parse_rtg_slug, make_rtg_strategy
from src.utils.data_utils import load_dataset
from src.utils.exp_db import ExperimentDB
from src.utils.model_code import short2long

plt.rcParams.update({'font.size': 10})

# helpers ─────────────────────────────────────────────────────────
long2short = {v: k for k, v in short2long.items()}
def to_long(s: str)  -> str: return short2long.get(s, s)
def to_short(l: str) -> str: return long2short.get(l, l)

# paper baseline (green bars) ─────────────────────────────────────
PAPER_XGB = {
    "halfcheetah-medium": 43.19,
    "halfcheetah-medium-replay": 40.91,
    "halfcheetah-medium-expert": 90.34,
    "hopper-medium": 72.91,
    "hopper-medium-replay": 91.66,
    "hopper-medium-expert": 109.85,
    "walker2d-medium": 82.73,
    "walker2d-medium-replay": 87.86,
    "walker2d-medium-expert": 108.96,
}
PAPER_XGB_SHORT = {to_short(k): v for k, v in PAPER_XGB.items()}

# ─────────────────────────────────────────────────────────────────
def main(task: str, models: str | None = None,
         env: str | None = None, out: str | None = None,
         max_depth: int | None = None): # Added max_depth here

    Path("plots").mkdir(exist_ok=True)
    db = ExperimentDB()

    if task == "families":
        plot_families(db, out)

    elif task == "rtg":
        if not (models and env): raise ValueError("--models & --env needed for rtg task")
        plot_rtg(db, models.split(","), env, out)

    elif task == "models": # This is the box plot task
        if not (models and env): raise ValueError("--models & --env needed for models task")
        plot_box(db, models.split(","), env, out)

    elif task == "complexity_sweep":
        if not env: raise ValueError("--env needed for complexity_sweep task")
        plot_complexity_sweep(db, env, out)

    elif task == "mcart_vs_cart_sweep":
        if not (env and max_depth is not None): raise ValueError("--env & --max_depth needed for mcart_vs_cart_sweep task")
        plot_mcart_vs_cart_sweep(db, env, max_depth, out)
    elif task == "capacity_tiers":
        plot_capacity_tiers(db, out)

    else:
        raise ValueError(f"Unknown task: {task}")

# ────────────────────────────────── 1. FAMILY BAR ───────────────
def plot_families(db: ExperimentDB, out: str | None):
    best = db.best()
    best["env_short"] = best["env"].map(to_short)

    # choose BEST model within each family for every seed
    fam_seed_best = (
        best.groupby(["model_type", "env_short", "seed"])["mean_score"]
            .max()
            .reset_index()
    )

    # average that over seeds
    fam_avg = (
        fam_seed_best.groupby(["model_type", "env_short"])["mean_score"]
                     .mean()
                     .unstack()
    )

    fam_avg.loc["xgb-paper"] = pd.Series(PAPER_XGB_SHORT)
    fam_avg = fam_avg.reindex(sorted(fam_avg.columns), axis=1)

    fam_avg.T.plot(kind="bar", figsize=(14, 4))
    plt.ylabel("Best-per-seed return (Normalized Score)")
    plt.xlabel("Environment")
    plt.title("Model-family performance across environments")
    plt.legend(title="Family", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    fname = out or "plots/family_performance.png"
    plt.savefig(fname, dpi=300)
    print("Saved ➜", fname)

# ────────────────────────────────── 2. RTG SWEEP ────────────────
def plot_rtg(db: ExperimentDB,
             model_codes: List[str],
             env_short: str,
             out: str | None):
    """
    Plot the mean-score vs. actual numeric RTG (return‐to‐go) for a given environment and
    a set of model_codes. Instead of using the slug labels as purely categorical x‐ticks,
    we re‐compute the real RTG value for each slug by reloading the dataset & applying the
    strategy—thus ensuring a bijection: slug ↔ numeric RTG.

    Args:
      db          : an ExperimentDB instance, with a method .df() returning a pandas DataFrame that
                    contains at least columns ["env", "model_code", "rtg_name", "mean_score", "std"].
      model_codes : list of strings, e.g. ["CART-D30L128-0-5-0", "M-CART-D25L32-0-1-0"]
      env_short   : short code like "Ho-ME" or "W-M"
                    (will be passed through to_long() to become the dataset name).
      out         : optional path; where to write the figure (PNG). If None, defaults to "plots/rtg_{env_short}.png".
    """

    # 1) Resolve full environment name and reload dataset to get data + ref_min/max
    env_long = to_long(env_short)  # e.g. "Ho-ME" → "hopper-medium-expert"
    # The same pattern used in _run_task: split on the first "-" to feed into load_dataset(...)
    # If env_long = "hopper-medium-expert", then:
    #   load_dataset("hopper", "medium-expert")
    try:
        X, y, env_obj = load_dataset(*env_long.split("-", 1))
    except Exception as e:
        raise RuntimeError(f"Could not load dataset for '{env_long}': {e}")

    data = env_obj.get_dataset()
    ref_min, ref_max = env_obj.ref_min_score, env_obj.ref_max_score

    # 2) Pull only the rows we need from the DB
    df = db.df().query("env == @env_long and model_code in @model_codes")
    if df.empty:
        raise RuntimeError(f"No data found for env='{env_long}' with models={model_codes}")

    # 3) Figure out all unique slugs and map → numeric RTG by re‐computing each strategy
    unique_slugs = sorted(df["rtg_name"].unique())  # arbitrary initial order
    slug_to_numeric: Dict[str, float] = {}
    for slug in unique_slugs:
        cfg = parse_rtg_slug(slug)
        strategy = make_rtg_strategy(cfg)
        numeric_value = strategy.compute(data, ref_min, ref_max)
        slug_to_numeric[slug] = numeric_value

    # 4) Add a new column “rtg_value” to df (float), then groupby on that
    df = df.copy()  # avoid SettingWithCopyWarning
    df["rtg_value"] = df["rtg_name"].map(slug_to_numeric)

    # 5) We want to plot in ascending numeric order, so sort by rtg_value
    #     (We’ll build an “agg” DataFrame exactly like before, but indexed by float)
    agg = (
        df
        .groupby(["model_code", "rtg_value"], sort=False)["mean_score"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # 6) Plot each model’s error bars vs numeric RTG
    plt.figure(figsize=(10, 4))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]
    colors = sns.color_palette("colorblind", len(model_codes))

    for i, model in enumerate(model_codes):
        sub = agg[agg["model_code"] == model].sort_values("rtg_value")
        x = sub["rtg_value"].to_numpy()
        y = sub["mean"].to_numpy()
        e = sub["std"].to_numpy()
        # --- START OF MODIFICATION ---
        # Determine the human-readable label based on the model code
        display_label = ""
        if "XGB" in model:
            display_label = "XGBoost"
        elif "M-CART" in model:
            display_label = "M-CART"
        elif "CART" in model:
            display_label = "CART"
        else:
            # Fallback in case of unexpected model codes, though it shouldn't happen
            display_label = model
        # --- END OF MODIFICATION ---

        plt.errorbar(
            x, y, yerr=e,
            label=display_label,
            fmt=markers[i % len(markers)],
            linestyle=linestyles[i % len(linestyles)],
            color=colors[i],
            elinewidth=1,
            capsize=2,
            alpha=0.9,
            markersize=5,
            linewidth=1.2
        )

    # 7) Format axes
    plt.xlabel("Return-to-Go Input (Normalized)")
    plt.ylabel("Mean return (Normalized Score)")
    plt.title(f"RTG sweep — {env_short}")
    plt.legend(title="Model", loc="upper left", fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    # 8) Save figure
    fname = out or f"plots/rtg_{env_short}.png"
    plt.savefig(fname, dpi=300)
    print("Saved ➜", fname)


# ────────────────────────────────── 3. BOX PLOT ─────────────────
def plot_box(db: ExperimentDB, model_codes: List[str],
             env_short: str, out: str | None):

    env_long = to_long(env_short)
    df = db.df().query("env == @env_long and model_code in @model_codes")
    if df.empty:
        raise RuntimeError(f"No data for {env_short}")

    order = (df.groupby("model_code")["mean_score"]
               .mean()
               .sort_values()
               .index.tolist())

    plt.figure(figsize=(0.9*len(order)+2, 4))
    sns.boxplot(data=df, x="model_code", y="mean_score", order=order)
    plt.ylabel("Best-per-seed return (Normalized Score)")
    plt.title(f"{env_short} — performance across seeds")
    plt.xticks(rotation=30, ha="right", style="italic", fontsize=8)
    plt.tight_layout()
    fname = out or f"plots/box_{env_short}.png"
    plt.savefig(fname, dpi=300)
    print("Saved ➜", fname)

# ─────────────────────── CAPACITY-TIER BAR (v2) ────────────────────────
from src.utils.model_code import parse_model_code

# ───────── helper: label each model_code ────────────────────
def _tier_name(code: str) -> str | None:
    hp     = parse_model_code(code)["hyperparams"]
    depth  = hp.get("max_depth")      or 30_000      # None → huge
    leaves = hp.get("max_leaf_nodes") or 1_000_000   # 0 / None → huge

    if leaves <= 8 or depth <= 3:
        return "SMALL"
    if depth <= 5 or leaves <= 32:
        return "MEDIUM"
    if depth <= 20 and leaves <= 1024:
        return "LARGE"
    return None           # everything else → “virtually unbounded”

# ───────── main capacity-tier plot ──────────────────────────
def plot_capacity_tiers(db: ExperimentDB, out: str | None):
    df = db.best()                                     # best-per-seed rows
    df = df[df["model_type"].isin(["cart", "m-cart"])]  # keep the two families
    df["tier"] = df["model_code"].map(_tier_name)

    # Promote CART rows with tier==None to "UNBOUNDED"; drop others with None
    df.loc[(df["tier"].isna()) & (df["model_type"] == "cart"), "tier"] = "UNBOUNDED"
    df = df.dropna(subset=["tier"])

    df_config_avg = (
        df.groupby(["env", "model_type", "model_code", "tier"])["mean_score"]
        .mean()
        .reset_index()
    )

    # Step 2: Now, from these averaged configurations, find the best performing one
    # per (env, tier, model_type) group. This is the 'best model per env / tier / family'.
    # This will pick the 'M-CART-Y' for hopper-medium-expert because its average is higher.
    best_env_configs = (
        df_config_avg.groupby(["env", "tier", "model_type"])["mean_score"]
        .max()  # Now max() picks the best *average* configuration
        .reset_index()
    )

    # Replace your original best_env with best_env_configs
    best_env = best_env_configs

    # ── Best model per env / tier / family ──────────────────
    # best_env = (
    #     df.groupby(["env", "tier", "model_type"])["mean_score"]
    #       .max()
    #       .reset_index()
    # )
    print(best_env[best_env["model_type"].isin(["m-cart"]) & (best_env["tier"].isin(["LARGE"]))])
    # ── Average those across environments ───────────────────
    tier_avg = (
        best_env.groupby(["tier", "model_type"])["mean_score"]
                .mean()
                .unstack("model_type")
                .reindex(["SMALL", "MEDIUM", "LARGE", "UNBOUNDED"])
    )

    # ── Print raw numbers for sanity check -- copy to paper ─
    print("\n===== CAPACITY-TIER AVERAGES (best-per-env, return) =====")
    print(tier_avg.round(1).fillna("-"))
    print("===== END =====\n")

    # ── Bar plot with dotted XGB baseline ───────────────────
    ax = tier_avg.plot(kind="bar", figsize=(8, 4), width=0.75,
                       color=sns.color_palette("colorblind", 2),
                       edgecolor="black")

    xgb_avg = sum(PAPER_XGB_SHORT.values()) / len(PAPER_XGB_SHORT)
    ax.axhline(xgb_avg, linestyle=":", linewidth=2,
               color="gray", label=f"XGBoost baseline (76.8)")

    ax.set_ylabel("Average normalized return")
    ax.set_xlabel("Capacity tier")
    ax.set_title("Interpretability vs. Performance by capacity tier")

    # Option 1: Slight rotation (most common and usually best for short labels)
    plt.xticks(rotation=0, ha="right")  # Rotate by 30 degrees, align right

    ax.legend(title="Model family", loc="upper left")
    plt.tight_layout()

    fname = out or "plots/capacity_tiers.png"
    plt.savefig(fname, dpi=300)
    print("Saved ➜", fname)

def get_common_min_samples_leaf(df):
    if df.empty:
        return 1

    unique_ml = df['min_samples_leaf'].dropna().unique()

    if len(unique_ml) == 0:
        return 1

    if len(unique_ml) == 1:
        return unique_ml[0]

    preferred_values = [5, 10, 50]
    for val in preferred_values:
        if val in unique_ml:
            if not df[df['min_samples_leaf'] == val].empty:
                return val

    mode_val = df['min_samples_leaf'].mode()
    if not mode_val.empty:
        return mode_val[0]

    return 1
# ────────────────────────────────── 4. COMPLEXITY SWEEP ─────────
def plot_complexity_sweep(db: ExperimentDB,
                                  env_short: str,
                                  out: str | None):
    """
    Generates multiple plots, each showing performance vs. tree depth for a fixed
    maximum number of leaf nodes. This is done for both CART and M-CART models.
    The x-axis (depth) is linear, not logarithmic.
    Includes a baseline performance line.
    """
    out_dir = out or "plots/"
    env_long = to_long(env_short)
    output_path = Path(out_dir) / env_short
    output_path.mkdir(parents=True, exist_ok=True)

    # Get the baseline score for the current environment
    baseline_score = PAPER_XGB_SHORT.get(env_short)
    if baseline_score is None:
        print(f"Warning: No XGBoost baseline found for environment '{env_short}'. Plot will not include baseline.")

    for model_type in {'m-cart', 'cart'}:
        df_base = db.best().query("env == @env_long and model_type == @model_type").copy()

        if df_base.empty:
            print(f"No {model_type.upper()} data found for env='{env_long}'. Skipping plots.")
            continue

        # Ensure 'hp_' columns are mapped to non-'hp_' names
        df_base['max_depth'] = df_base.get('hp_max_depth', df_base['max_depth'] if 'max_depth' in df_base.columns else None)
        df_base['max_leaf_nodes'] = df_base.get('hp_max_leaf_nodes', df_base['max_leaf_nodes'] if 'max_leaf_nodes' in df_base.columns else None)
        df_base['ccp_alpha'] = df_base.get('hp_ccp_alpha', df_base['ccp_alpha'] if 'ccp_alpha' in df_base.columns else None)
        df_base['min_samples_leaf'] = df_base.get('hp_min_samples_leaf', df_base['min_samples_leaf'] if 'min_samples_leaf' in df_base.columns else None)

        # Fix 1: Convert NaN in max_leaf_nodes back to 0 if 0 represents unbounded and might have become NaN
        # This assumes that if max_leaf_nodes is NaN, it was intended to be 0 (unbounded)
        df_base['max_leaf_nodes'] = df_base['max_leaf_nodes'].replace(np.nan, 0.0)


        # Handle NaN max_depth values: Map NaN (unbounded) to a very large number for plotting order
        # Using 50.0 as chosen previously, assuming it's beyond all other tested depths
        df_base['depth_numeric'] = df_base['max_depth'].replace(np.nan, 50.0)

        # Filter for ccp_alpha == 0.0
        df_filtered_ccp = df_base.query("ccp_alpha == 0.0").copy()
        if df_filtered_ccp.empty:
            print(f"No {model_type.upper()} data with ccp_alpha=0.0 found for {env_short}. Skipping plots.")
            continue

        # Determine a common min_samples_leaf for consistency
        common_min_samples_leaf = get_common_min_samples_leaf(df_filtered_ccp)

        df_filtered_final = df_filtered_ccp.query("min_samples_leaf == @common_min_samples_leaf").copy()
        if df_filtered_final.empty:
            print(f"No {model_type.upper()} data with ccp_alpha=0.0 and min_samples_leaf={common_min_samples_leaf} for {env_short}. Re-trying with min_samples_leaf == 1 (if available).")
            if common_min_samples_leaf != 1 and (df_filtered_ccp['min_samples_leaf'] == 1).any():
                 df_filtered_final = df_filtered_ccp.query("min_samples_leaf == 1").copy()
            if df_filtered_final.empty:
                print(f"No {model_type.upper()} data with ccp_alpha=0.0 for {env_short} after all min_samples_leaf filters. Skipping plots.")
                continue

        # Define the specific max_leaf_nodes to create separate plots for
        target_leaf_nodes = [8, 32, 128, 1024, 0] # 0 for 'Unbounded'
        leaf_node_labels = {
            8: "8 Leaves", 32: "32 Leaves", 128: "128 Leaves",
            1024: "1024 Leaves", 0: "Unbounded Leaves (max_leaf_nodes=0)"
        }

        # Removed the global x_ticks and x_tick_labels determination here.
        # It will now be determined inside the loop for each plot.

        meaningful_leaf_nodes = []
        performance_threshold = 5
        # Re-introducing this filter based on your previous input that you want to avoid "irrelevant" plots
        # If you want ALL plots generated regardless of performance, remove this block.
        for leaf_node_val in target_leaf_nodes:
            temp_df = df_filtered_final.query("max_leaf_nodes == @leaf_node_val")
            if not temp_df.empty and temp_df['mean_score'].max() > performance_threshold:
                meaningful_leaf_nodes.append(leaf_node_val)
            elif temp_df.empty:
                print(f"DEBUG: No data for leaf_node_val={leaf_node_val} in filtered_final df for {model_type.upper()} {env_short}")
            else:
                print(f"DEBUG: Skipping plot for leaf_node_val={leaf_node_val} in {model_type.upper()} {env_short} as max performance ({temp_df['mean_score'].max():.1f}) is below threshold ({performance_threshold}).")

        if not meaningful_leaf_nodes:
            print(f"No meaningful leaf node configurations found for {model_type.upper()} in {env_short}. Skipping plots.")
            continue


        for leaf_node_val in meaningful_leaf_nodes:
            sub_df = df_filtered_final.query("max_leaf_nodes == @leaf_node_val").copy()

            if sub_df.empty: # This check should ideally not trigger if meaningful_leaf_nodes logic is correct, but safe to keep
                print(f"No data for {model_type.upper()} in {env_short} for max_leaf_nodes={leaf_node_val}. Skipping plot.")
                continue

            # Group by depth and calculate mean/std return
            agg_sub_df = (
                sub_df.groupby('depth_numeric', as_index=False)
                .agg(mean_return=('mean_score', 'mean'), std_return=('mean_score', 'std'))
                .sort_values('depth_numeric')
            )

            # Fix: Determine x_ticks and x_tick_labels *per plot* based on agg_sub_df
            plot_depth_values = sorted(agg_sub_df['depth_numeric'].unique())
            x_ticks = sorted([d for d in plot_depth_values if d != 50.0])
            x_tick_labels = [str(int(d)) for d in x_ticks]
            if 50.0 in plot_depth_values: # Only add 'Unbounded' if 50.0 (NaN original) is present in THIS plot's data
                x_ticks.append(50.0)
                x_tick_labels.append('Unbounded')


            fig, ax = plt.subplots(figsize=(9, 5)) # Get fig and ax objects
            ax.errorbar(
                agg_sub_df['depth_numeric'],
                agg_sub_df['mean_return'],
                yerr=agg_sub_df['std_return'],
                marker='o',
                linestyle='-',
                capsize=5,
                color='teal',
                elinewidth=1.5,
                markersize=7
            )

            # --- Add the baseline dotted line ---
            if baseline_score is not None:
                ax.axhline(baseline_score, linestyle=":", linewidth=2,
                           color="gray", label=f"XGBoost baseline ({baseline_score:.1f})")
                ax.legend(title="Reference", loc="lower right") # Add legend for baseline

            ax.set_xlabel("Maximum Tree Depth")
            ax.set_ylabel("Best-per-seed Mean Return (Normalized Score)")
            ax.set_title(f"{model_type.upper()} Performance on {env_short}\n(Max Leaves: {leaf_node_labels.get(leaf_node_val)}, Min Samples Leaf: {int(common_min_samples_leaf)})")
            ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
            ax.set_ylim(bottom=0) # Ensure y-axis starts at 0

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels, rotation=45, ha="right")

            plt.tight_layout()
            fname = output_path / f"{model_type}_{env_short}_depth_vs_performance_leaves_{leaf_node_val}_msl_{int(common_min_samples_leaf)}.png"
            plt.savefig(fname, dpi=300)
            plt.close(fig) # Close the figure to free memory
            print(f"Saved ➜ {fname}")

# ────────────────────────── 5. M-CART VS CART SWEEP ─────────────
def plot_mcart_vs_cart_sweep(db: ExperimentDB,
                             env_short: str,
                             max_depth: int,  # Fixed depth for this comparison
                             out: str | None):
    """
    Plots mean normalized return vs. leaf count, comparing M-CART and CART
    for a fixed max_depth.
    """
    env_long = to_long(env_short)

    # Get best-per-seed scores for both 'cart' and 'm-cart' model types.
    df_best_per_seed = db.best().query("env == @env_long and model_type in ['cart', 'm-cart']")

    if df_best_per_seed.empty:
        raise RuntimeError(f"No CART or M-CART data found for env='{env_long}' to plot comparison.")

    # CORRECTED: Access flattened hyperparams directly
    df_best_per_seed['max_depth'] = df_best_per_seed['hp_max_depth']
    df_best_per_seed['max_leaf_nodes'] = df_best_per_seed['hp_max_leaf_nodes']
    df_best_per_seed['ccp_alpha'] = df_best_per_seed['hp_ccp_alpha']
    df_best_per_seed['model_type'] = df_best_per_seed['model_type'].apply(lambda x: x.upper())  # Standardize to 'CART', 'M-CART' for legend

    # Filter for the specific max_depth and no ccp_alpha pruning, and valid leaf nodes
    df_filtered = df_best_per_seed.query(
        "max_depth == @max_depth and ccp_alpha == 0.0 and `max_leaf_nodes` > 0"
    )

    # Group by model_type and max_leaf_nodes, then average over seeds
    agg_df = (
        df_filtered.groupby(['model_type', 'max_leaf_nodes'], as_index=False)
        .agg(mean_return=('mean_score', 'mean'), std_return=('mean_score', 'std'))
    )

    # Convert mean_return to percentage for plotting consistency
    agg_df['mean_return'] = agg_df['mean_return'] * 100

    plt.figure(figsize=(10, 6))

    # Plotting for each model_type
    model_types_order = ['CART', 'M-CART']  # Ensure consistent order

    markers = {"CART": "o", "M-CART": "s"}  # Specific markers for clarity
    linestyles = {"CART": "solid", "M-CART": "dashed"}
    colors = sns.color_palette("deep", len(model_types_order))

    for i, model_type in enumerate(model_types_order):
        # Ensure model type actually exists in the data before plotting
        if model_type not in agg_df['model_type'].unique():
            continue

        sub_df = agg_df[agg_df['model_type'] == model_type].sort_values('max_leaf_nodes')
        plt.errorbar(
            sub_df['max_leaf_nodes'],
            sub_df['mean_return'],
            yerr=sub_df['std_return'],
            label=model_type,
            fmt=markers.get(model_type, "o"),
            linestyle=linestyles.get(model_type, "solid"),
            color=colors[i],
            elinewidth=1,
            capsize=3,
            alpha=0.9,
            markersize=6,
            linewidth=1.5
        )

    plt.xscale('log', base=2)
    leaf_ticks = [8, 16, 32, 64, 128, 256, 512]
    plt.xticks(leaf_ticks, [str(t) for t in leaf_ticks])

    plt.xlabel("Total Leaf Count")
    plt.ylabel("Best-per-seed Mean Return (Normalized Score)")
    plt.title(f"CART vs M-CART Performance (Depth {max_depth}) on {env_short}")
    plt.legend(title="Model Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.ylim(0, 120)  # Set y-axis limits to 0-120 for consistency
    plt.tight_layout()

    fname = out or f"plots/mcart_vs_cart_sweep_{env_short}_D{max_depth}.png"
    plt.savefig(fname, dpi=300)
    print("Saved ➜", fname)


# ───────────────────────── CLI shim (stand-alone) ───────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["families", "rtg", "models", "complexity_sweep", "mcart_vs_cart_sweep"])
    ap.add_argument("--models", help="comma-sep list (for rtg|models)")
    ap.add_argument("--env", help="short env name (for rtg|models|complexity_sweep|mcart_vs_cart_sweep)")
    ap.add_argument("--out")
    ap.add_argument("--max_depth", type=int, help="Fixed max_depth for mcart_vs_cart_sweep task") # Added this argument
    a = ap.parse_args()
    main(a.task, a.models, a.env, a.out, a.max_depth)