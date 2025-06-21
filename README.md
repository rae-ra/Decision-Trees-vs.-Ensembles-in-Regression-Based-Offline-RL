# Decision Trees vs. Ensembles in Regression‑Based Offline RL

A practical and reproducible framework for the experiments described in “Decision Trees vs. Ensembles in Regression‑Based Offline RL” (Bachelor thesis, TU Delft, 2025). This repository collects code, configs and helper scripts to train decision‑tree and gradient‑boosted baselines on the [D4RL](https://github.com/rail-berkeley/d4rl) benchmarks.

---

## ✨ Features

* **Single command‐line entry‑point** – `python -m main` wraps training, evaluation, result collection and plotting.
* **Declarative experiment specs** – YAML (or Python) configs expand into full factorial grids of *model × env × seed × RTG strategy*.
* **Reproducible storage layout** – models and scores live under a predictable `models/{MODEL_CODE}/{ENV}/seed‑{N}/{RTG_SLUG}` tree.
* **Zero‑boilerplate plotting** – built‑in scripts generate the paper figures (families, RTG sweeps, capacity tiers, …).
* **Self‑contained model codes** – strings like `CART-D2L16-0-10-0` make huge hyper‑parameter grids reproducible without extra bookkeeping.

---

## 📦 Installation

```bash
# Clone
$ git clone https://github.com/rae-ra/Decision-Trees-vs.-Ensembles-in-Regression-Based-Offline-RL
$ cd Decision-Trees-vs.-Ensembles-in-Regression-Based-Offline-RL

# Create env from template
$ conda env create -f environment.yml
$ conda activate offline_rl_shallow_dt
```
---

## 🗂 Repository layout

```
├── main.py                     # Unified CLI (run/collect/plot/delete/aggregate)
├── configs/                    # Experiment YAMLs (examples below)
├── src/
│   ├── models/                 # Model wrappers (CART, OptimalTrees, XGB …)
│   ├── runners/                # Command implementations used by main.py
│   └── utils/                  # Data, FS‑layout, YAML loader, code enc/dec …
├── models/                     # Generated artefacts (ignored by Git)
├── plots/                      # Generated artefacts (ignored by Git)
└── results/                    # Generated artefacts (ignored by Git)
```

A full description of the *generated* folder structure lives in `src/utils/fs_layout.py` but the gist is:

```
models/
  CART-D0L32-0-0-1/
    HC-M/
      seed-0/
        max/score.json
        fixed-1p00/score.json
        tree.pkl  hyperparams.json
      env_best.json   # summarised over seeds
```

---

## ⚙️  Writing experiment YAMLs

Below is a minimal yet complete example covering all three supported task formats (flat, code shorthand, and grid). Save it as `configs/example.yml` and run `python -m main run configs/example.yml`.

```yaml
tasks:
  # 1) Flat definition ---------------------------------------------------
  - type: m-cart
    hyperparams: {max_depth: 6, max_leaf_nodes: 64, scale_obs: true}
    envs: [HC-M, HO-M]
    seeds: [0, 1]
    rtgs:
      - {name: max}
      - {name: fixed, fixed_norm: 1.0}
    save_model: true

  # 2) Code shorthand ----------------------------------------------------
  - code: M-CART-D6L64-1-10-2   # encodes type + hyperparams
    envs: [HC-M]
    seeds: [0]
    rtgs: [{name: fixed, fixed_norm: 1.0}]

  # 3) Grid expansion ----------------------------------------------------
  - grid:
      models:
        - code: CART-D0L32-0-0-1
        - type: xgb
          hyperparams: {n_estimators: 1024, scale_obs: false}
      envs: [W-M, W-ME]
      seeds: [0, 42]
      rtgs: [{name: max}]
```

*YAML tips*

* Use the short env codes (`HC-M`) or the full names (`halfcheetah-medium`) interchangeably.
* Grids expand to **all** combinations, so the above spawns 2 × 2 × 2 × 1 = 8 tasks automatically.

---

## 🗝 Model codes & runtime flags

### Model code anatomy

```
CART-D{depth}L{leafs}-{scale}-{min_leaf}-{alpha_idx}
M-CART-D{depth}L{leafs}-{scale}-{min_leaf}-{alpha_idx}
OPTI-D{depth}L{leafs}-{min_leaf}
XGB-{scale}-{n_estimators}
```

| Token             | Meaning                                                       |
| ----------------- | ------------------------------------------------------------- |
| **CART / M‑CART** | Single‑ vs multi‑output decision tree                         |
| **D**             | Max depth (0 → unlimited)                                     |
| **L**             | Max leaf nodes (0 → unlimited)                                |
| **scale**         | `1` = apply `StandardScaler` to observations, `0` = raw       |
| **min\_leaf**     | Minimum samples per leaf                                      |
| **alpha\_idx**    | Cost‑complexity pruning α index (0 → 0.0, 1 → 1e‑3, 2 → 1e‑2) |
| **n\_estimators** | Number of boosting rounds for XGBoost                         |

### TaskSpec runtime flags

| Flag             | Default | Purpose                                                             |
| ---------------- | ------- | ------------------------------------------------------------------- |
| `save_model`     | `false` | Persist the trained estimator under `models/…` for later inspection |
| `load_model`     | `false` | Skip training and load an existing model if the files are present   |
| `skip_if_result` | `true`  | Don’t re‑evaluate if a valid `score.json` already exists            |

These options let you checkpoint heavy sweeps, resume interrupted runs, or regenerate plots without touching any source code.

---

## 🚀 Running the pipeline

| Stage                | Command                                             | Description                                                  |
| -------------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| **Train + Eval**     | `python -m main run configs/halfcheetah_medium.yml` | Train each model and roll out *N* episodes per RTG strategy. |
| **Collect**          | `python -m main collect`                            | Aggregate raw `score.json` files into tidy CSV summaries.    |
| **Plot**             | `python -m main plot --task families`               | Produce paper figures (saved under `plots/`).                |
| **Clean**            | `python -m main delete --pattern CART-*`            | Delete artefacts that match a glob under `models/`.          |
| **Aggregate tables** | `python -m main aggregate`                          | Output cross‑task CSV / LaTeX tables.                        |

---

## 🖼  Plotting tasks

* `families` – Figure 2: score vs. environment grouped by model family.
* `rtg` – Figure 3: RTG sweeps (needs `--models` and `--env` args).
* `models` – Box‑plots comparing multiple model codes for one env.
* `complexity_sweep` – Depth/leaf sweeps for CART‐style trees.
* `capacity_tiers` – Paper appendix: capacity tiers breakdown.

All figures are dropped into the `plots/` folder unless `--out` is provided.

---

## 📊 Reproducing paper numbers

1. Install the env `(conda env create -f environment.yml)`.
2. Download the D4RL datasets once (Gym will cache them).
3. For every result reported in the paper, write its model code(s), hyper-parameters, seeds, RTG strategies, and target environments into a YAML file that follows the schema in *Writing experiment YAMLs*.
4. `python -m main run path/to/your_experiments.yml` followed by `python -m main collect` will generate a `results/<MODEL_CODE>.csv` with the results.

---

## License

GNU General Public License v3.0 – see `LICENSE` for details.
