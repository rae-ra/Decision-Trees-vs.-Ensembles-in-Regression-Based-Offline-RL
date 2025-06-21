#!/usr/bin/env python
"""Main entry‑point for the Offline‑RL experiment suite.

This module exposes a single command‑line interface (CLI) that wraps the most
common maintenance tasks for the project so users don’t have to remember half a
dozen separate scripts.  Keeping everything under one roof makes automation and
CI far easier.

Sub‑commands
------------
run        Execute one or more training pipelines defined in YAML or Python
           config files.
collect    Gather raw metrics, produce tidy CSV summaries and LaTeX tables.
delete     Remove trained model folders that match a glob pattern.
plot       Quick‑look visualisation of evaluation curves for a given task.
aggregate  Produce cross‑task aggregate tables from multiple experiments.

Because we want `python -m main --help` to remain snappy, the sub‑command logic
is implemented in *lazy* imports—heavy dependencies are only pulled in when the
associated sub‑command is executed.

Typical usage
-------------
>>> python -m main run configs/halfcheetah_medium.yml
>>> python -m main collect
>>> python -m main plot --task rtg --models modelA,modelB --env Ho-ME
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List

from src.models.model_runner import ModelRunner
from src.utils.task_spec import TaskSpec

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_python_cfg(path: str) -> List[TaskSpec]:
    """Import a Python config file and return its *get_cfg()* output.

    The given file must define a top‑level function ``get_cfg()`` that returns a
    list of :class:`~src.utils.task_spec.TaskSpec` objects.
    """
    spec = importlib.util.spec_from_file_location("exp_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config module from: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module.get_cfg()


def _load_cfg(path: Path) -> List[TaskSpec]:
    """Load a configuration file *path* and return its pipeline list.

    Parameters
    ----------
    path
        Path to a YAML or Python experiment description.
    """
    if path.suffix in {".yml", ".yaml"}:
        # Lazy import because YAML parsing is only required for the *run* command.
        from src.utils.yaml_loader import load_yaml_tasks

        return load_yaml_tasks(path)

    if path.suffix == ".py":
        return _load_python_cfg(str(path))

    raise ValueError(f"Unsupported config type: {path}")


# ---------------------------------------------------------------------------
# CLI builder
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Create the top‑level :class:`argparse.ArgumentParser`."""
    parser = argparse.ArgumentParser(
        prog="main", description="Offline‑RL experiment driver"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------------- run ---------------------------------- #
    p_run = sub.add_parser("run", help="Run an experiment YAML/.py config")
    p_run.add_argument("config", type=Path, help="Path to experiment config file")

    # ----------------------------- collect -------------------------------- #
    sub.add_parser("collect", help="Collect & summarise results")

    # ------------------------------- plot --------------------------------- #
    p_plot = sub.add_parser("plot", help="Quick plotting stub")
    p_plot.add_argument("--task", required=True, help="Which evaluation task to plot")
    p_plot.add_argument(
        "--models",
        help="Comma‑separated list of model names (task=rtg|models)",
    )
    p_plot.add_argument("--env", help="Environment shorthand (task=rtg|models)")
    p_plot.add_argument("--out", help="Destination path for the generated plot image")
    p_plot.add_argument(
        "--max-depth",
        dest="max_depth",
        type=int,
        help="Maximum trajectory depth to include in the plot.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry‑point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901  (function is complex by design)
    """Execute the selected sub‑command."""
    args = _build_parser().parse_args()

    # -------------------------------------------------------------------- #
    # Dispatch
    # -------------------------------------------------------------------- #
    if args.cmd == "run":
        cfg = _load_cfg(args.config)
        ModelRunner(cfg).run_all()

    elif args.cmd == "collect":
        from src.runners.collect_results import main as collect_main

        collect_main()

    elif args.cmd == "plot":
        from src.runners.plot_results import main as plot_task

        plot_task(
            task=args.task,
            models=getattr(args, "models", None),
            env=getattr(args, "env", None),
            out=getattr(args, "out", None),
            max_depth=getattr(args, "max_depth", None),
        )

    else:  # pragma: no cover – argparse should guarantee this never happens.
        raise ValueError(f"Unknown command {args.cmd!r}")


if __name__ == "__main__":
    # Ensure that an unhandled exception leads to a non‑zero exit status.
    sys.exit(main())
