"""
Execute the reward-behavior notebook non-interactively for a given simulation.

The notebook is responsible for saving figures and tables under:
    analysis/<sim_id>/reward_behavior/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 0_design_functionReward.ipynb for a simulation id."
    )
    parser.add_argument("sim_id", help="Simulation folder id, e.g. 20260406_0736")
    parser.add_argument(
        "--episode-mode",
        default="stabilization_success_best_reward",
        help="Episode selection mode used by the notebook.",
    )
    parser.add_argument(
        "--manual-episode-id",
        type=int,
        default=None,
        help="Episode id when --episode-mode=manual.",
    )
    return parser.parse_args()


def load_notebook_code_cells(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "0_design_functionReward.ipynb"

    if not notebook_path.exists():
        print(f"ERROR: notebook not found at {notebook_path}", file=sys.stderr)
        return 1

    os.environ["RL_REWARD_ANALYSIS_RUN_ID"] = args.sim_id
    os.environ["RL_REWARD_ANALYSIS_EPISODE_MODE"] = args.episode_mode
    if args.manual_episode_id is None:
        os.environ.pop("RL_REWARD_ANALYSIS_EPISODE_ID", None)
    else:
        os.environ["RL_REWARD_ANALYSIS_EPISODE_ID"] = str(args.manual_episode_id)

    matplotlib.use("Agg")
    os.chdir(project_root)

    namespace: dict[str, object] = {"__name__": "__main__"}
    code_cells = load_notebook_code_cells(notebook_path)

    for index, code in enumerate(code_cells):
        try:
            exec(compile(code, f"{notebook_path.name}::cell_{index}", "exec"), namespace)
        except Exception as exc:
            print(f"ERROR: notebook execution failed at code cell {index}: {exc}", file=sys.stderr)
            return 1

    output_dir = namespace.get("ANALYSIS_OUTPUT_DIR")
    selected_episode_id = namespace.get("selected_episode_id")
    tables_xlsx_path = namespace.get("tables_xlsx_path")

    print(f"Notebook ejecutado para sim_id={args.sim_id}")
    if selected_episode_id is not None:
        print(f"selected_episode_id={selected_episode_id}")
    if output_dir is not None:
        print(f"analysis_output_dir={output_dir}")
    if tables_xlsx_path is not None:
        print(f"tables_xlsx={tables_xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
