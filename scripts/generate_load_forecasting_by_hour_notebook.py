"""Generate the by-hour PJM load forecasting notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    """Write the notebook to disk."""

    notebook_path = Path(__file__).resolve().parent.parent / "notebooks" / "load_forecasting_by_hour_evaluation.ipynb"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)

    notebook = nbf.v4.new_notebook()
    notebook.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        }
    )

    notebook.cells = [
        nbf.v4.new_markdown_cell(
            "# PJM Load Forecasting by Hour-of-Day Experts\n\n"
            "This notebook creates 24 separate datasets by hour of day, trains one version of each model per hour, "
            "merges the timestamped predictions back together, and then evaluates the merged model families."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            "The notebook reuses the helper functions from `experiments/load_forecasting_by_hour.py` so the notebook "
            "and script stay synchronized."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = Path.cwd().resolve().parent\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from experiments.load_forecasting_by_hour import (\n"
            "    HOUR_QRC_TEST,\n"
            "    HOUR_QRC_TRAIN,\n"
            "    HOUR_WASHOUT,\n"
            "    HOURS,\n"
            "    OUT_DIR,\n"
            "    fit_full_hourly_classical,\n"
            "    fit_hourly_experts,\n"
            "    prepare_hourly_dataframe,\n"
            "    print_table,\n"
            "    save_hourly_plots,\n"
            ")\n\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "len(HOURS)"
        ),
        nbf.v4.new_markdown_cell(
            "## Build the 24 Hourly Datasets"
        ),
        nbf.v4.new_code_cell(
            "train, test, feature_cols = prepare_hourly_dataframe()\n"
            "print(f'Train rows: {len(train):,}')\n"
            "print(f'Test rows:  {len(test):,}')\n"
            "print(f'Hour buckets: {len(HOURS)}')\n"
            "print(f'Per-hour QRC window: {HOUR_QRC_TRAIN} train / {HOUR_QRC_TEST} test / washout {HOUR_WASHOUT}')"
        ),
        nbf.v4.new_markdown_cell(
            "## Full-Year Classical Experts by Hour"
        ),
        nbf.v4.new_code_cell(
            "full_merged, full_results = fit_full_hourly_classical(train, test, feature_cols)\n"
            "print_table(full_results, 'Classical by-hour experts - full 2018 test set')\n"
            "full_results"
        ),
        nbf.v4.new_markdown_cell(
            "## Matched By-Hour Classical vs QRC Comparison"
        ),
        nbf.v4.new_code_cell(
            "merged, sub_results, timings = fit_hourly_experts(train, test, feature_cols)\n"
            "print_table(sub_results, 'By-hour experts - matched classical vs QRC comparison')\n"
            "timings"
        ),
        nbf.v4.new_markdown_cell(
            "## Inline Plots for the By-Hour Experiment"
        ),
        nbf.v4.new_code_cell(
            "save_hourly_plots(merged, sub_results, full_results)"
        ),
        nbf.v4.new_markdown_cell(
            "## Notes\n\n"
            "- Each hour of day gets its own expert model family.\n"
            "- Predictions are merged back in chronological order before computing the final metrics.\n"
            "- The resulting plots are directly comparable to the original single-model-family experiment."
        ),
    ]

    notebook_path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote notebook to {notebook_path}")


if __name__ == "__main__":
    main()
