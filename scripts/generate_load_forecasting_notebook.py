"""Generate the PJM load forecasting notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    """Write the notebook to disk."""

    notebook_path = Path(__file__).resolve().parent.parent / "notebooks" / "load_forecasting_evaluation.ipynb"
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
            "# PJM Load Forecasting Evaluation\n\n"
            "This notebook displays the real-world electric load forecasting benchmark added to `qrc-engine`. "
            "It compares classical baselines and QRC backends on PJM East hourly demand using the same "
            "features, split, normalization, and metrics."
        ),
        nbf.v4.new_markdown_cell(
            "## Setup\n\n"
            "The notebook reuses the helper functions from `experiments/load_forecasting.py`, so the notebook "
            "and script stay aligned."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "ROOT = Path.cwd().resolve().parent\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n\n"
            "from experiments.load_forecasting import (\n"
            "    DATA_PATH,\n"
            "    N_QRC_TEST,\n"
            "    N_QRC_TRAIN,\n"
            "    OUT_DIR,\n"
            "    WASHOUT,\n"
            "    evaluate,\n"
            "    fit_classical_models,\n"
            "    fit_qrc_models,\n"
            "    load_dataset,\n"
            "    print_table,\n"
            "    save_context_plot,\n"
            "    save_rmse_plot,\n"
            "    save_time_domain_plot,\n"
            "    scale_features,\n"
            ")\n\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "DATA_PATH"
        ),
        nbf.v4.new_markdown_cell(
            "## Load and Prepare Data"
        ),
        nbf.v4.new_code_cell(
            "X_train, y_train, X_test, y_test = load_dataset()\n"
            "X_train_scaled, X_test_scaled = scale_features(X_train, X_test)\n\n"
            "print(f'Train shape: {X_train_scaled.shape}')\n"
            "print(f'Test shape:  {X_test_scaled.shape}')\n"
            "print('Feature count:', X_train_scaled.shape[1])"
        ),
        nbf.v4.new_markdown_cell(
            "## Classical Models on the Full 2018 Test Set"
        ),
        nbf.v4.new_code_cell(
            "classical_full = fit_classical_models(X_train_scaled, y_train, X_test_scaled)\n"
            "full_results = [\n"
            "    evaluate('Ridge (full data)', y_test, classical_full['Ridge']),\n"
            "    evaluate('Random Forest (full data)', y_test, classical_full['Random Forest']),\n"
            "    evaluate('MLP (full data)', y_test, classical_full['MLP']),\n"
            "    evaluate('GAM-like (full data)', y_test, classical_full['GAM-like']),\n"
            "]\n"
            "print_table(full_results, 'Classical models - full training set')\n"
            "full_results"
        ),
        nbf.v4.new_markdown_cell(
            "## Fair 5,000 / 2,000 Window for Classical vs QRC"
        ),
        nbf.v4.new_code_cell(
            "X_qrc_train = X_train_scaled[-N_QRC_TRAIN:]\n"
            "y_qrc_train = y_train[-N_QRC_TRAIN:]\n"
            "X_qrc_test = X_test_scaled[:N_QRC_TEST]\n"
            "y_qrc_test = y_test[:N_QRC_TEST]\n"
            "y_aligned = y_qrc_test[WASHOUT:]\n\n"
            "classical_sub = fit_classical_models(X_qrc_train, y_qrc_train, X_qrc_test)\n"
            "aligned_classical = {name: prediction[WASHOUT:] for name, prediction in classical_sub.items()}\n\n"
            "qrc_predictions, qrc_timings = fit_qrc_models(X_qrc_train, y_qrc_train, X_qrc_test)\n\n"
            "sub_results = [\n"
            "    evaluate('Ridge', y_aligned, aligned_classical['Ridge']),\n"
            "    evaluate('Random Forest', y_aligned, aligned_classical['Random Forest']),\n"
            "    evaluate('MLP', y_aligned, aligned_classical['MLP']),\n"
            "    evaluate('GAM-like', y_aligned, aligned_classical['GAM-like']),\n"
            "    evaluate('QRC Qiskit (ridge)', y_aligned, qrc_predictions['QRC Qiskit (ridge)']),\n"
            "    evaluate('QRC Qiskit (RF readout)', y_aligned, qrc_predictions['QRC Qiskit (RF readout)']),\n"
            "    evaluate('QRC Dynamiqs', y_aligned, qrc_predictions['QRC Dynamiqs']),\n"
            "    evaluate('QRC Perceval Fock+FB', y_aligned, qrc_predictions['QRC Perceval Fock+FB']),\n"
            "]\n\n"
            "print_table(sub_results, f'Fair comparison - same {N_QRC_TRAIN:,}-sample window')\n"
            "qrc_timings"
        ),
        nbf.v4.new_markdown_cell(
            "## Inline Comparison Plots"
        ),
        nbf.v4.new_code_cell(
            "save_time_domain_plot(y_aligned, aligned_classical, qrc_predictions, sub_results)\n"
            "save_rmse_plot(sub_results)\n"
            "save_context_plot(full_results, sub_results)"
        ),
        nbf.v4.new_markdown_cell(
            "## Notes\n\n"
            "- Classical models currently win clearly on this real-world benchmark.\n"
            "- Among the tested QRC variants here, Dynamiqs performed best on the fair subsample.\n"
            "- The notebook stays aligned with the script by importing its helpers directly."
        ),
    ]

    notebook_path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote notebook to {notebook_path}")


if __name__ == "__main__":
    main()
