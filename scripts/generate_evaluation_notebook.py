"""Generate the qrc-engine evaluation notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def main() -> None:
    """Write the evaluation notebook to disk."""

    notebook_path = Path(__file__).resolve().parent.parent / "notebooks" / "qrc_engine_evaluation.ipynb"
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

    cells = [
        nbf.v4.new_markdown_cell(
            "# qrc-engine Evaluation\n\n"
            "This notebook exercises the `qrc-engine` API end to end on two standard reservoir-computing tasks.\n"
            "It uses the same unified `Reservoir` workflow across three backend families: gate-based, photonic, and open-system."
        ),
        nbf.v4.new_code_cell(
            "import warnings\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n\n"
            "from qrc_engine import Reservoir\n"
            "from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend\n"
            "from qrc_engine.tasks import mackey_glass, narma10\n"
            "from qrc_engine.utils import nrmse\n\n"
            "warnings.filterwarnings('ignore', module='perceval.utils.persistent_data')\n"
            "plt.style.use('seaborn-v0_8-whitegrid')"
        ),
        nbf.v4.new_markdown_cell(
            "## NARMA-10 Benchmark\n\n"
            "The benchmark below uses tuned but fixed hyperparameters per backend. "
            "The point is not to win a leaderboard, but to show that the same fit/predict loop runs cleanly across different quantum paradigms."
        ),
        nbf.v4.new_code_cell(
            "X_train, y_train, X_test, y_test = narma10(n_samples=500, split=0.8, seed=11)\n\n"
            "configs = [\n"
            "    ('Qiskit (3q, d3)', {'washout': 50, 'alpha': 1e-2}, QiskitBackend(n_qubits=3, depth=3, seed=11)),\n"
            "    ('Perceval (5m)', {'washout': 75, 'alpha': 1e-1}, PercevalBackend(n_modes=5, n_photons=2, depth=2, seed=3)),\n"
            "    ('dynamiqs (4lvl)', {'washout': 50, 'alpha': 1e-4}, DynamiqsBackend(levels=4, dt=0.4, gamma=0.06, seed=3)),\n"
            "]\n\n"
            "def evaluate_backend(label, config, backend):\n"
            "    reservoir = Reservoir(backend=backend, washout=config['washout'], alpha=config['alpha'])\n"
            "    reservoir.fit(X_train, y_train)\n"
            "    predictions = reservoir.predict(X_test)\n"
            "    score = nrmse(y_test[config['washout']:], predictions)\n"
            "    return reservoir, predictions, score\n\n"
            "results = []\n"
            "figure, axes = plt.subplots(len(configs), 1, figsize=(11, 8), sharex=True)\n"
            "for axis, (label, config, backend) in zip(np.atleast_1d(axes), configs):\n"
            "    reservoir, predictions, score = evaluate_backend(label, config, backend)\n"
            "    results.append((label, config, score))\n"
            "    axis.plot(y_test[config['washout']:], label='Ground truth', linewidth=1.5)\n"
            "    axis.plot(predictions, label=label, linewidth=1.2)\n"
            "    axis.set_ylabel('Output')\n"
            "    axis.legend(loc='upper right')\n"
            "axes[-1].set_xlabel('Time step')\n"
            "figure.suptitle('NARMA-10 Predictions Across Backends', y=1.02)\n"
            "figure.tight_layout()\n\n"
            "print('Backend           | Washout | Alpha   | NRMSE')\n"
            "print('------------------|---------|---------|-------')\n"
            "for label, config, score in results:\n"
            "    print(f\"{label:<18}| {config['washout']:<7} | {config['alpha']:<7.0e} | {score:0.4f}\")\n"
        ),
        nbf.v4.new_markdown_cell(
            "## Mackey-Glass Forecasting\n\n"
            "A second task checks that the same API can handle a chaotic one-step-ahead forecasting problem."
        ),
        nbf.v4.new_code_cell(
            "X_train, y_train, X_test, y_test = mackey_glass(n_samples=600, split=0.8, seed=5)\n"
            "backend = DynamiqsBackend(levels=4, dt=0.4, gamma=0.06, seed=3)\n"
            "reservoir = Reservoir(backend=backend, washout=75, alpha=1e-4)\n"
            "reservoir.fit(X_train, y_train)\n"
            "predictions = reservoir.predict(X_test)\n"
            "score = nrmse(y_test[75:], predictions)\n\n"
            "plt.figure(figsize=(11, 4))\n"
            "plt.plot(y_test[75:], label='Ground truth', linewidth=1.5)\n"
            "plt.plot(predictions, label='Prediction', linewidth=1.2)\n"
            "plt.title(f'Mackey-Glass Forecasting (NRMSE={score:0.4f})')\n"
            "plt.xlabel('Time step')\n"
            "plt.ylabel('Value')\n"
            "plt.legend()\n"
            "plt.tight_layout()\n"
            "print(f'Mackey-Glass NRMSE: {score:0.4f}')"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- The user-facing API stays fixed while the backend changes.\n"
            "- Backends expose different internal physics, but all return fixed-width state vectors for the same readout pipeline.\n"
            "- The benchmark configuration is deterministic, making it suitable for regression testing and portfolio review."
        ),
    ]
    notebook.cells = cells
    notebook_path.write_text(nbf.writes(notebook), encoding="utf-8")
    print(f"Wrote notebook to {notebook_path}")


if __name__ == "__main__":
    main()
