"""Benchmark all available qrc-engine backends on NARMA-10."""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qrc_engine import Reservoir
from qrc_engine.tasks import narma10
from qrc_engine.utils import nrmse


def load_backends() -> list[tuple[str, object]]:
    """Instantiate all backends that are available in the environment."""

    backends: list[tuple[str, object]] = []
    warnings.filterwarnings("ignore", module="perceval.utils.persistent_data")
    try:
        from qrc_engine.backends import QiskitBackend

        backends.append(
            (
                "Qiskit (3q, d3)",
                {"washout": 50, "alpha": 1e-2},
                QiskitBackend(n_qubits=3, depth=3, seed=11),
            )
        )
    except ImportError as exc:
        print(f"Skipping Qiskit backend: {exc}")
    try:
        from qrc_engine.backends import PercevalBackend

        backends.append(
            (
                "Perceval (5m)",
                {"washout": 75, "alpha": 1e-1},
                PercevalBackend(n_modes=5, n_photons=2, depth=2, seed=3),
            )
        )
    except ImportError as exc:
        print(f"Skipping Perceval backend: {exc}")
    try:
        from qrc_engine.backends import DynamiqsBackend

        backends.append(
            (
                "dynamiqs (4lvl)",
                {"washout": 50, "alpha": 1e-4},
                DynamiqsBackend(levels=4, dt=0.4, gamma=0.06, seed=3),
            )
        )
    except ImportError as exc:
        print(f"Skipping dynamiqs backend: {exc}")
    return backends


def main() -> None:
    """Run the benchmark and save a comparison plot."""

    X_train, y_train, X_test, y_test = narma10(n_samples=500, split=0.8, seed=11)
    available_backends = load_backends()
    if not available_backends:
        raise RuntimeError("No optional backends are installed.")

    figure, axes = plt.subplots(len(available_backends), 1, figsize=(10, 3 * len(available_backends)), sharex=True)
    axes_list = np.atleast_1d(axes)
    results: list[tuple[str, float, float]] = []

    for axis, (label, config, backend) in zip(axes_list, available_backends):
        start = time.perf_counter()
        reservoir = Reservoir(backend=backend, washout=config["washout"], alpha=config["alpha"])
        reservoir.fit(X_train, y_train)
        predictions = reservoir.predict(X_test)
        elapsed = time.perf_counter() - start
        score = nrmse(y_test[config["washout"] :], predictions)
        results.append((label, score, elapsed))

        axis.plot(y_test[config["washout"] :], label="Ground truth", linewidth=1.6)
        axis.plot(predictions, label=label, linewidth=1.2)
        axis.set_ylabel("Output")
        axis.legend(loc="upper right")

    axes_list[-1].set_xlabel("Time step")
    figure.tight_layout()
    output_path = Path(__file__).with_name("benchmark_backends.png")
    figure.savefig(output_path, dpi=180)

    print("Backend         | NRMSE  | Time (s)")
    print("----------------|--------|--------")
    for label, score, elapsed in results:
        print(f"{label:<16}| {score:0.3f}  | {elapsed:0.2f}")
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    main()
