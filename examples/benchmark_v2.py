"""Benchmark qrc-engine v0.2 backend modes on NARMA-10."""

from __future__ import annotations

import time
import warnings

from qrc_engine import Reservoir
from qrc_engine.tasks import narma10


def benchmark_configs() -> list[tuple[str, object, dict[str, float]]]:
    """Instantiate all benchmark configurations available in the environment."""

    configs: list[tuple[str, object, dict[str, float]]] = []
    warnings.filterwarnings("ignore", module="perceval.utils.persistent_data")
    try:
        from qrc_engine.backends import QiskitBackend

        configs.extend(
            [
                (
                    "Qiskit default",
                    QiskitBackend(n_qubits=3, depth=3, seed=11),
                    {"washout": 50, "alpha": 1e-2},
                ),
                (
                    "Qiskit persistent",
                    QiskitBackend(n_qubits=3, depth=3, persistent_state=True, seed=11),
                    {"washout": 50, "alpha": 1e-2},
                ),
                (
                    "Qiskit shots",
                    QiskitBackend(n_qubits=3, depth=3, use_shots=True, shots=4096, seed=11),
                    {"washout": 50, "alpha": 1e-2},
                ),
            ]
        )
    except ImportError as exc:
        print(f"Skipping Qiskit configurations: {exc}")
    try:
        from qrc_engine.backends import PercevalBackend

        configs.extend(
            [
                (
                    "Perceval field",
                    PercevalBackend(n_modes=5, n_photons=2, depth=2, seed=3),
                    {"washout": 75, "alpha": 1e-1},
                ),
                (
                    "Perceval fock",
                    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True, seed=3),
                    {"washout": 75, "alpha": 1e-1},
                ),
                (
                    "Perceval feedback",
                    PercevalBackend(
                        n_modes=5,
                        n_photons=2,
                        depth=2,
                        fock_mode=True,
                        feedback=True,
                        seed=3,
                    ),
                    {"washout": 75, "alpha": 1e-1},
                ),
            ]
        )
    except ImportError as exc:
        print(f"Skipping Perceval configurations: {exc}")
    try:
        from qrc_engine.backends import DynamiqsBackend

        configs.extend(
            [
                (
                    "Dynamiqs mixture",
                    DynamiqsBackend(levels=4, dt=0.4, gamma=0.06, seed=3),
                    {"washout": 50, "alpha": 1e-4},
                ),
                (
                    "Dynamiqs lindblad",
                    DynamiqsBackend(levels=4, dt=0.4, gamma=0.06, lindblad=True, seed=3),
                    {"washout": 50, "alpha": 1e-4},
                ),
                (
                    "Dynamiqs 2-subsys",
                    DynamiqsBackend(levels=2, dt=0.4, gamma=0.06, lindblad=True, n_subsystems=2, seed=3),
                    {"washout": 50, "alpha": 1e-4},
                ),
            ]
        )
    except ImportError as exc:
        print(f"Skipping dynamiqs configurations: {exc}")
    return configs


def main() -> None:
    """Run the v0.2 benchmark table."""

    X_train, y_train, X_test, y_test = narma10(n_samples=500, split=0.8, seed=11)
    configs = benchmark_configs()
    if not configs:
        raise RuntimeError("No optional backends are installed.")

    results: list[tuple[str, float, float]] = []
    for label, backend, config in configs:
        start = time.perf_counter()
        reservoir = Reservoir(backend=backend, washout=int(config["washout"]), alpha=float(config["alpha"]))
        reservoir.fit(X_train, y_train)
        score = reservoir.score(X_test, y_test)
        elapsed = time.perf_counter() - start
        results.append((label, score, elapsed))

    print("Configuration        | NRMSE  | Time (s)")
    print("---------------------|--------|--------")
    for label, score, elapsed in results:
        print(f"{label:<21}| {score:0.3f}  | {elapsed:0.2f}")


if __name__ == "__main__":
    main()
