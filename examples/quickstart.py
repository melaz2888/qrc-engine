"""Minimal qrc-engine example."""

from __future__ import annotations

from qrc_engine import Reservoir
from qrc_engine.backends import QiskitBackend
from qrc_engine.tasks import narma10


def main() -> None:
    """Run a small NARMA-10 example on the Qiskit backend."""

    backend = QiskitBackend(n_qubits=3, depth=3, shots=1024, seed=11)
    reservoir = Reservoir(backend=backend, washout=50, alpha=1e-2)
    X_train, y_train, X_test, y_test = narma10(n_samples=500, split=0.8, seed=11)
    reservoir.fit(X_train, y_train)
    score = reservoir.score(X_test, y_test)
    print(f"NRMSE: {score:.4f}")


if __name__ == "__main__":
    main()
