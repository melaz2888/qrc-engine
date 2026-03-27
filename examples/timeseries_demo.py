"""Time-series forecasting demo using the dynamiqs-style backend."""

from __future__ import annotations

import matplotlib.pyplot as plt
from pathlib import Path

from qrc_engine import Reservoir
from qrc_engine.backends import DynamiqsBackend
from qrc_engine.tasks import mackey_glass
from qrc_engine.utils import nrmse


def main() -> None:
    """Fit the reservoir on Mackey-Glass and save a plot."""

    X_train, y_train, X_test, y_test = mackey_glass(n_samples=600, split=0.8, seed=5)
    backend = DynamiqsBackend(levels=3, dt=0.3, gamma=0.06, seed=5)
    reservoir = Reservoir(backend=backend, washout=75, alpha=1e-2)
    reservoir.fit(X_train, y_train)
    predictions = reservoir.predict(X_test)
    score = nrmse(y_test[75:], predictions)

    plt.figure(figsize=(10, 4))
    plt.plot(y_test[75:], label="Ground truth", linewidth=1.6)
    plt.plot(predictions, label="Prediction", linewidth=1.2)
    plt.title(f"Mackey-Glass Forecasting (NRMSE={score:.4f})")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    output_path = Path(__file__).with_name("timeseries_demo.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
