# qrc-engine

`qrc-engine` is a hardware-agnostic quantum reservoir computing library. Define a reservoir workflow once, then swap the backend between gate-based, photonic, and open-system simulators without changing the training loop.

## Benchmark Results

All results use 2 000 samples (80/20 split), ridge readout, tuned regularisation. Lower NRMSE is better; > 1.0 means worse than predicting the mean.

| Task | Qiskit | Dynamiqs | Perceval (field) | Perceval (Fock) |
|------|--------|----------|-----------------|-----------------|
| Mackey-Glass (chaotic) | **0.07** | 0.23 | 0.83 | 1.00 |
| NARMA-10 (nonlinear memory) | 0.81 | **0.75** | 1.05 | **0.99** |
| Lorenz (multivariate 3-D) | **0.28** | 0.46 | — | — |

See `notebooks/qrc_engine_evaluation.ipynb` for the full executed notebook with plots.

## Installation

```bash
pip install qrc-engine[qiskit]      # gate-based backend
pip install qrc-engine[dynamiqs]    # open-system backend
pip install qrc-engine[perceval]    # photonic backend
pip install qrc-engine[all]         # everything
```

## Quickstart

```python
from qrc_engine import Reservoir
from qrc_engine.backends import QiskitBackend, DynamiqsBackend
from qrc_engine.tasks import mackey_glass, narma10

# --- Mackey-Glass: one-step-ahead chaotic forecasting ---
X_train, y_train, X_test, y_test = mackey_glass(n_samples=2000, split=0.8, seed=11)

reservoir = Reservoir(
    backend=QiskitBackend(n_qubits=4, depth=4, seed=11),
    washout=50,
    alpha=1e-3,
)
reservoir.fit(X_train, y_train)
print(f"Mackey-Glass NRMSE: {reservoir.score(X_test, y_test):.4f}")  # ~0.07

# --- Swap backend without changing training code ---
reservoir.set_backend(DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3))
reservoir.fit(X_train, y_train)
print(f"Dynamiqs NRMSE: {reservoir.score(X_test, y_test):.4f}")  # ~0.23

# --- Multivariate Lorenz ---
from qrc_engine.tasks import lorenz_system

X_train, y_train, X_test, y_test = lorenz_system(n_samples=2000, split=0.8, seed=7)
reservoir = Reservoir(
    backend=QiskitBackend(n_qubits=5, depth=3, seed=5),
    washout=50,
    alpha=1e-3,
)
reservoir.fit(X_train, y_train)
print(f"Lorenz NRMSE: {reservoir.score(X_test, y_test):.4f}")  # ~0.28
```

## Architecture

```
Reservoir.fit / fit_online / predict / predict_online
        |
        v
  BaseBackend.evolve(input_t)          # scalar or feature vector
        |
        +-- QiskitBackend   -> gate-based statevector features
        +-- PercevalBackend -> photonic field or Fock-space features
        +-- DynamiqsBackend -> open-system density-matrix features
        |
        v
  ReadoutLayer (batch ridge/kernel/RF) or OnlineReadoutLayer (Kalman)
```

## Backends

| Backend | Paradigm | Modes | Dependency |
|---------|----------|-------|------------|
| `QiskitBackend` | Gate-based circuit | fresh statevector · persistent statevector · shot-based | `qiskit`, `qiskit-aer` |
| `PercevalBackend` | Photonic interferometer | classical field · Fock-space (Ryser permanent) · feedback Fock | `perceval-quandela` |
| `DynamiqsBackend` | Open quantum system | convex-mixture dissipation · Lindblad master eq. · multi-subsystem | `dynamiqs` |

Every backend exposes a `metadata` dict (`paradigm`, `state_type`, `has_noise`, `has_persistent_state`) so you can write capability-aware code without `isinstance` checks.

## Readout

Batch: `ridge` (default), `linear`, `kernel_ridge`, `random_forest`

Online: `fit_online(X, y, q, r)` + `predict_online(X)` — Kalman-updated weights, no matrix inversion, suitable for streaming data.

## Tasks

| Task | Description | Input dim |
|------|-------------|-----------|
| `narma10` | Nonlinear autoregressive moving average order 10 | scalar |
| `mackey_glass` | Mackey-Glass delay-differential equation | scalar |
| `sine_forecasting` | One-step-ahead sinusoid | scalar |
| `lorenz_system` | 3-D Lorenz attractor, predict x(t+1) | 3-D vector |

## Reproducing the Evaluation

```bash
# regenerate the notebook source
python scripts/generate_evaluation_notebook.py

# execute it
python -m jupyter nbconvert --to notebook --execute --inplace \
    notebooks/qrc_engine_evaluation.ipynb
```

On Windows:
```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/qrc_engine_evaluation.ipynb
```
