# qrc-engine

`qrc-engine` is a multi-backend quantum reservoir computing library. Its main idea is simple: keep the reservoir workflow fixed, and let the reservoir itself change underneath.

With the same `Reservoir` API, you can define and train:
- a gate-based reservoir built from parameterized circuits
- a photonic reservoir built from interferometers and Fock-space evolution
- an open-system reservoir built from density-matrix dynamics

The goal of the project is not to lock reservoir computing to one simulator stack or one physical picture. It is to make different reservoir paradigms comparable under the same fit / predict loop.

## What This Repo Is About

Most quantum reservoir code is tightly coupled to a single backend style. In practice, gate-based, photonic, and open-system models all have different internal states, evolution rules, and feature extraction strategies.

`qrc-engine` wraps those differences behind one backend contract:
- `initialize()`
- `evolve(input_t)`
- `reset()`
- `state_dim`
- `metadata`

That means the user-facing workflow stays the same even when the reservoir definition changes completely.

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
from qrc_engine.backends import DynamiqsBackend, QiskitBackend
from qrc_engine.tasks import mackey_glass

X_train, y_train, X_test, y_test = mackey_glass(n_samples=2000, split=0.8, seed=11)

reservoir = Reservoir(
    backend=QiskitBackend(n_qubits=4, depth=4, seed=11),
    washout=50,
    alpha=1e-3,
)
reservoir.fit(X_train, y_train)
print(reservoir.score(X_test, y_test))

# Swap the reservoir definition without changing the workflow
reservoir.set_backend(DynamiqsBackend(levels=5, dt=0.3, gamma=0.1, seed=3))
reservoir.fit(X_train, y_train)
print(reservoir.score(X_test, y_test))
```

## Same Workflow, Different Reservoirs

```python
from qrc_engine import Reservoir
from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend

reservoir = Reservoir(backend=QiskitBackend(n_qubits=4, depth=4), washout=50)

# Gate-based circuit reservoir
reservoir.set_backend(QiskitBackend(n_qubits=4, depth=4, persistent_state=True))

# Photonic interferometer reservoir
reservoir.set_backend(PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True))

# Open-system density-matrix reservoir
reservoir.set_backend(DynamiqsBackend(levels=4, lindblad=True, n_subsystems=2))
```

The training code does not need to know whether the backend state is a statevector, a Fock-state amplitude vector, or a density matrix. It only sees a stream of reservoir features.

## Architecture

```text
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
  ReadoutLayer (batch) or OnlineReadoutLayer
```

## Backends

| Backend | Paradigm | Modes | Dependency |
|---------|----------|-------|------------|
| `QiskitBackend` | Gate-based circuit reservoir | fresh statevector, persistent statevector, shot-based readout | `qiskit`, `qiskit-aer` |
| `PercevalBackend` | Photonic reservoir | classical field, Fock-space, feedback-driven Fock mode | `perceval-quandela` |
| `DynamiqsBackend` | Open-system reservoir | convex-mixture dissipation, Lindblad evolution, multi-subsystem mode | `dynamiqs` |

Each backend exposes `metadata` such as:
- `paradigm`
- `state_type`
- `has_noise`
- `has_persistent_state`

This makes it possible to write capability-aware experiments without hard-coding backend types throughout the workflow.

## Readouts

Batch readouts:
- `ridge`
- `linear`
- `kernel_ridge`
- `random_forest`

Online readout:
- `fit_online(X, y, q, r)`
- `predict_online(X)`

## Tasks

| Task | Description | Input dim |
|------|-------------|-----------|
| `narma10` | Nonlinear autoregressive moving average order 10 | scalar |
| `mackey_glass` | Mackey-Glass delay-differential equation | scalar |
| `sine_forecasting` | One-step-ahead sinusoid | scalar |
| `lorenz_system` | 3-D Lorenz attractor, predict `x(t+1)` | 3-D vector |

## Notebooks And Experiments

The repository includes:
- synthetic evaluation notebooks for the core QRC workflows
- real-world electric load forecasting notebooks
- a by-hour variant where separate model families are trained for different times of day

See:
- `notebooks/qrc_engine_evaluation.ipynb`
- `notebooks/load_forecasting_evaluation.ipynb`
- `notebooks/load_forecasting_by_hour_evaluation.ipynb`

## Reproducing The Main Notebook

```bash
python scripts/generate_evaluation_notebook.py
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/qrc_engine_evaluation.ipynb
```

On Windows:

```powershell
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/qrc_engine_evaluation.ipynb
```
