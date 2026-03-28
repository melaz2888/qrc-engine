# qrc-engine

`qrc-engine` is a hardware-agnostic quantum reservoir computing library. It lets you define a reservoir workflow once, then swap the backend between gate-based, photonic, and open-system simulators without changing the training loop. The package stays compact, but it now supports persistent-state evolution, Fock-space photonic dynamics, multivariate inputs, online readouts, optional noise models, and deterministic tests.

## Installation

```bash
pip install qrc-engine
pip install qrc-engine[qiskit]
pip install qrc-engine[perceval]
pip install qrc-engine[dynamiqs]
pip install qrc-engine[all]

# repo-local installs
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-all.txt
```

## Quickstart

```python
from qrc_engine import Reservoir
from qrc_engine.backends import PercevalBackend, QiskitBackend
from qrc_engine.tasks import lorenz_system, narma10

X_train, y_train, X_test, y_test = narma10(n_samples=500, split=0.8, seed=11)

backend = QiskitBackend(n_qubits=3, depth=3, persistent_state=True, seed=11)
reservoir = Reservoir(backend=backend, washout=50, alpha=1e-2)
reservoir.fit(X_train, y_train)
print(f"Qiskit NRMSE: {reservoir.score(X_test, y_test):.4f}")

reservoir.set_backend(
    PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True, feedback=True, seed=3)
)
reservoir.fit(X_train, y_train)
print(f"Perceval Fock NRMSE: {reservoir.score(X_test, y_test):.4f}")

X_train_mv, y_train_mv, X_test_mv, y_test_mv = lorenz_system(n_samples=1200, split=0.8, seed=7)
online_reservoir = Reservoir(
    backend=QiskitBackend(n_qubits=3, depth=2, persistent_state=True, seed=5),
    washout=40,
)
online_reservoir.fit_online(X_train_mv, y_train_mv, q=1.0, r=1e-2)
predictions = online_reservoir.predict_online(X_test_mv)
print(predictions[:5])
```

## Architecture

```text
Reservoir.fit / fit_online / predict
        |
        v
  BaseBackend.evolve(input_t)
        |
        +-- QiskitBackend   -> gate-based circuit features
        +-- PercevalBackend -> photonic field or Fock-space features
        +-- DynamiqsBackend -> open-system observable features
        |
        v
  ReadoutLayer / OnlineReadoutLayer
```

## Backend Comparison

| Backend | Paradigm | State options | Optional dependency |
| --- | --- | --- | --- |
| `QiskitBackend` | Gate-based circuit reservoir | fresh statevector, persistent statevector, shot-based estimation | `qiskit`, `qiskit-aer` |
| `PercevalBackend` | Photonic interferometer reservoir | classical field, Fock space, feedback-driven Fock mode | `perceval-quandela` |
| `DynamiqsBackend` | Open quantum system reservoir | convex-mixture dissipation, Lindblad, multi-subsystem | `dynamiqs` |

## Readout Options

`Reservoir` supports the following batch readouts:

- `ridge`
- `linear`
- `kernel_ridge`
- `random_forest`

It also supports `fit_online(...)` and `predict_online(...)` through a Kalman-updated `OnlineReadoutLayer`.

## Tasks

Built-in tasks include:

- `narma10`
- `mackey_glass`
- `sine_forecasting`
- `lorenz_system`

`lorenz_system` exercises the multivariate input path by returning a 3D state vector at each time step and the next-step `x` component as the target.

## Examples

- `python examples/quickstart.py`
- `python examples/benchmark_backends.py`
- `python examples/benchmark_v2.py`
- `python examples/timeseries_demo.py`
- `python scripts/generate_evaluation_notebook.py`

## Evaluation Notebook

An executed notebook is included at `notebooks/qrc_engine_evaluation.ipynb`. It benchmarks the backends on NARMA-10 and runs a Mackey-Glass forecasting pass with the same public API.

PowerShell execution on Windows:

```powershell
$env:JUPYTER_CONFIG_DIR="$PWD\.jupyter"
$env:JUPYTER_DATA_DIR="$env:TEMP\qrc-engine-jupyter-data"
$env:JUPYTER_RUNTIME_DIR="$env:TEMP\qrc-engine-jupyter-runtime"
$env:JUPYTER_ALLOW_INSECURE_WRITES="1"
$env:IPYTHONDIR="$env:TEMP\qrc-engine-ipython"
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/qrc_engine_evaluation.ipynb
```
