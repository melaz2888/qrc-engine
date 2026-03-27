# qrc-engine

`qrc-engine` is a hardware-agnostic quantum reservoir computing library. It lets you define a reservoir workflow once, then swap the backend between gate-based, photonic, and open-system simulators without changing the training loop. The package is intentionally small, but it is structured like an infrastructure library: typed backend contracts, deterministic tasks, optional backend dependencies, automated tests, examples, CI, and an executed evaluation notebook.

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
from qrc_engine.backends import QiskitBackend, PercevalBackend
from qrc_engine.tasks import narma10

backend = QiskitBackend(n_qubits=3, depth=3, shots=1024, seed=11)
reservoir = Reservoir(backend=backend, washout=50, alpha=1e-2)

X_train, y_train, X_test, y_test = narma10(n_samples=500, split=0.8, seed=11)
reservoir.fit(X_train, y_train)
print(f"Qiskit NRMSE: {reservoir.score(X_test, y_test):.4f}")

reservoir.set_backend(PercevalBackend(n_modes=5, n_photons=2, depth=2, seed=3))
reservoir.fit(X_train, y_train)
print(f"Perceval NRMSE: {reservoir.score(X_test, y_test):.4f}")
```

## Architecture

```text
Reservoir.fit / predict
        |
        v
  BaseBackend.evolve(input_t)
        |
        +-- QiskitBackend   -> gate-based circuit features
        +-- PercevalBackend -> photonic interferometer features
        +-- DynamiqsBackend -> open-system observable features
        |
        v
  ReadoutLayer (ridge / linear)
```

## Backend Comparison

| Backend | Paradigm | Feature output | Optional dependency |
| --- | --- | --- | --- |
| `QiskitBackend` | Gate-based circuit reservoir | Pauli-Z expectations, ZZ correlations, or basis probabilities | `qiskit`, `qiskit-aer` |
| `PercevalBackend` | Photonic interferometer reservoir | Mode occupation probabilities and adjacent coherences | `perceval-quandela` |
| `DynamiqsBackend` | Open quantum system reservoir | Populations and coherences | `dynamiqs` |

The backends keep a compact internal memory so consecutive inputs produce non-trivial reservoir dynamics rather than independent one-shot outputs.

## Representative Results

The current benchmark settings in `examples/benchmark_backends.py` produce deterministic local results on NARMA-10:

| Backend | NRMSE | Time (s) |
| --- | ---: | ---: |
| `Qiskit (3q, d3)` | `0.741` | `0.68` |
| `Perceval (5m)` | `0.929` | `0.06` |
| `dynamiqs (4lvl)` | `0.674` | `0.04` |

These are not meant as absolute research numbers. They are reproducible reference numbers for the current compact backend implementations.

## Why This Project

Quantum reservoir computing becomes infrastructure-heavy as soon as each backend needs a different execution model, data path, and feature extraction strategy. `qrc-engine` is deliberately small, but it shows the abstraction layer that makes backend switching possible: one reservoir API, one readout pipeline, multiple quantum paradigms underneath.

## Project Layout

```text
qrc-engine/
|-- qrc_engine/
|   |-- reservoir.py
|   |-- readout.py
|   |-- backends/
|   `-- tasks/
|-- examples/
|-- notebooks/
|-- tests/
`-- docs/
```

## Examples

- `python examples/quickstart.py`
- `python examples/benchmark_backends.py`
- `python examples/timeseries_demo.py`
- `python scripts/generate_evaluation_notebook.py`

`examples/benchmark_backends.py` prints an NRMSE/time table and saves a comparison figure as `benchmark_backends.png`.

## Evaluation Notebook

An executed notebook is included at `notebooks/qrc_engine_evaluation.ipynb`. It benchmarks all three backends on NARMA-10 and runs a Mackey-Glass forecasting pass with the same public API.

PowerShell execution on Windows:

```powershell
$env:JUPYTER_CONFIG_DIR="$PWD\.jupyter"
$env:JUPYTER_DATA_DIR="$env:TEMP\qrc-engine-jupyter-data"
$env:JUPYTER_RUNTIME_DIR="$env:TEMP\qrc-engine-jupyter-runtime"
$env:JUPYTER_ALLOW_INSECURE_WRITES="1"
$env:IPYTHONDIR="$env:TEMP\qrc-engine-ipython"
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/qrc_engine_evaluation.ipynb
```
