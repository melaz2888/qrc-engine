# qrc-engine Architecture

## Problem

Quantum reservoir computing code is usually tied to one simulator stack at a time. Gate-based, photonic, and open-system frameworks all expose different primitives, state representations, and execution paths, so benchmark code often gets rewritten backend by backend.

## Solution

`qrc-engine` introduces a single `BaseBackend` interface and a single `Reservoir` workflow. The reservoir owns the sequence logic, washout handling, state collection, and classical readout fitting. Each backend only needs to answer three questions: how to initialize the reservoir, how to evolve it for one scalar input, and how to reset it for a fresh sequence.

The core flow is:

1. Encode `x_t` into the backend.
2. Evolve the backend state with backend-specific dynamics.
3. Extract a fixed-width feature vector.
4. Fit a linear or ridge readout on the collected states.

This keeps the user API close to scikit-learn while isolating backend-specific logic behind a narrow contract.

## Why These Three Backends

- `QiskitBackend` represents a gate-based circuit reservoir with random fixed parameters and entangling layers.
- `PercevalBackend` represents a photonic interferometer reservoir with phase modulation and beam-splitter mixing.
- `DynamiqsBackend` represents an open quantum system reservoir with density-matrix evolution and dissipative dynamics.

Together they cover three genuinely different quantum computing paradigms while preserving the same fit/predict loop.

## Trade-offs and Limitations

This project is a reference library, not a production orchestration system. The current backends are intentionally small and deterministic so examples remain readable and reproducible. The Qiskit backend is closest to a native execution path. The Perceval and dynamiqs adapters keep the same abstraction boundary but use compact reference evolutions rather than full-blown workflow orchestration across every capability of those ecosystems.

That trade-off keeps the codebase lean and portfolio-friendly, but it also means:

- no real QPU submission layer yet
- no batching or asynchronous execution
- no circuit optimization or backend-aware compilation passes
- no advanced dataset or hyperparameter management

## Future Directions

- Add real backend adapters for IBM Runtime, IQM, and Quandela cloud execution.
- Extend the feature-extraction API to support shot-based observables and richer tomography-like summaries.
- Add circuit and pulse optimization hooks before execution.
- Support multivariate inputs, parameter sweeps, and reproducible benchmark harnesses.
- Introduce backend capability metadata so workflows can adapt automatically to qubit count, connectivity, or photonic mode limits.
