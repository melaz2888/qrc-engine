# qrc-engine Architecture

## Backend Contract

`qrc-engine` keeps a single `BaseBackend` contract and a single `Reservoir` workflow. In v0.2 the backend contract accepts either a scalar input or a multivariate feature vector at each time step:

1. `initialize()` builds the backend-specific state.
2. `evolve(input_val)` advances the backend by one step and returns a fixed-width feature vector.
3. `reset()` clears the persistent backend state for a fresh sequence.
4. `state_dim` reports the feature width.
5. `metadata` reports backend capabilities:
   - `paradigm`
   - `state_type`
   - `has_noise`
   - `has_persistent_state`

This keeps the public reservoir API stable while letting downstream scripts adapt to backend capabilities without `isinstance` checks.

## Reservoir Flow

The reservoir owns sequence handling, washout, state collection, and readout fitting:

1. Accept either a 1D scalar sequence or a 2D multivariate sequence.
2. Pass each time step to `backend.evolve(...)`.
3. Discard the washout prefix.
4. Fit either a batch readout (`ridge`, `linear`, `kernel_ridge`, `random_forest`) or an online Kalman-updated readout.

`fit_online` uses an `OnlineReadoutLayer` that treats the readout weights as a random walk and updates them with a Kalman-style correction after each observation.

## Backend Evolution Modes

### QiskitBackend

The gate-based backend now supports three execution modes:

- Default mode: rebuild a fresh circuit from `|0...0>` at each step, matching v0.1 behavior.
- Persistent-state mode: carry the output statevector forward so the next step evolves `|psi_t>` rather than restarting from the ground state.
- Shot-based mode: sample measurement outcomes from the statevector probabilities and estimate expectation features from counts.

It also supports additive Gaussian gate noise on every rotation angle.

### PercevalBackend

The photonic backend now supports three modes:

- Classical-field mode: propagate a complex field through phase shifters and beam splitters, matching v0.1 behavior.
- Fock-space mode: evolve a small symmetric Fock-space state vector and extract occupation expectations and nearest-neighbor occupation correlations.
- Feedback mode: reset to the initial Fock state after each step and feed the measured occupations back into the next phase encoding.

The Fock-space lift uses the linear-optics permanent formula. For a mode-space unitary `U`, each Fock-basis transition amplitude is computed from a repeated-row, repeated-column submatrix and a permanent:

`<n'|U_hat|n> = perm(U_sub) / sqrt(prod_j n_j! * prod_k n'_k!)`

`qrc_engine.utils.permanent_ryser` implements Ryser's formula in pure NumPy for the small systems used here.

### DynamiqsBackend

The open-system backend now supports:

- Default convex-mixture dissipation, matching v0.1 behavior.
- Lindblad mode, integrated with a single RK4 step of the Lindblad master equation.
- Multi-subsystem mode, where the Hilbert space is extended to `(C^d)^⊗n`, the Hamiltonian includes local terms plus nearest-neighbor couplings, and jump operators act locally on each subsystem.

## Noise Models

Two backend-specific noise models are included in v0.2:

- Qiskit gate noise:
  - `theta_applied = theta_target + epsilon`
  - `epsilon ~ N(0, sigma^2)`
- Perceval phase noise and phase resolution:
  - `phi_applied = round(phi_target / Delta) * Delta + epsilon`
  - `epsilon ~ N(0, sigma_phi^2)`

All stochastic paths use seeded NumPy generators so runs remain deterministic for a fixed configuration.

## Feature Shapes

The reservoir always receives a fixed-width real feature vector:

- `QiskitBackend`: qubit expectations and pair correlations, or full basis probabilities.
- `PercevalBackend`: mode occupations plus nearest-neighbor correlations/coherences.
- `DynamiqsBackend`: populations plus nearest-neighbor coherences in the working Hilbert basis.

The exact physics differs by mode, but the downstream readout stays unchanged.
