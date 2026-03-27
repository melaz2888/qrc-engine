"""Backend contract tests."""

from __future__ import annotations

import numpy as np
import pytest


def test_qiskit_backend_shape() -> None:
    """Qiskit backend should emit the declared number of features."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qrc_engine.backends.qiskit_backend import QiskitBackend

    backend = QiskitBackend(n_qubits=3, depth=2, seed=3)
    state = backend.evolve(0.25)
    assert state.shape == (backend.state_dim,)
    backend.reset()
    reset_state = backend.evolve(0.25)
    assert np.allclose(state, reset_state)


def test_perceval_backend_shape() -> None:
    """Perceval backend should emit normalized probabilities plus coherences."""

    pytest.importorskip("perceval")
    from qrc_engine.backends.perceval_backend import PercevalBackend

    backend = PercevalBackend(n_modes=4, n_photons=2, depth=2, seed=4)
    state = backend.evolve(0.12)
    assert state.shape == (backend.state_dim,)
    assert np.isclose(np.sum(state[: backend.n_modes]), 1.0)


def test_dynamiqs_backend_shape() -> None:
    """dynamiqs backend should emit population and coherence features."""

    pytest.importorskip("dynamiqs")
    from qrc_engine.backends.dynamiqs_backend import DynamiqsBackend

    backend = DynamiqsBackend(levels=3, seed=5)
    state = backend.evolve(0.1)
    assert state.shape == (backend.state_dim,)
    assert np.all(np.isfinite(state))
