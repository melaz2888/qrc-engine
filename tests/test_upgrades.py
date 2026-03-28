"""Tests covering the qrc-engine v0.2 upgrade surface."""

from __future__ import annotations

import numpy as np
import pytest

from qrc_engine import Reservoir
from qrc_engine.backends.base import BaseBackend
from qrc_engine.readout import OnlineReadoutLayer, ReadoutLayer
from qrc_engine.tasks import lorenz_system
from qrc_engine.utils import permanent_ryser


QISKIT_REFERENCE = np.asarray(
    [
        0.0138420979461730,
        0.4501880377368368,
        -0.0987904564301863,
        -0.2998991862785218,
        0.1684127587525519,
        -0.0629042012230276,
    ],
    dtype=float,
)
PERCEVAL_REFERENCE = np.asarray(
    [
        0.4821757521403132,
        0.4480207733072025,
        0.0273057641306383,
        0.0424977104218460,
        0.7273776707651168,
        0.1318808717169261,
        -0.0646497980484653,
    ],
    dtype=float,
)
DYNAMIQS_REFERENCE = np.asarray(
    [
        9.9952655307237037e-01,
        4.7338572645604375e-04,
        6.1201173594722173e-08,
        -5.8232451038649534e-03,
        -1.5023222828067734e-06,
    ],
    dtype=float,
)


class MultivariateDummyBackend(BaseBackend):
    """Simple deterministic backend for multivariate reservoir tests."""

    def __init__(self) -> None:
        super().__init__(seed=0)
        self.initialize()

    def initialize(self) -> None:
        self._state = np.zeros(4, dtype=float)

    def evolve(self, input_val: float | np.ndarray) -> np.ndarray:
        vector = self._as_input_vector(input_val)
        padded = np.pad(vector, (0, max(0, 4 - vector.size)), mode="constant")[:4]
        self._state = (0.5 * self._state) + padded
        return self._state.copy()

    def reset(self) -> None:
        self._state = np.zeros(4, dtype=float)

    @property
    def state_dim(self) -> int:
        return 4


def test_multivariate_evolve() -> None:
    """Each backend should accept a vector input and emit the declared state dimension."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("perceval")
    pytest.importorskip("dynamiqs")
    from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend

    backends = [
        QiskitBackend(n_qubits=3, depth=2, seed=3),
        PercevalBackend(n_modes=4, n_photons=2, depth=2, seed=4),
        DynamiqsBackend(levels=3, seed=5),
    ]
    for backend in backends:
        state = backend.evolve(np.asarray([0.1, 0.2, -0.3], dtype=float))
        assert state.shape == (backend.state_dim,)


def test_scalar_backward_compat() -> None:
    """Default scalar mode should reproduce the saved v0.1-style reference outputs."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("perceval")
    pytest.importorskip("dynamiqs")
    from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend

    qiskit_state = QiskitBackend(n_qubits=3, depth=2, seed=3).evolve(0.25)
    perceval_state = PercevalBackend(n_modes=4, n_photons=2, depth=2, seed=4).evolve(0.12)
    dynamiqs_state = DynamiqsBackend(levels=3, seed=5).evolve(0.1)
    assert np.allclose(qiskit_state, QISKIT_REFERENCE)
    assert np.allclose(perceval_state, PERCEVAL_REFERENCE)
    assert np.allclose(dynamiqs_state, DYNAMIQS_REFERENCE)


def test_qiskit_persistent_state() -> None:
    """Persistent-state mode should make the second-step state depend on the prior statevector."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qrc_engine.backends import QiskitBackend

    backend_a = QiskitBackend(n_qubits=3, depth=2, persistent_state=True, seed=7)
    backend_b = QiskitBackend(n_qubits=3, depth=2, persistent_state=True, seed=7)
    backend_a.evolve(0.1)
    backend_b.evolve(-0.2)
    state_a = backend_a.evolve(0.3)
    state_b = backend_b.evolve(0.3)
    assert not np.allclose(state_a, state_b)


def test_qiskit_shots() -> None:
    """Shot-based features should be close to the exact expectations over many shots."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qrc_engine.backends import QiskitBackend

    exact_backend = QiskitBackend(n_qubits=3, depth=2, seed=11)
    sampled_backend = QiskitBackend(n_qubits=3, depth=2, seed=11, use_shots=True, shots=100000)
    for value in np.linspace(-0.4, 0.4, 20):
        exact = exact_backend.evolve(float(value))
        sampled = sampled_backend.evolve(float(value))
        assert np.allclose(exact, sampled, atol=0.05)


def test_perceval_fock_shape() -> None:
    """Fock mode should emit the declared feature shape and preserve state norm."""

    pytest.importorskip("perceval")
    from qrc_engine.backends import PercevalBackend

    backend = PercevalBackend(n_modes=5, n_photons=2, depth=2, fock_mode=True, seed=6)
    state = backend.evolve(0.2)
    assert state.shape == (backend.state_dim,)
    assert np.isclose(np.linalg.norm(backend._fock_state), 1.0)


def test_perceval_fock_single_photon() -> None:
    """Single-photon Fock occupations should match field probabilities exactly."""

    pytest.importorskip("perceval")
    from qrc_engine.backends import PercevalBackend

    field_backend = PercevalBackend(n_modes=4, n_photons=1, depth=2, seed=8)
    fock_backend = PercevalBackend(n_modes=4, n_photons=1, depth=2, fock_mode=True, seed=8)
    field_state = field_backend.evolve(0.12)
    fock_state = fock_backend.evolve(0.12)
    assert np.allclose(fock_state[: field_backend.n_modes], field_state[: field_backend.n_modes], atol=1e-10)


def test_perceval_feedback_resets() -> None:
    """Feedback mode should reset the stored Fock state after each step."""

    pytest.importorskip("perceval")
    from qrc_engine.backends import PercevalBackend

    backend = PercevalBackend(n_modes=4, n_photons=2, depth=2, fock_mode=True, feedback=True, seed=9)
    initial_state = backend._initial_fock_state.copy()
    backend.evolve(0.1)
    assert np.allclose(backend._fock_state, initial_state)
    backend.evolve(0.2)
    assert np.allclose(backend._fock_state, initial_state)


def test_dynamiqs_lindblad_trace() -> None:
    """Lindblad evolution should preserve unit trace over many steps."""

    pytest.importorskip("dynamiqs")
    from qrc_engine.backends import DynamiqsBackend

    backend = DynamiqsBackend(levels=3, lindblad=True, seed=10)
    for value in np.linspace(-0.2, 0.2, 100):
        backend.evolve(float(value))
    assert np.isclose(np.trace(backend._rho), 1.0)


def test_dynamiqs_lindblad_positive() -> None:
    """Lindblad evolution should keep the density matrix positive semidefinite."""

    pytest.importorskip("dynamiqs")
    from qrc_engine.backends import DynamiqsBackend

    backend = DynamiqsBackend(levels=3, lindblad=True, seed=10)
    for value in np.linspace(-0.2, 0.2, 100):
        backend.evolve(float(value))
    eigenvalues = np.linalg.eigvalsh(backend._rho)
    assert np.all(eigenvalues >= -1e-12)


def test_readout_random_forest() -> None:
    """Random-forest readout should fit and emit finite predictions."""

    rng = np.random.default_rng(2)
    states = rng.normal(size=(60, 5))
    targets = np.sin(states[:, 0]) + (0.5 * states[:, 1] ** 2)
    readout = ReadoutLayer(kind="random_forest", n_estimators=20, max_depth=5).fit(states, targets)
    predictions = readout.predict(states)
    assert predictions.shape == (60,)
    assert np.all(np.isfinite(predictions))


def test_readout_kernel_ridge() -> None:
    """Kernel-ridge readout should fit and emit finite predictions."""

    rng = np.random.default_rng(3)
    states = rng.normal(size=(50, 4))
    targets = np.cos(states[:, 0]) + states[:, 1]
    readout = ReadoutLayer(kind="kernel_ridge", alpha=1e-3, kernel_gamma=0.5).fit(states, targets)
    predictions = readout.predict(states)
    assert predictions.shape == (50,)
    assert np.all(np.isfinite(predictions))


def test_online_readout() -> None:
    """Online readout should track a drifting linear target with finite error."""

    rng = np.random.default_rng(4)
    features = rng.normal(size=(120, 3))
    readout = OnlineReadoutLayer(state_dim=3, q=0.1, r=1e-2)
    predictions: list[float] = []
    targets: list[float] = []
    for step, feature in enumerate(features):
        weights = np.asarray([0.4 + (0.002 * step), -0.3, 0.2], dtype=float)
        target = float(feature @ weights + (0.1 * np.sin(step / 10.0)))
        predictions.append(readout.predict_one(feature))
        targets.append(target)
        readout.update(feature, target)
    rmse = float(np.sqrt(np.mean((np.asarray(predictions[20:]) - np.asarray(targets[20:])) ** 2)))
    assert np.isfinite(rmse)
    assert rmse < 1.0


def test_reservoir_multivariate() -> None:
    """Reservoir.fit should work with 2D inputs."""

    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 3))
    y = X[:, 0] - (0.5 * X[:, 1]) + (0.25 * X[:, 2])
    reservoir = Reservoir(backend=MultivariateDummyBackend(), washout=5, alpha=1e-3)
    reservoir.fit(X, y)
    predictions = reservoir.predict(X)
    assert predictions.shape == (75,)


def test_reservoir_online_fit() -> None:
    """Online fit should produce finite online predictions."""

    rng = np.random.default_rng(6)
    X = rng.normal(size=(90, 2))
    y = (0.6 * X[:, 0]) - (0.2 * X[:, 1])
    reservoir = Reservoir(backend=MultivariateDummyBackend(), washout=4)
    reservoir.fit_online(X, y, q=0.5, r=1e-2)
    predictions = reservoir.predict_online(X)
    assert np.all(np.isfinite(predictions))


def test_permanent_ryser() -> None:
    """Ryser permanent should match a known 3x3 reference value."""

    matrix = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [0.0, 4.0, 5.0],
            [1.0, 0.0, 6.0],
        ],
        dtype=np.complex128,
    )
    assert np.isclose(permanent_ryser(matrix), 46.0 + 0.0j)


def test_lorenz_task() -> None:
    """Lorenz task should return finite multivariate arrays with aligned shapes."""

    X_train, y_train, X_test, y_test = lorenz_system(n_samples=200, split=0.75, seed=7)
    assert X_train.shape == (150, 3)
    assert X_test.shape == (50, 3)
    assert y_train.shape == (150,)
    assert y_test.shape == (50,)
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(y_test))


def test_phase_noise() -> None:
    """Phase noise should alter the photonic backend output."""

    pytest.importorskip("perceval")
    from qrc_engine.backends import PercevalBackend

    clean = PercevalBackend(n_modes=4, n_photons=2, depth=2, seed=12)
    noisy = PercevalBackend(n_modes=4, n_photons=2, depth=2, phase_noise_std=0.05, seed=12)
    assert not np.allclose(clean.evolve(0.15), noisy.evolve(0.15))


def test_gate_noise() -> None:
    """Gate noise should alter the Qiskit backend output."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    from qrc_engine.backends import QiskitBackend

    clean = QiskitBackend(n_qubits=3, depth=2, seed=13)
    noisy = QiskitBackend(n_qubits=3, depth=2, gate_noise_std=0.05, seed=13)
    assert not np.allclose(clean.evolve(0.2), noisy.evolve(0.2))


def test_metadata() -> None:
    """Each backend metadata dictionary should expose the required keys."""

    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("perceval")
    pytest.importorskip("dynamiqs")
    from qrc_engine.backends import DynamiqsBackend, PercevalBackend, QiskitBackend

    required = {"paradigm", "state_type", "has_noise", "has_persistent_state"}
    backends = [
        QiskitBackend(n_qubits=2, depth=1, seed=1),
        PercevalBackend(n_modes=3, n_photons=1, depth=1, seed=1),
        DynamiqsBackend(levels=2, seed=1),
    ]
    for backend in backends:
        metadata = backend.metadata
        assert required.issubset(metadata.keys())
