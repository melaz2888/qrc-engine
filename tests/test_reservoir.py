"""Reservoir API tests."""

from __future__ import annotations

import numpy as np

from qrc_engine import Reservoir
from qrc_engine.backends.base import BaseBackend


class DummyBackend(BaseBackend):
    """Small deterministic backend for API testing."""

    def __init__(self, gain: float = 1.0) -> None:
        super().__init__(seed=0)
        self.gain = gain
        self.initialize()

    def initialize(self) -> None:
        self._state = np.zeros(3, dtype=float)

    def evolve(self, input_val: float) -> np.ndarray:
        self._state = 0.7 * self._state + self.gain * np.asarray(
            [np.tanh(input_val), np.sin(input_val), np.cos(input_val)],
            dtype=float,
        )
        return self._state.copy()

    def reset(self) -> None:
        self._state = np.zeros(3, dtype=float)

    @property
    def state_dim(self) -> int:
        return 3


def test_reservoir_fit_predict_score() -> None:
    """Reservoir should fit and predict end-to-end."""

    X = np.linspace(0.0, 1.0, 80)
    y = np.sin(2.0 * X)
    reservoir = Reservoir(backend=DummyBackend(), washout=5, alpha=1e-3)
    reservoir.fit(X, y)
    predictions = reservoir.predict(X)
    assert predictions.shape == (75,)
    assert np.isfinite(reservoir.score(X, y))


def test_switching_backends_clears_readout() -> None:
    """Switching backends should invalidate the fitted readout."""

    X = np.linspace(0.0, 1.0, 60)
    y = np.cos(X)
    reservoir = Reservoir(backend=DummyBackend(gain=1.0), washout=4)
    reservoir.fit(X, y)
    reservoir.set_backend(DummyBackend(gain=0.5))
    try:
        reservoir.predict(X)
    except RuntimeError:
        pass
    else:
        raise AssertionError("predict() should fail after set_backend() resets the readout.")
