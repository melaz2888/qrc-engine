"""Backend-agnostic reservoir computing interface."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from qrc_engine.backends.base import BaseBackend
from qrc_engine.readout import OnlineReadoutLayer, ReadoutLayer
from qrc_engine.utils import as_float_array, nrmse

LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]


class Reservoir:
    """Unified reservoir API with a scikit-learn style workflow."""

    def __init__(
        self,
        backend: BaseBackend,
        washout: int = 50,
        readout: str = "ridge",
        alpha: float = 1.0,
        kernel_gamma: float = 1.0,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> None:
        """Initialize a reservoir wrapper."""

        self.backend = backend
        self.washout = washout
        self.readout_type = readout
        self.alpha = alpha
        self.kernel_gamma = kernel_gamma
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._readout: ReadoutLayer | None = None
        self._readout_online: OnlineReadoutLayer | None = None

    def _collect_states(self, inputs: FloatArray) -> FloatArray:
        """Run a sequence through the reservoir and collect feature vectors."""

        array = np.asarray(inputs, dtype=float)
        if array.ndim == 1:
            steps = [float(value) for value in array]
        elif array.ndim == 2:
            steps = [np.asarray(row, dtype=float) for row in array]
        else:
            raise ValueError("Input data must be 1D or 2D.")

        self.backend.reset()
        states: list[FloatArray] = []
        for index, value in enumerate(steps):
            state = np.asarray(self.backend.evolve(value), dtype=float).reshape(-1)
            if state.size != self.backend.state_dim:
                raise ValueError(
                    f"Backend returned {state.size} features, expected {self.backend.state_dim}."
                )
            if index >= self.washout:
                states.append(state)
        if not states:
            raise ValueError("Washout is too large for the provided sequence.")
        return np.vstack(states)

    def fit(self, X: FloatArray | list[float], y: FloatArray | list[float]) -> "Reservoir":
        """Collect states and fit the classical readout."""

        inputs = np.asarray(X, dtype=float)
        targets = as_float_array(y)
        if inputs.shape[0] != targets.size:
            raise ValueError("X and y must contain the same number of samples.")
        states = self._collect_states(inputs)
        trimmed_targets = targets[self.washout :]
        self._readout = ReadoutLayer(
            kind=self.readout_type,
            alpha=self.alpha,
            kernel_gamma=self.kernel_gamma,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
        ).fit(states, trimmed_targets)
        self._readout_online = None
        LOGGER.info("Fitted readout on %d reservoir states.", states.shape[0])
        return self

    def fit_online(
        self,
        X: FloatArray | list[float],
        y: FloatArray | list[float],
        q: float = 10.0,
        r: float = 1e-3,
    ) -> "Reservoir":
        """Collect states and fit the online Kalman-updated readout."""

        inputs = np.asarray(X, dtype=float)
        targets = as_float_array(y)
        if inputs.shape[0] != targets.size:
            raise ValueError("X and y must contain the same number of samples.")
        states = self._collect_states(inputs)
        trimmed_targets = targets[self.washout :]
        self._readout_online = OnlineReadoutLayer(state_dim=states.shape[1], q=q, r=r)
        for state, target in zip(states, trimmed_targets):
            self._readout_online.predict_one(state)
            self._readout_online.update(state, float(target))
        self._readout = None
        LOGGER.info("Fitted online readout on %d reservoir states.", states.shape[0])
        return self

    def predict(self, X: FloatArray | list[float]) -> FloatArray:
        """Predict targets for a new sequence."""

        if self._readout is None:
            raise RuntimeError("Reservoir is not fitted. Call fit() before predict().")
        inputs = np.asarray(X, dtype=float)
        states = self._collect_states(inputs)
        return self._readout.predict(states)

    def predict_online(self, X: FloatArray | list[float]) -> FloatArray:
        """Predict using the current online readout weights without updating them."""

        if self._readout_online is None:
            raise RuntimeError("Reservoir has no online readout. Call fit_online() first.")
        inputs = np.asarray(X, dtype=float)
        states = self._collect_states(inputs)
        return np.asarray([self._readout_online.predict_one(state) for state in states], dtype=float)

    def score(self, X: FloatArray | list[float], y: FloatArray | list[float]) -> float:
        """Compute NRMSE on a sequence."""

        targets = as_float_array(y)
        if self._readout_online is not None and self._readout is None:
            predictions = self.predict_online(X)
        else:
            predictions = self.predict(X)
        return nrmse(targets[self.washout :], predictions)

    def set_backend(self, backend: BaseBackend) -> None:
        """Swap to a different backend and clear the fitted readout."""

        self.backend = backend
        self._readout = None
        self._readout_online = None
        LOGGER.info("Swapped reservoir backend to %s.", type(backend).__name__)
