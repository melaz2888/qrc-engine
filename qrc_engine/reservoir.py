"""Backend-agnostic reservoir computing interface."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from qrc_engine.backends.base import BaseBackend
from qrc_engine.readout import ReadoutLayer
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
    ) -> None:
        """Initialize a reservoir wrapper.

        Args:
            backend: Quantum backend used to evolve reservoir states.
            washout: Number of initial steps to discard.
            readout: Readout model name.
            alpha: Ridge regularization strength.
        """

        self.backend = backend
        self.washout = washout
        self.readout_type = readout
        self.alpha = alpha
        self._readout: ReadoutLayer | None = None

    def _collect_states(self, inputs: FloatArray) -> FloatArray:
        """Run a sequence through the reservoir and collect feature vectors."""

        self.backend.reset()
        states: list[FloatArray] = []
        for index, value in enumerate(inputs):
            state = np.asarray(self.backend.evolve(float(value)), dtype=float).reshape(-1)
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
        """Collect states and fit the classical readout.

        Args:
            X: Input sequence.
            y: Target sequence aligned with X.

        Returns:
            The fitted reservoir.
        """

        inputs = as_float_array(X)
        targets = as_float_array(y)
        if inputs.size != targets.size:
            raise ValueError("X and y must contain the same number of samples.")
        states = self._collect_states(inputs)
        trimmed_targets = targets[self.washout :]
        self._readout = ReadoutLayer(kind=self.readout_type, alpha=self.alpha).fit(
            states, trimmed_targets
        )
        LOGGER.info("Fitted readout on %d reservoir states.", states.shape[0])
        return self

    def predict(self, X: FloatArray | list[float]) -> FloatArray:
        """Predict targets for a new sequence.

        Args:
            X: Input sequence.

        Returns:
            Predicted values after washout.
        """

        if self._readout is None:
            raise RuntimeError("Reservoir is not fitted. Call fit() before predict().")
        inputs = as_float_array(X)
        states = self._collect_states(inputs)
        return self._readout.predict(states)

    def score(self, X: FloatArray | list[float], y: FloatArray | list[float]) -> float:
        """Compute NRMSE on a sequence.

        Args:
            X: Input sequence.
            y: Ground-truth target sequence.

        Returns:
            Normalized RMSE after washout.
        """

        targets = as_float_array(y)
        predictions = self.predict(X)
        return nrmse(targets[self.washout :], predictions)

    def set_backend(self, backend: BaseBackend) -> None:
        """Swap to a different backend and clear the fitted readout.

        Args:
            backend: New backend instance.
        """

        self.backend = backend
        self._readout = None
        LOGGER.info("Swapped reservoir backend to %s.", type(backend).__name__)
