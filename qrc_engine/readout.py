"""Classical readout layers for reservoir states."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, Ridge

FloatArray = NDArray[np.float64]


class ReadoutLayer:
    """Thin scikit-learn style wrapper around linear readouts."""

    def __init__(self, kind: Literal["ridge", "linear"] = "ridge", alpha: float = 1.0) -> None:
        """Initialize the readout layer.

        Args:
            kind: Readout model family.
            alpha: L2 regularization strength for ridge regression.
        """

        self.kind = kind
        self.alpha = alpha
        if kind == "ridge":
            self._model = Ridge(alpha=alpha)
        elif kind == "linear":
            self._model = LinearRegression()
        else:
            raise ValueError("readout must be 'ridge' or 'linear'.")

    def fit(self, states: FloatArray, targets: FloatArray) -> "ReadoutLayer":
        """Fit the readout weights.

        Args:
            states: Reservoir feature matrix.
            targets: Target values aligned with states.

        Returns:
            The fitted readout.
        """

        self._model.fit(states, targets)
        return self

    def predict(self, states: FloatArray) -> FloatArray:
        """Predict target values from reservoir states.

        Args:
            states: Reservoir feature matrix.

        Returns:
            Predicted targets.
        """

        return np.asarray(self._model.predict(states), dtype=float)
