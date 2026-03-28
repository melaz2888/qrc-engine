"""Classical readout layers for reservoir states."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge

FloatArray = NDArray[np.float64]


class ReadoutLayer:
    """Thin scikit-learn style wrapper around classical readouts."""

    def __init__(
        self,
        kind: Literal["ridge", "linear", "kernel_ridge", "random_forest"] = "ridge",
        alpha: float = 1.0,
        kernel_gamma: float = 1.0,
        n_estimators: int = 100,
        max_depth: int | None = None,
    ) -> None:
        """Initialize the readout layer."""

        self.kind = kind
        self.alpha = alpha
        self.kernel_gamma = kernel_gamma
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        if kind == "ridge":
            self._model = Ridge(alpha=alpha)
        elif kind == "linear":
            self._model = LinearRegression()
        elif kind == "kernel_ridge":
            self._model = KernelRidge(alpha=alpha, kernel="rbf", gamma=kernel_gamma)
        elif kind == "random_forest":
            self._model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=0,
            )
        else:
            raise ValueError("readout must be 'ridge', 'linear', 'kernel_ridge', or 'random_forest'.")

    def fit(self, states: FloatArray, targets: FloatArray) -> "ReadoutLayer":
        """Fit the readout weights."""

        self._model.fit(states, targets)
        return self

    def predict(self, states: FloatArray) -> FloatArray:
        """Predict target values from reservoir states."""

        return np.asarray(self._model.predict(states), dtype=float)


class OnlineReadoutLayer:
    """Kalman-filter-based online readout with a random-walk weight model."""

    def __init__(self, state_dim: int, q: float = 10.0, r: float = 1e-3) -> None:
        """Initialize the online readout state."""

        self.theta = np.zeros(state_dim + 1, dtype=float)
        self.P = np.eye(state_dim + 1, dtype=float) * 1e3
        self.Q = np.eye(state_dim + 1, dtype=float) * q
        self.R = r

    def predict_one(self, features: NDArray[np.float64]) -> float:
        """One-step-ahead prediction using current weights."""

        x = np.append(np.asarray(features, dtype=float), 1.0)
        return float(x @ self.theta)

    def update(self, features: NDArray[np.float64], y_true: float) -> None:
        """Kalman update after observing the true value."""

        x = np.append(np.asarray(features, dtype=float), 1.0)
        p_pred = self.P + self.Q
        s_val = float(x @ p_pred @ x + self.R)
        gain = p_pred @ x / s_val
        innovation = float(y_true - (x @ self.theta))
        self.theta = self.theta + (gain * innovation)
        self.P = p_pred - (np.outer(gain, gain) * s_val)
