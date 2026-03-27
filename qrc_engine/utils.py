"""Shared helpers for encoding and evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def as_float_array(values: float | Iterable[float] | np.ndarray) -> FloatArray:
    """Convert inputs to a contiguous 1D float array."""

    return np.asarray(values, dtype=float).reshape(-1)


def angle_encode(value: float, scale: float = 1.0) -> float:
    """Map a scalar to a bounded rotation angle."""

    return float(np.arctan(scale * float(value)))


def nrmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    """Compute normalized root-mean-square error."""

    target_std = float(np.std(y_true))
    if target_std == 0.0:
        raise ValueError("NRMSE is undefined when the target variance is zero.")
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) / target_std)


def pairwise_products(values: FloatArray) -> FloatArray:
    """Return adjacent pairwise products for compact correlation features."""

    if values.size < 2:
        return np.empty(0, dtype=float)
    return values[:-1] * values[1:]
