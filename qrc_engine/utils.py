"""Shared helpers for encoding, evaluation, and small linear-algebra routines."""

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


def permanent_ryser(matrix: NDArray[np.complex128] | NDArray[np.float64]) -> complex:
    """Compute the matrix permanent with Ryser's formula.

    Args:
        matrix: Square matrix.

    Returns:
        Complex permanent value.
    """

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("permanent_ryser expects a square matrix.")

    dimension = array.shape[0]
    if dimension == 0:
        return 1.0 + 0.0j

    total = 0.0 + 0.0j
    for subset_mask in range(1, 1 << dimension):
        parity = -1.0 if ((dimension - int(subset_mask.bit_count())) % 2) else 1.0
        subset_sum = np.zeros(dimension, dtype=np.complex128)
        for column in range(dimension):
            if (subset_mask >> column) & 1:
                subset_sum += array[:, column]
        total += parity * np.prod(subset_sum)
    return total
