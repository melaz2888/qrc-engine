"""NARMA benchmark generators."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def narma10(n_samples: int = 500, split: float = 0.8, seed: int = 0) -> tuple[FloatArray, ...]:
    """Generate a NARMA-10 forecasting dataset.

    Args:
        n_samples: Total sequence length.
        split: Fraction assigned to the training prefix.
        seed: Deterministic seed for the input drive.

    Returns:
        Train/test splits as ``X_train, y_train, X_test, y_test``.
    """

    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0.0, 0.5, size=n_samples)
    targets = np.zeros(n_samples, dtype=float)
    for step in range(10, n_samples):
        history = np.sum(targets[step - 10 : step])
        targets[step] = (
            0.3 * targets[step - 1]
            + 0.05 * targets[step - 1] * history
            + 1.5 * inputs[step - 10] * inputs[step - 1]
            + 0.1
        )
    cutoff = int(n_samples * split)
    return inputs[:cutoff], targets[:cutoff], inputs[cutoff:], targets[cutoff:]
