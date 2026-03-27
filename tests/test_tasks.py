"""Task generator tests."""

from __future__ import annotations

import numpy as np

from qrc_engine.tasks import mackey_glass, narma10, sine_forecasting


def test_narma10_shapes_are_consistent() -> None:
    """NARMA-10 should return aligned train/test splits."""

    X_train, y_train, X_test, y_test = narma10(n_samples=100, split=0.7, seed=2)
    assert X_train.shape == y_train.shape == (70,)
    assert X_test.shape == y_test.shape == (30,)


def test_sine_forecasting_is_deterministic_without_noise() -> None:
    """Sine forecasting should be deterministic for a fixed setup."""

    first = sine_forecasting(n_samples=50, split=0.6, noise=0.0, seed=4)
    second = sine_forecasting(n_samples=50, split=0.6, noise=0.0, seed=9)
    for left, right in zip(first, second):
        assert np.allclose(left, right)


def test_mackey_glass_returns_finite_values() -> None:
    """Mackey-Glass generator should produce finite outputs."""

    arrays = mackey_glass(n_samples=80, split=0.75, seed=5)
    for array in arrays:
        assert np.all(np.isfinite(array))
