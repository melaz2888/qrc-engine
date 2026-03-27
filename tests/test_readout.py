"""Readout layer tests."""

from __future__ import annotations

import numpy as np

from qrc_engine.readout import ReadoutLayer


def test_ridge_readout_predicts_expected_shape() -> None:
    """Readout should fit and return a flat prediction vector."""

    rng = np.random.default_rng(2)
    states = rng.normal(size=(40, 5))
    weights = np.asarray([0.4, -0.2, 0.1, 0.3, 0.5])
    targets = states @ weights
    readout = ReadoutLayer(kind="ridge", alpha=1e-6).fit(states, targets)
    predictions = readout.predict(states)
    assert predictions.shape == (40,)
