"""Simple time-series generation helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def sine_forecasting(
    n_samples: int = 400,
    split: float = 0.8,
    frequency: float = 0.05,
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[FloatArray, ...]:
    """Generate a one-step-ahead sine forecasting task."""

    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    time = np.arange(n_samples + 1, dtype=float)
    series = np.sin(2.0 * np.pi * frequency * time)
    if noise:
        series = series + rng.normal(scale=noise, size=series.shape)
    inputs, targets = series[:-1], series[1:]
    cutoff = int(n_samples * split)
    return inputs[:cutoff], targets[:cutoff], inputs[cutoff:], targets[cutoff:]


def mackey_glass(
    n_samples: int = 500,
    split: float = 0.8,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    exponent: int = 10,
    delta_t: float = 1.0,
    seed: int = 0,
) -> tuple[FloatArray, ...]:
    """Generate a Mackey-Glass one-step-ahead forecasting task."""

    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0 and 1.")
    warmup = tau + 100
    total_steps = n_samples + warmup + 1
    rng = np.random.default_rng(seed)
    series = np.zeros(total_steps, dtype=float)
    series[: tau + 1] = 1.2 + rng.normal(scale=0.05, size=tau + 1)
    for step in range(tau, total_steps - 1):
        delayed = series[step - tau]
        derivative = beta * delayed / (1.0 + delayed**exponent) - gamma * series[step]
        series[step + 1] = series[step] + (derivative * delta_t)
    usable = series[warmup:]
    inputs, targets = usable[:-1], usable[1:]
    cutoff = int(n_samples * split)
    return inputs[:cutoff], targets[:cutoff], inputs[cutoff:], targets[cutoff:]


def lorenz_system(
    n_samples: int = 2000,
    split: float = 0.8,
    dt: float = 0.01,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate a multivariate Lorenz-system forecasting task."""

    if not 0.0 < split < 1.0:
        raise ValueError("split must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    series = np.zeros((n_samples + 1, 3), dtype=float)
    series[0] = np.asarray([1.0, 1.0, 1.0], dtype=float) + rng.normal(scale=0.01, size=3)
    for step in range(n_samples):
        x_val, y_val, z_val = series[step]
        dx = sigma * (y_val - x_val)
        dy = x_val * (rho - z_val) - y_val
        dz = (x_val * y_val) - (beta * z_val)
        series[step + 1] = series[step] + (dt * np.asarray([dx, dy, dz], dtype=float))
    inputs = series[:-1]
    targets = series[1:, 0]
    cutoff = int(n_samples * split)
    return inputs[:cutoff], targets[:cutoff], inputs[cutoff:], targets[cutoff:]
