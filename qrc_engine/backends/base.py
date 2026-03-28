"""Abstract backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class BaseBackend(ABC):
    """Abstract interface for quantum reservoir backends."""

    def __init__(self, seed: int = 0) -> None:
        """Store the backend seed.

        Args:
            seed: Random seed used for deterministic initialization.
        """

        self.seed = seed

    @abstractmethod
    def initialize(self) -> None:
        """Build the backend-specific reservoir representation."""

    @abstractmethod
    def evolve(self, input_val: float | FloatArray) -> FloatArray:
        """Encode an input value, evolve the reservoir, and return features.

        Args:
            input_val: Scalar input or feature vector to encode.

        Returns:
            Backend feature vector for the current time step.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the backend state for a fresh sequence."""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Return the feature dimension emitted by evolve()."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Return backend capability metadata."""

        return {
            "paradigm": "unknown",
            "state_type": "unknown",
            "has_noise": False,
            "has_persistent_state": False,
        }

    @staticmethod
    def _as_input_vector(input_val: float | FloatArray) -> FloatArray:
        """Normalize scalar or array input to a contiguous 1D float vector."""

        return np.asarray(input_val, dtype=float).reshape(-1)
