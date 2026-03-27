"""Abstract backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

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
    def evolve(self, input_val: float) -> FloatArray:
        """Encode an input value, evolve the reservoir, and return features.

        Args:
            input_val: Scalar input to encode.

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
