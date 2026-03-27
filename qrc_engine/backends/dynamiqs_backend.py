"""Open-system style reservoir backend aligned with dynamiqs workflows."""

from __future__ import annotations

import logging

import numpy as np

from qrc_engine.backends.base import BaseBackend, FloatArray
from qrc_engine.utils import angle_encode

LOGGER = logging.getLogger(__name__)


class DynamiqsBackend(BaseBackend):
    """Small open quantum system reservoir with density-matrix evolution."""

    def __init__(
        self,
        levels: int = 3,
        dt: float = 0.35,
        gamma: float = 0.08,
        encoding_scale: float = 1.0,
        memory_decay: float = 0.55,
        seed: int = 0,
    ) -> None:
        """Initialize the open-system backend.

        Args:
            levels: Hilbert-space dimension.
            dt: Evolution step size.
            gamma: Dissipation strength.
            encoding_scale: Scale factor for input encoding.
            memory_decay: Exponential decay on observable memory.
            seed: Deterministic initialization seed.
        """

        super().__init__(seed=seed)
        self.levels = levels
        self.dt = dt
        self.gamma = gamma
        self.encoding_scale = encoding_scale
        self.memory_decay = memory_decay
        self._initialized = False
        self.initialize()

    def initialize(self) -> None:
        """Initialize Hamiltonian terms and the starting density matrix."""

        try:
            import dynamiqs as dq
        except ImportError as exc:
            raise ImportError("dynamiqs backend requires: pip install qrc-engine[dynamiqs]") from exc

        rng = np.random.default_rng(self.seed)
        self._dq = dq
        energies = np.linspace(0.0, 1.6, self.levels)
        self._h0 = np.diag(energies)
        couplings = np.zeros((self.levels, self.levels), dtype=np.complex128)
        for idx in range(self.levels - 1):
            strength = rng.uniform(0.25, 0.75)
            couplings[idx, idx + 1] = strength
            couplings[idx + 1, idx] = strength
        self._drive = couplings
        self._nonlinear = np.diag(rng.uniform(-0.4, 0.4, size=self.levels))
        self._projector = np.zeros((self.levels, self.levels), dtype=np.complex128)
        self._projector[0, 0] = 1.0
        self._rho = self._ground_state()
        self._memory = np.zeros(self.levels, dtype=float)
        self._initialized = True
        LOGGER.debug("Initialized DynamiqsBackend with %d levels.", self.levels)

    def evolve(self, input_val: float) -> FloatArray:
        """Evolve the density matrix for one input value.

        Args:
            input_val: Scalar input encoded as a drive amplitude.

        Returns:
            Population and nearest-neighbor coherence features.
        """

        if not self._initialized:
            self.initialize()

        encoded = angle_encode(input_val, self.encoding_scale)
        memory_term = np.pad(self._memory[:-1], (1, 0))
        hamiltonian = self._h0 + (encoded * self._drive) + (float(np.mean(memory_term)) * self._nonlinear)
        unitary = self._unitary(hamiltonian)
        rho = unitary @ self._rho @ unitary.conj().T
        rho = (1.0 - self.gamma) * rho + (self.gamma * self._projector * np.trace(rho))
        rho = rho / np.trace(rho)

        populations = np.real(np.diag(rho))
        coherences = [2.0 * float(np.real(rho[idx, idx + 1])) for idx in range(self.levels - 1)]
        features = np.asarray(np.concatenate([populations, coherences]), dtype=float)

        self._rho = rho
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * populations
        return features

    def reset(self) -> None:
        """Reset the density matrix and memory state."""

        self._rho = self._ground_state()
        self._memory = np.zeros(self.levels, dtype=float)

    @property
    def state_dim(self) -> int:
        """Return the observable feature dimension."""

        return (2 * self.levels) - 1

    def _ground_state(self) -> np.ndarray:
        """Construct the ground-state density matrix."""

        rho = np.zeros((self.levels, self.levels), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def _unitary(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Compute a short-step unitary from a Hermitian Hamiltonian."""

        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        phases = np.exp(-1j * eigenvalues * self.dt)
        return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T
