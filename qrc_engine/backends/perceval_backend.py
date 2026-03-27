"""Photonic-style reservoir backend aligned with Perceval concepts."""

from __future__ import annotations

import logging

import numpy as np

from qrc_engine.backends.base import BaseBackend, FloatArray
from qrc_engine.utils import angle_encode

LOGGER = logging.getLogger(__name__)


class PercevalBackend(BaseBackend):
    """Compact linear-optics reservoir with Perceval as the integration point."""

    def __init__(
        self,
        n_modes: int = 4,
        n_photons: int = 2,
        depth: int = 3,
        encoding_scale: float = 1.0,
        memory_decay: float = 0.5,
        seed: int = 0,
    ) -> None:
        """Initialize the photonic backend.

        Args:
            n_modes: Number of optical modes.
            n_photons: Number of injected photons used to seed the amplitudes.
            depth: Number of interferometer layers.
            encoding_scale: Scale factor for phase encoding.
            memory_decay: Exponential decay on mode-memory carry-over.
            seed: Deterministic initialization seed.
        """

        super().__init__(seed=seed)
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.depth = depth
        self.encoding_scale = encoding_scale
        self.memory_decay = memory_decay
        self._initialized = False
        self.initialize()

    def initialize(self) -> None:
        """Initialize the reference interferometer parameters."""

        try:
            import perceval as pcvl
        except ImportError as exc:
            raise ImportError("Perceval backend requires: pip install qrc-engine[perceval]") from exc

        rng = np.random.default_rng(self.seed)
        self._pcvl = pcvl
        self._phase_bias = rng.uniform(-np.pi, np.pi, size=(self.depth, self.n_modes))
        self._phase_weights = rng.uniform(0.4, 1.2, size=(self.depth, self.n_modes))
        self._memory_weights = rng.uniform(-0.8, 0.8, size=(self.depth, self.n_modes))
        self._bs_theta = rng.uniform(0.2, 1.1, size=(self.depth, self.n_modes - 1))
        self._bs_phi = rng.uniform(-np.pi, np.pi, size=(self.depth, self.n_modes - 1))
        occupations = [1 if idx < self.n_photons else 0 for idx in range(self.n_modes)]
        try:
            self._input_state = pcvl.BasicState(occupations)
        except Exception:
            self._input_state = occupations
        self._memory = np.zeros(self.n_modes, dtype=float)
        self._field = self._initial_field()
        self._initialized = True
        LOGGER.debug("Initialized PercevalBackend with %d modes.", self.n_modes)

    def evolve(self, input_val: float) -> FloatArray:
        """Propagate a photonic field through the interferometer.

        Args:
            input_val: Scalar input to encode as phase shifts.

        Returns:
            Mode occupation probabilities.
        """

        if not self._initialized:
            self.initialize()

        encoded = angle_encode(input_val, self.encoding_scale)
        field = self._field.copy()
        for layer in range(self.depth):
            phases = np.exp(
                1j
                * (
                    self._phase_bias[layer]
                    + (self._phase_weights[layer] * encoded)
                    + (self._memory_weights[layer] * self._memory)
                )
            )
            field = phases * field
            for mode in range(self.n_modes - 1):
                block = self._beam_splitter(self._bs_theta[layer, mode], self._bs_phi[layer, mode] + encoded)
                field[[mode, mode + 1]] = block @ field[[mode, mode + 1]]
        power = np.abs(field) ** 2
        total = float(np.sum(power))
        probabilities = power / total if total else np.zeros(self.n_modes, dtype=float)
        normalized_field = field / np.sqrt(total) if total else self._initial_field()
        coherences = [
            2.0 * float(np.real(np.conjugate(normalized_field[idx]) * normalized_field[idx + 1]))
            for idx in range(self.n_modes - 1)
        ]
        features = np.asarray(np.concatenate([probabilities, coherences]), dtype=float)
        norm = float(np.linalg.norm(field))
        self._field = field / norm if norm else self._initial_field()
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * probabilities
        return features

    def reset(self) -> None:
        """Reset the photonic field and memory state."""

        self._memory = np.zeros(self.n_modes, dtype=float)
        self._field = self._initial_field()

    @property
    def state_dim(self) -> int:
        """Return the mode-probability feature dimension."""

        return (2 * self.n_modes) - 1

    def _initial_field(self) -> np.ndarray:
        """Create the deterministic injected field amplitudes."""

        field = np.zeros(self.n_modes, dtype=np.complex128)
        active_modes = max(1, min(self.n_modes, self.n_photons))
        field[:active_modes] = 1.0 / np.sqrt(active_modes)
        return field

    @staticmethod
    def _beam_splitter(theta: float, phi: float) -> np.ndarray:
        """Construct a 2x2 beam-splitter unitary."""

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        phase = np.exp(1j * phi)
        return np.asarray(
            [
                [cos_theta, -phase * sin_theta],
                [np.conjugate(phase) * sin_theta, cos_theta],
            ],
            dtype=np.complex128,
        )
