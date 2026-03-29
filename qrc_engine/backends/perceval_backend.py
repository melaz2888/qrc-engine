"""Photonic-style reservoir backend aligned with Perceval concepts."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from qrc_engine.backends.base import BaseBackend, FloatArray
from qrc_engine.utils import angle_encode, permanent_ryser

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
        fock_mode: bool = False,
        feedback: bool = False,
        feedback_strength: float = 1.0,
        phase_noise_std: float = 0.0,
        phase_resolution: float = 0.0,
        threshold_detection: bool = False,
        n_detection_samples: int = 1000,
        seed: int = 0,
    ) -> None:
        """Initialize the photonic backend.

        Args:
            n_modes: Number of optical modes.
            n_photons: Number of injected photons used to seed the amplitudes.
            depth: Number of interferometer layers.
            encoding_scale: Scale factor for phase encoding.
            memory_decay: Exponential decay on mode-memory carry-over.
            fock_mode: Whether to evolve a Fock-space state vector.
            feedback: Whether to reset the Fock state each step and feed occupations back.
            feedback_strength: Scale applied to feedback occupations.
            phase_noise_std: Standard deviation of Gaussian phase noise.
            phase_resolution: Phase quantization grid.
            seed: Deterministic initialization seed.
        """

        super().__init__(seed=seed)
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.depth = depth
        self.encoding_scale = encoding_scale
        self.memory_decay = memory_decay
        self.fock_mode = fock_mode
        self.feedback = feedback
        self.feedback_strength = feedback_strength
        self.phase_noise_std = phase_noise_std
        self.phase_resolution = phase_resolution
        self.threshold_detection = threshold_detection
        self.n_detection_samples = n_detection_samples
        self._initialized = False
        self._input_projection_cache: dict[int, np.ndarray] = {}
        self.initialize()

    def initialize(self) -> None:
        """Initialize the reference interferometer parameters."""

        try:
            import perceval as pcvl
        except ImportError as exc:
            raise ImportError("Perceval backend requires: pip install qrc-engine[perceval]") from exc

        rng = np.random.default_rng(self.seed)
        self._pcvl = pcvl
        self._rng = np.random.default_rng(self.seed)
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
        self._feedback_memory = np.zeros(self.n_modes, dtype=float)
        self._fock_basis = self._enumerate_fock_basis()
        self._basis_index = {state: idx for idx, state in enumerate(self._fock_basis)}
        self._initial_fock_state = self._build_initial_fock_state()
        self._fock_state = self._initial_fock_state.copy()
        self._initialized = True
        LOGGER.debug("Initialized PercevalBackend with %d modes.", self.n_modes)

    def evolve(self, input_val: float | FloatArray) -> FloatArray:
        """Propagate a photonic field or Fock state through the interferometer."""

        if not self._initialized:
            self.initialize()

        input_vector = self._as_input_vector(input_val)
        mode_unitary = self._mode_unitary(input_vector)
        if self.fock_mode:
            return self._evolve_fock(mode_unitary)
        return self._evolve_field(mode_unitary)

    def reset(self) -> None:
        """Reset the photonic state and memory state."""

        self._memory = np.zeros(self.n_modes, dtype=float)
        self._field = self._initial_field()
        self._feedback_memory = np.zeros(self.n_modes, dtype=float)
        self._fock_state = self._initial_fock_state.copy()
        self._rng = np.random.default_rng(self.seed)

    @property
    def state_dim(self) -> int:
        """Return the emitted feature dimension."""

        return (2 * self.n_modes) - 1

    @property
    def metadata(self) -> dict[str, Any]:
        """Return backend capability metadata."""

        return {
            "paradigm": "photonic",
            "state_type": "fock" if self.fock_mode else "field",
            "has_noise": (self.phase_noise_std > 0.0) or (self.phase_resolution > 0.0),
            "has_persistent_state": self.fock_mode and not self.feedback,
        }

    def _mode_unitary(self, input_vector: FloatArray) -> np.ndarray:
        """Construct the interferometer unitary for one step."""

        encoded = self._encode_inputs(input_vector)
        state = np.eye(self.n_modes, dtype=np.complex128)
        feedback_term = self.feedback_strength * self._feedback_memory if (self.fock_mode and self.feedback) else 0.0
        for layer in range(self.depth):
            target_phase = (
                self._phase_bias[layer]
                + (self._phase_weights[layer] * encoded[layer])
                + (self._memory_weights[layer] * self._memory)
                + feedback_term
            )
            phases = np.exp(1j * self._apply_phase_noise(target_phase))
            state = np.diag(phases) @ state
            for mode in range(self.n_modes - 1):
                theta = float(self._bs_theta[layer, mode])
                phi = float(self._apply_phase_noise(self._bs_phi[layer, mode] + encoded[layer]))
                block = self._beam_splitter(theta, phi)
                embedded = np.eye(self.n_modes, dtype=np.complex128)
                embedded[np.ix_([mode, mode + 1], [mode, mode + 1])] = block
                state = embedded @ state
        return state

    def _evolve_field(self, mode_unitary: np.ndarray) -> FloatArray:
        """Propagate the classical field."""

        field = mode_unitary @ self._field
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

    def _evolve_fock(self, mode_unitary: np.ndarray) -> FloatArray:
        """Propagate the Fock-space state, optionally with measurement collapse."""

        if self.feedback:
            current_state = self._initial_fock_state
        else:
            current_state = self._fock_state
        fock_unitary = self._fock_unitary(mode_unitary)
        next_state = fock_unitary @ current_state
        norm = float(np.linalg.norm(next_state))
        if norm:
            next_state = next_state / norm

        if self.threshold_detection:
            features, occupations = self._threshold_detect(next_state)
            self._fock_state = self._initial_fock_state.copy()
        else:
            occupations, correlations = self._occupation_features(next_state)
            features = np.asarray(np.concatenate([occupations, correlations]), dtype=float)
            if self.feedback:
                self._fock_state = self._initial_fock_state.copy()
            else:
                self._fock_state = next_state
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * occupations
        self._feedback_memory = occupations.copy()
        return features

    def _threshold_detect(self, state: np.ndarray) -> tuple[FloatArray, FloatArray]:
        """Simulate threshold detection: sample Fock outcomes, binarize per mode.

        A threshold detector on mode j returns 1 if n_j >= 1, else 0.
        We sample n_detection_samples outcomes from the Born distribution,
        binarize each, and return click probabilities + adjacent correlations.
        """

        born_probs = np.abs(state) ** 2
        born_probs = born_probs / float(np.sum(born_probs))
        detection_rng = np.random.default_rng(
            self.seed + hash(tuple(born_probs[:4].tolist())) % (2**31)
        )
        sampled_indices = detection_rng.choice(
            len(self._fock_basis), size=self.n_detection_samples, p=born_probs
        )
        click_counts = np.zeros(self.n_modes, dtype=float)
        corr_counts = np.zeros(self.n_modes - 1, dtype=float)
        for idx in sampled_indices:
            occ = self._fock_basis[idx]
            clicks = np.asarray([1.0 if n > 0 else 0.0 for n in occ], dtype=float)
            click_counts += clicks
            corr_counts += clicks[:-1] * clicks[1:]
        click_probs = click_counts / self.n_detection_samples
        corr_probs = corr_counts / self.n_detection_samples
        features = np.concatenate([click_probs, corr_probs])
        return features, click_probs

    def _encode_inputs(self, input_vector: FloatArray) -> np.ndarray:
        """Map scalar or vector inputs to per-layer encoded values."""

        if input_vector.size == 1:
            encoded_scalar = angle_encode(float(input_vector[0]), self.encoding_scale)
            return np.full(self.depth, encoded_scalar, dtype=float)
        projections = self._input_projection(input_vector.size)
        encoded_features = np.asarray(
            [angle_encode(float(value), self.encoding_scale) for value in input_vector],
            dtype=float,
        )
        return projections @ encoded_features

    def _input_projection(self, input_dim: int) -> np.ndarray:
        """Return deterministic input projection weights."""

        if input_dim not in self._input_projection_cache:
            rng = np.random.default_rng(self.seed + (17 * input_dim))
            self._input_projection_cache[input_dim] = rng.uniform(0.4, 1.2, size=(self.depth, input_dim))
        return self._input_projection_cache[input_dim]

    def _apply_phase_noise(self, target: float | np.ndarray) -> float | np.ndarray:
        """Apply phase quantization and additive Gaussian noise."""

        phase = np.asarray(target, dtype=float)
        if self.phase_resolution > 0.0:
            phase = np.round(phase / self.phase_resolution) * self.phase_resolution
        if self.phase_noise_std > 0.0:
            phase = phase + self._rng.normal(scale=self.phase_noise_std, size=phase.shape)
        if np.ndim(target) == 0:
            return float(phase)
        return phase

    def _enumerate_fock_basis(self) -> list[tuple[int, ...]]:
        """Enumerate all occupation vectors with a fixed photon count."""

        basis: list[tuple[int, ...]] = []

        def recurse(remaining_modes: int, remaining_photons: int, prefix: list[int]) -> None:
            if remaining_modes == 1:
                basis.append(tuple(prefix + [remaining_photons]))
                return
            for photons in range(remaining_photons + 1):
                recurse(remaining_modes - 1, remaining_photons - photons, prefix + [photons])

        recurse(self.n_modes, self.n_photons, [])
        return basis

    def _build_initial_fock_state(self) -> np.ndarray:
        """Construct the deterministic initial Fock state vector."""

        state = np.zeros(len(self._fock_basis), dtype=np.complex128)
        occupation = tuple(1 if idx < self.n_photons else 0 for idx in range(self.n_modes))
        state[self._basis_index[occupation]] = 1.0 + 0.0j
        return state

    def _fock_unitary(self, mode_unitary: np.ndarray) -> np.ndarray:
        """Lift a mode-space unitary into the symmetric Fock space."""

        dimension = len(self._fock_basis)
        lifted = np.zeros((dimension, dimension), dtype=np.complex128)
        for output_index, output_state in enumerate(self._fock_basis):
            output_norm = float(np.prod([math.factorial(value) for value in output_state]))
            output_rows = [mode for mode, count in enumerate(output_state) for _ in range(count)]
            for input_index, input_state in enumerate(self._fock_basis):
                input_norm = float(np.prod([math.factorial(value) for value in input_state]))
                input_cols = [mode for mode, count in enumerate(input_state) for _ in range(count)]
                if not output_rows and not input_cols:
                    lifted[output_index, input_index] = 1.0 + 0.0j
                    continue
                submatrix = mode_unitary[np.ix_(output_rows, input_cols)]
                lifted[output_index, input_index] = permanent_ryser(submatrix) / np.sqrt(output_norm * input_norm)
        return lifted

    def _occupation_features(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute occupation expectations and nearest-neighbor correlations."""

        probabilities = np.abs(state) ** 2
        occupations = np.zeros(self.n_modes, dtype=float)
        correlations = np.zeros(self.n_modes - 1, dtype=float)
        for basis_index, occupation in enumerate(self._fock_basis):
            weight = float(probabilities[basis_index])
            occupation_array = np.asarray(occupation, dtype=float)
            occupations += weight * occupation_array
            correlations += weight * (occupation_array[:-1] * occupation_array[1:])
        return occupations, correlations

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
