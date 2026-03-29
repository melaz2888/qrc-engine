"""Open-system style reservoir backend aligned with dynamiqs workflows."""

from __future__ import annotations

import logging
from typing import Any

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
        lindblad: bool = False,
        n_subsystems: int = 1,
        projective_measurement: bool = False,
        n_measured_levels: int | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the open-system backend."""

        super().__init__(seed=seed)
        self.levels = levels
        self.dt = dt
        self.gamma = gamma
        self.encoding_scale = encoding_scale
        self.memory_decay = memory_decay
        self.lindblad = lindblad
        self.n_subsystems = n_subsystems
        self.projective_measurement = projective_measurement
        self.n_measured_levels = n_measured_levels
        self._initialized = False
        self._input_projection_cache: dict[int, np.ndarray] = {}
        self.initialize()

    def initialize(self) -> None:
        """Initialize Hamiltonian terms and the starting density matrix."""

        try:
            import dynamiqs as dq
        except ImportError as exc:
            raise ImportError("dynamiqs backend requires: pip install qrc-engine[dynamiqs]") from exc

        rng = np.random.default_rng(self.seed)
        self._dq = dq
        self._dimension = self.levels**self.n_subsystems
        self._local_identity = np.eye(self.levels, dtype=np.complex128)
        self._global_identity = np.eye(self._dimension, dtype=np.complex128)
        self._local_h0 = np.diag(np.linspace(0.0, 1.6, self.levels))
        self._local_drive = np.zeros((self.levels, self.levels), dtype=np.complex128)
        for idx in range(self.levels - 1):
            strength = rng.uniform(0.25, 0.75)
            self._local_drive[idx, idx + 1] = strength
            self._local_drive[idx + 1, idx] = strength
        self._local_nonlinear = np.diag(rng.uniform(-0.4, 0.4, size=self.levels))
        self._local_ground_projector = np.zeros((self.levels, self.levels), dtype=np.complex128)
        self._local_ground_projector[0, 0] = 1.0
        self._subsystem_h0 = [self._embed_local(self._local_h0, site) for site in range(self.n_subsystems)]
        self._subsystem_drive = [self._embed_local(self._local_drive, site) for site in range(self.n_subsystems)]
        self._subsystem_nonlinear = [
            self._embed_local(self._local_nonlinear, site) for site in range(self.n_subsystems)
        ]
        self._local_occupations = [self._local_level_projector(level) for level in range(self.levels)]
        self._global_occupations = np.vstack(
            [
                np.diag(self._embed_local(projector, self.n_subsystems - 1)).real
                for projector in self._local_occupations
            ]
        )
        self._interaction_terms = self._build_interaction_terms(rng)
        self._jump_operators = self._build_jump_operators()
        self._rho = self._ground_state()
        self._memory = np.zeros(self._dimension, dtype=float)
        self._initialized = True
        LOGGER.debug(
            "Initialized DynamiqsBackend with %d levels across %d subsystem(s).",
            self.levels,
            self.n_subsystems,
        )

    def evolve(self, input_val: float | FloatArray) -> FloatArray:
        """Evolve the density matrix for one input value or feature vector."""

        if not self._initialized:
            self.initialize()

        if self.n_subsystems == 1 and not self.lindblad:
            return self._evolve_v1_compatible(input_val)

        input_vector = self._as_input_vector(input_val)
        drive_values = self._encode_inputs(input_vector)
        memory_term = np.pad(self._memory[:-1], (1, 0))
        nonlinear_scale = float(np.mean(memory_term))
        hamiltonian = np.zeros((self._dimension, self._dimension), dtype=np.complex128)
        for site in range(self.n_subsystems):
            hamiltonian = hamiltonian + self._subsystem_h0[site]
            hamiltonian = hamiltonian + (drive_values[site] * self._subsystem_drive[site])
            hamiltonian = hamiltonian + (nonlinear_scale * self._subsystem_nonlinear[site])
        for interaction in self._interaction_terms:
            hamiltonian = hamiltonian + interaction

        if self.lindblad:
            rho = self._rk4_lindblad(self._rho, hamiltonian)
        else:
            unitary = self._unitary(hamiltonian)
            rho = unitary @ self._rho @ unitary.conj().T
            rho = (1.0 - self.gamma) * rho + (self.gamma * self._ground_state() * np.trace(rho))
        rho = self._stabilize_density_matrix(rho)
        if self.projective_measurement:
            rho = self._projective_collapse_rho(rho)

        features = self._extract_features(rho)
        populations = np.real(np.diag(rho))
        self._rho = rho
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * populations
        return features

    def reset(self) -> None:
        """Reset the density matrix and memory state."""

        self._rho = self._ground_state()
        self._memory = np.zeros(self._dimension, dtype=float)

    @property
    def state_dim(self) -> int:
        """Return the observable feature dimension."""

        return (2 * self._dimension) - 1

    @property
    def metadata(self) -> dict[str, Any]:
        """Return backend capability metadata."""

        return {
            "paradigm": "open-system",
            "state_type": "density_matrix",
            "has_noise": self.gamma > 0.0,
            "has_persistent_state": True,
        }

    def _evolve_v1_compatible(self, input_val: float | FloatArray) -> FloatArray:
        """Run the original v0.1 single-system evolution exactly."""

        input_vector = self._as_input_vector(input_val)
        if input_vector.size == 1:
            encoded = angle_encode(float(input_vector[0]), self.encoding_scale)
        else:
            encoded = float(self._encode_inputs(input_vector)[0])
        memory_term = np.pad(self._memory[: self.levels - 1], (1, 0))
        hamiltonian = self._local_h0 + (encoded * self._local_drive) + (float(np.mean(memory_term)) * self._local_nonlinear)
        unitary = self._unitary(hamiltonian)
        rho = unitary @ self._rho @ unitary.conj().T
        rho = (1.0 - self.gamma) * rho + (self.gamma * self._local_ground_projector * np.trace(rho))
        rho = rho / np.trace(rho)

        populations = np.real(np.diag(rho))
        coherences = [2.0 * float(np.real(rho[idx, idx + 1])) for idx in range(self.levels - 1)]
        features = np.asarray(np.concatenate([populations, coherences]), dtype=float)

        self._rho = rho
        self._memory[: self.levels] = self.memory_decay * self._memory[: self.levels] + (1.0 - self.memory_decay) * populations
        return features

    def _encode_inputs(self, input_vector: FloatArray) -> np.ndarray:
        """Map scalar or vector inputs to subsystem drive amplitudes."""

        if input_vector.size == 1:
            encoded_scalar = angle_encode(float(input_vector[0]), self.encoding_scale)
            return np.full(self.n_subsystems, encoded_scalar, dtype=float)
        projection = self._input_projection(input_vector.size)
        encoded_features = np.asarray(
            [angle_encode(float(value), self.encoding_scale) for value in input_vector],
            dtype=float,
        )
        return projection @ encoded_features

    def _input_projection(self, input_dim: int) -> np.ndarray:
        """Return deterministic multivariate drive weights."""

        if input_dim not in self._input_projection_cache:
            rng = np.random.default_rng(self.seed + (23 * input_dim))
            self._input_projection_cache[input_dim] = rng.uniform(0.4, 1.2, size=(self.n_subsystems, input_dim))
        return self._input_projection_cache[input_dim]

    def _build_interaction_terms(self, rng: np.random.Generator) -> list[np.ndarray]:
        """Create nearest-neighbor subsystem couplings."""

        interactions: list[np.ndarray] = []
        if self.n_subsystems <= 1:
            return interactions
        ladder_up = np.zeros((self.levels, self.levels), dtype=np.complex128)
        ladder_down = np.zeros((self.levels, self.levels), dtype=np.complex128)
        for idx in range(self.levels - 1):
            ladder_up[idx, idx + 1] = 1.0
            ladder_down[idx + 1, idx] = 1.0
        for site in range(self.n_subsystems - 1):
            strength = rng.uniform(0.05, 0.25)
            interactions.append(
                strength
                * (
                    self._embed_two_local(ladder_up, site, ladder_down, site + 1)
                    + self._embed_two_local(ladder_down, site, ladder_up, site + 1)
                )
            )
        return interactions

    def _build_jump_operators(self) -> list[np.ndarray]:
        """Construct local nearest-neighbor amplitude-damping jump operators."""

        jump_operators: list[np.ndarray] = []
        for site in range(self.n_subsystems):
            for level in range(self.levels - 1):
                local = np.zeros((self.levels, self.levels), dtype=np.complex128)
                local[level, level + 1] = np.sqrt(self.gamma)
                jump_operators.append(self._embed_local(local, site))
        return jump_operators

    def _extract_features(self, rho: np.ndarray) -> FloatArray:
        """Compute populations and nearest-neighbor coherences."""

        populations = np.real(np.diag(rho))
        coherences = [2.0 * float(np.real(rho[idx, idx + 1])) for idx in range(self._dimension - 1)]
        return np.asarray(np.concatenate([populations, coherences]), dtype=float)

    def _projective_collapse_rho(self, rho: np.ndarray) -> np.ndarray:
        """Perform projective measurement on the density matrix and collapse.

        Full measurement: sample outcome k from Born probabilities rho_kk,
        collapse to |k><k|.
        Partial measurement: measure only a subset of levels.
        """

        collapse_rng = np.random.default_rng(
            self.seed + int(np.abs(np.trace(rho)) * 1e8) % (2**31)
        )
        populations = np.real(np.diag(rho))
        populations = np.clip(populations, 0, None)
        total = float(np.sum(populations))
        if total == 0:
            return self._ground_state()
        populations = populations / total

        n_meas = self.n_measured_levels if self.n_measured_levels is not None else self._dimension
        n_meas = min(n_meas, self._dimension)

        if n_meas == self._dimension:
            outcome = int(collapse_rng.choice(self._dimension, p=populations))
            collapsed = np.zeros((self._dimension, self._dimension), dtype=np.complex128)
            collapsed[outcome, outcome] = 1.0
            return collapsed

        meas_probs = populations[:n_meas].copy()
        total_meas = float(np.sum(meas_probs))
        if total_meas > 0 and collapse_rng.random() < total_meas:
            outcome = int(collapse_rng.choice(n_meas, p=meas_probs / total_meas))
            projector = np.zeros((self._dimension, self._dimension), dtype=np.complex128)
            projector[outcome, outcome] = 1.0
            collapsed = projector @ rho @ projector
            trace = float(np.trace(collapsed))
            return collapsed / trace if trace > 0 else self._ground_state()
        else:
            unmeas_block = rho[n_meas:, n_meas:].copy()
            collapsed = np.zeros((self._dimension, self._dimension), dtype=np.complex128)
            collapsed[n_meas:, n_meas:] = unmeas_block
            trace = float(np.trace(collapsed))
            return collapsed / trace if trace > 0 else self._ground_state()

    def _ground_state(self) -> np.ndarray:
        """Construct the ground-state density matrix."""

        rho = np.zeros((self._dimension, self._dimension), dtype=np.complex128)
        rho[0, 0] = 1.0
        return rho

    def _unitary(self, hamiltonian: np.ndarray) -> np.ndarray:
        """Compute a short-step unitary from a Hermitian Hamiltonian."""

        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        phases = np.exp(-1j * eigenvalues * self.dt)
        return eigenvectors @ np.diag(phases) @ eigenvectors.conj().T

    def _rk4_lindblad(self, rho: np.ndarray, hamiltonian: np.ndarray) -> np.ndarray:
        """Integrate one Lindblad step with RK4."""

        def derivative(current_rho: np.ndarray) -> np.ndarray:
            commutator = -1j * (hamiltonian @ current_rho - current_rho @ hamiltonian)
            dissipator = np.zeros_like(current_rho)
            for jump in self._jump_operators:
                jump_dag_jump = jump.conj().T @ jump
                dissipator += jump @ current_rho @ jump.conj().T
                dissipator -= 0.5 * (jump_dag_jump @ current_rho + current_rho @ jump_dag_jump)
            return commutator + dissipator

        k1 = derivative(rho)
        k2 = derivative(rho + (0.5 * self.dt * k1))
        k3 = derivative(rho + (0.5 * self.dt * k2))
        k4 = derivative(rho + (self.dt * k3))
        return rho + ((self.dt / 6.0) * (k1 + (2.0 * k2) + (2.0 * k3) + k4))

    def _stabilize_density_matrix(self, rho: np.ndarray) -> np.ndarray:
        """Restore Hermiticity, positivity, and unit trace numerically."""

        hermitian = 0.5 * (rho + rho.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        clipped = np.clip(eigenvalues.real, 0.0, None)
        if float(np.sum(clipped)) == 0.0:
            return self._ground_state()
        stabilized = eigenvectors @ np.diag(clipped) @ eigenvectors.conj().T
        return stabilized / np.trace(stabilized)

    def _local_level_projector(self, level: int) -> np.ndarray:
        """Construct a projector onto a local energy level."""

        projector = np.zeros((self.levels, self.levels), dtype=np.complex128)
        projector[level, level] = 1.0
        return projector

    def _embed_local(self, operator: np.ndarray, site: int) -> np.ndarray:
        """Embed a local operator on a subsystem into the full Hilbert space."""

        result = np.asarray([[1.0 + 0.0j]])
        for subsystem in range(self.n_subsystems):
            factor = operator if subsystem == site else self._local_identity
            result = np.kron(result, factor)
        return result

    def _embed_two_local(
        self,
        left_operator: np.ndarray,
        left_site: int,
        right_operator: np.ndarray,
        right_site: int,
    ) -> np.ndarray:
        """Embed a two-site operator product into the full Hilbert space."""

        result = np.asarray([[1.0 + 0.0j]])
        for subsystem in range(self.n_subsystems):
            if subsystem == left_site:
                factor = left_operator
            elif subsystem == right_site:
                factor = right_operator
            else:
                factor = self._local_identity
            result = np.kron(result, factor)
        return result
