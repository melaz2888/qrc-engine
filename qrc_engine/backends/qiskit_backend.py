"""Gate-based reservoir backend built around Qiskit imports."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from qrc_engine.backends.base import BaseBackend, FloatArray
from qrc_engine.utils import angle_encode

LOGGER = logging.getLogger(__name__)


class QiskitBackend(BaseBackend):
    """Random parameterized circuit reservoir with input-dependent rotations."""

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 3,
        shots: int = 1024,
        feature_mode: Literal["expectation", "probabilities"] = "expectation",
        encoding_scale: float = 1.0,
        memory_decay: float = 0.6,
        persistent_state: bool = False,
        use_shots: bool = False,
        gate_noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        """Initialize a gate-based backend.

        Args:
            n_qubits: Number of qubits in the reservoir.
            depth: Number of repeated entangling layers.
            shots: Sampling budget reserved for shot-based readout.
            feature_mode: Feature extraction mode.
            encoding_scale: Scale factor for angle encoding.
            memory_decay: Exponential decay on carry-over memory.
            persistent_state: Whether to evolve from the previous statevector.
            use_shots: Whether to estimate features from sampled outcomes.
            gate_noise_std: Standard deviation of additive Gaussian angle noise.
            seed: Deterministic initialization seed.
        """

        super().__init__(seed=seed)
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.feature_mode = feature_mode
        self.encoding_scale = encoding_scale
        self.memory_decay = memory_decay
        self.persistent_state = persistent_state
        self.use_shots = use_shots
        self.gate_noise_std = gate_noise_std
        self._memory = np.zeros(self.n_qubits, dtype=float)
        self._initialized = False
        self._step_counter = 0
        self._rng = np.random.default_rng(self.seed)
        self._input_projection_cache: dict[int, np.ndarray] = {}
        self.initialize()

    def initialize(self) -> None:
        """Initialize circuit parameters and the simulator handle."""

        try:
            import qiskit_aer
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
        except ImportError as exc:
            raise ImportError("Qiskit backend requires: pip install qrc-engine[qiskit]") from exc

        rng = np.random.default_rng(self.seed)
        self._aer_version = qiskit_aer.__version__
        self._QuantumCircuit = QuantumCircuit
        self._Statevector = Statevector
        self._input_weights = rng.uniform(0.6, 1.4, size=(self.depth, self.n_qubits))
        self._memory_weights = rng.uniform(-0.9, 0.9, size=(self.depth, self.n_qubits))
        self._bias = rng.uniform(-np.pi, np.pi, size=(self.depth, self.n_qubits))
        self._phase_bias = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=(self.depth, self.n_qubits))
        self._basis_index = np.arange(2**self.n_qubits, dtype=np.uint32)
        self._memory = np.zeros(self.n_qubits, dtype=float)
        self._statevector = self._Statevector.from_label("0" * self.n_qubits)
        self._step_counter = 0
        self._rng = np.random.default_rng(self.seed)
        self._initialized = True
        LOGGER.debug("Initialized QiskitBackend with %d qubits.", self.n_qubits)

    def evolve(self, input_val: float | FloatArray) -> FloatArray:
        """Evolve the reservoir for one scalar input or feature vector."""

        if not self._initialized:
            self.initialize()

        input_vector = self._as_input_vector(input_val)
        circuit = self._build_circuit(input_vector)
        if self.persistent_state:
            statevector = self._statevector.evolve(circuit)
        else:
            statevector = self._Statevector.from_instruction(circuit)
        probabilities = np.abs(np.asarray(statevector.data, dtype=np.complex128)) ** 2
        features = self._extract_features(probabilities)
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * features[: self.n_qubits]
        self._statevector = statevector
        self._step_counter += 1
        return features

    def reset(self) -> None:
        """Reset the carry-over memory between sequences."""

        self._memory = np.zeros(self.n_qubits, dtype=float)
        self._statevector = self._Statevector.from_label("0" * self.n_qubits)
        self._step_counter = 0
        self._rng = np.random.default_rng(self.seed)

    @property
    def state_dim(self) -> int:
        """Return the emitted feature dimension."""

        if self.feature_mode == "probabilities":
            return 2**self.n_qubits
        return self.n_qubits + ((self.n_qubits * (self.n_qubits - 1)) // 2)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return backend capability metadata."""

        return {
            "paradigm": "gate-based",
            "state_type": "statevector",
            "has_noise": self.use_shots or (self.gate_noise_std > 0.0),
            "has_persistent_state": self.persistent_state,
        }

    def _build_circuit(self, input_vector: FloatArray) -> Any:
        """Build the per-step update circuit."""

        circuit = self._QuantumCircuit(self.n_qubits)
        encoded = self._encode_inputs(input_vector)
        noise_rng = np.random.default_rng(self.seed + self._step_counter)
        for layer in range(self.depth):
            layer_memory = np.roll(self._memory, layer)
            for qubit in range(self.n_qubits):
                theta = (
                    self._input_weights[layer, qubit] * encoded[layer, qubit]
                    + self._memory_weights[layer, qubit] * layer_memory[qubit]
                    + self._bias[layer, qubit]
                )
                if self.gate_noise_std > 0.0:
                    theta += float(noise_rng.normal(scale=self.gate_noise_std))
                phase_angle = self._phase_bias[layer, qubit] + 0.5 * layer_memory[qubit]
                if self.gate_noise_std > 0.0:
                    phase_angle += float(noise_rng.normal(scale=self.gate_noise_std))
                circuit.ry(theta, qubit)
                circuit.rz(phase_angle, qubit)
            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            if self.n_qubits > 2:
                circuit.cx(self.n_qubits - 1, 0)
        return circuit

    def _encode_inputs(self, input_vector: FloatArray) -> FloatArray:
        """Map scalar or vector inputs to per-layer, per-qubit encoded values."""

        if input_vector.size == 1:
            encoded_scalar = angle_encode(float(input_vector[0]), self.encoding_scale)
            return np.full((self.depth, self.n_qubits), encoded_scalar, dtype=float)

        projections = self._input_projection(input_vector.size)
        encoded_features = np.asarray(
            [angle_encode(float(value), self.encoding_scale) for value in input_vector],
            dtype=float,
        )
        return np.tensordot(projections, encoded_features, axes=([2], [0]))

    def _input_projection(self, input_dim: int) -> np.ndarray:
        """Return deterministic multivariate input weights."""

        if input_dim not in self._input_projection_cache:
            rng = np.random.default_rng(self.seed + (31 * input_dim))
            self._input_projection_cache[input_dim] = rng.uniform(
                0.4,
                1.2,
                size=(self.depth, self.n_qubits, input_dim),
            )
        return self._input_projection_cache[input_dim]

    def _extract_features(self, probabilities: FloatArray) -> FloatArray:
        """Extract either probabilities or expectation-style observables."""

        if self.feature_mode == "probabilities":
            return np.asarray(probabilities, dtype=float)
        if not self.use_shots:
            return self._exact_expectation_features(probabilities)
        return self._sampled_expectation_features(probabilities)

    def _exact_expectation_features(self, probabilities: FloatArray) -> FloatArray:
        """Compute expectation features exactly from probabilities."""

        observables: list[float] = []
        for qubit in range(self.n_qubits):
            bits = (self._basis_index >> qubit) & 1
            observables.append(float(np.dot(probabilities, 1.0 - (2.0 * bits))))
        for left_qubit in range(self.n_qubits - 1):
            left = 1.0 - (2.0 * ((self._basis_index >> left_qubit) & 1))
            for right_qubit in range(left_qubit + 1, self.n_qubits):
                right = 1.0 - (2.0 * ((self._basis_index >> right_qubit) & 1))
                observables.append(float(np.dot(probabilities, left * right)))
        return np.asarray(observables, dtype=float)

    def _sampled_expectation_features(self, probabilities: FloatArray) -> FloatArray:
        """Estimate expectation features from sampled measurement outcomes."""

        sample_rng = np.random.default_rng(self.seed + self._step_counter)
        samples = sample_rng.choice(self._basis_index, size=self.shots, p=probabilities)
        observables: list[float] = []
        for qubit in range(self.n_qubits):
            bits = (samples >> qubit) & 1
            observables.append(float(np.mean(1.0 - (2.0 * bits))))
        for left_qubit in range(self.n_qubits - 1):
            left = 1.0 - (2.0 * ((samples >> left_qubit) & 1))
            for right_qubit in range(left_qubit + 1, self.n_qubits):
                right = 1.0 - (2.0 * ((samples >> right_qubit) & 1))
                observables.append(float(np.mean(left * right)))
        return np.asarray(observables, dtype=float)
