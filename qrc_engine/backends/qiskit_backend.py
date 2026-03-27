"""Gate-based reservoir backend built around Qiskit imports."""

from __future__ import annotations

import logging
from typing import Literal

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
        seed: int = 0,
    ) -> None:
        """Initialize a gate-based backend.

        Args:
            n_qubits: Number of qubits in the reservoir.
            depth: Number of repeated entangling layers.
            shots: Sampling budget reserved for future sampled readout modes.
            feature_mode: Feature extraction mode.
            encoding_scale: Scale factor for angle encoding.
            memory_decay: Exponential decay on carry-over memory.
            seed: Deterministic initialization seed.
        """

        super().__init__(seed=seed)
        self.n_qubits = n_qubits
        self.depth = depth
        self.shots = shots
        self.feature_mode = feature_mode
        self.encoding_scale = encoding_scale
        self.memory_decay = memory_decay
        self._memory = np.zeros(self.n_qubits, dtype=float)
        self._initialized = False
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
        self._initialized = True
        LOGGER.debug("Initialized QiskitBackend with %d qubits.", self.n_qubits)

    def evolve(self, input_val: float) -> FloatArray:
        """Evolve the reservoir for one scalar input.

        Args:
            input_val: Scalar input to encode.

        Returns:
            Feature vector extracted from the resulting quantum state.
        """

        if not self._initialized:
            self.initialize()

        circuit = self._QuantumCircuit(self.n_qubits)
        encoded = angle_encode(input_val, self.encoding_scale)
        for layer in range(self.depth):
            layer_memory = np.roll(self._memory, layer)
            for qubit in range(self.n_qubits):
                theta = (
                    self._input_weights[layer, qubit] * encoded
                    + self._memory_weights[layer, qubit] * layer_memory[qubit]
                    + self._bias[layer, qubit]
                )
                circuit.ry(theta, qubit)
                circuit.rz(self._phase_bias[layer, qubit] + 0.5 * layer_memory[qubit], qubit)
            for qubit in range(self.n_qubits - 1):
                circuit.cx(qubit, qubit + 1)
            if self.n_qubits > 2:
                circuit.cx(self.n_qubits - 1, 0)
        statevector = np.asarray(self._Statevector.from_instruction(circuit).data, dtype=np.complex128)
        probabilities = np.abs(statevector) ** 2
        features = self._extract_features(probabilities)
        self._memory = self.memory_decay * self._memory + (1.0 - self.memory_decay) * features[: self.n_qubits]
        return features

    def reset(self) -> None:
        """Reset the carry-over memory between sequences."""

        self._memory = np.zeros(self.n_qubits, dtype=float)

    @property
    def state_dim(self) -> int:
        """Return the emitted feature dimension."""

        if self.feature_mode == "probabilities":
            return 2**self.n_qubits
        return self.n_qubits + ((self.n_qubits * (self.n_qubits - 1)) // 2)

    def _extract_features(self, probabilities: FloatArray) -> FloatArray:
        """Extract either probabilities or expectation-style observables."""

        if self.feature_mode == "probabilities":
            return np.asarray(probabilities, dtype=float)

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
