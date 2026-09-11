"""
Unit tests for Quantum Kernel Fuel Regressor and Quantum Circuit Embeddings.
"""

import unittest
import numpy as np
from quantum_predictor import QuantumKernelFuelRegressor


class TestQuantumPredictor(unittest.TestCase):

    def setUp(self):
        self.n_qubits = 4
        self.model = QuantumKernelFuelRegressor(n_qubits=self.n_qubits, alpha=0.5, n_layers=1)

    def test_quantum_state_normalization(self):
        """Quantum statevectors must be normalized unit vectors in Hilbert space."""
        X_toy = np.random.uniform(0.1, 3.0, (5, self.n_qubits))
        states = self.model._compute_quantum_states(X_toy)

        for i in range(len(states)):
            norm = np.linalg.norm(states[i])
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"State {i} not normalized")

    def test_quantum_kernel_properties(self):
        """
        Quantum fidelity kernel must satisfy:
        1. Diagonal elements K_ii = 1.0 (self-overlap)
        2. Symmetry: K_ij = K_ji
        3. Values bounded in [0.0, 1.0]
        """
        X_toy = np.random.uniform(0.2, 2.8, (8, self.n_qubits))
        states = self.model._compute_quantum_states(X_toy)
        K = self.model._compute_kernel_matrix(states, states)

        self.assertEqual(K.shape, (8, 8))

        # Check unit diagonal
        for i in range(8):
            self.assertAlmostEqual(K[i, i], 1.0, places=5)

        # Check symmetry
        np.testing.assert_allclose(K, K.T, atol=1e-6)

        # Check bounds
        self.assertTrue(np.all(K >= 0.0))
        self.assertTrue(np.all(K <= 1.0 + 1e-6))

    def test_fit_and_predict_flow(self):
        """Model must successfully fit toy training data and output non-negative predictions."""
        np.random.seed(42)
        X_train = np.random.uniform(0.2, 2.5, (20, self.n_qubits))
        # Fuel target: non-linear function with noise
        y_train = 50000.0 * (X_train[:, 0] ** 2) + 20000.0 * X_train[:, 1] + 1000.0

        self.model.fit(X_train, y_train)

        X_test = np.random.uniform(0.2, 2.5, (5, self.n_qubits))
        preds = self.model.predict(X_test)

        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all(preds > 0.0), "Predicted fuel must be positive")


if __name__ == "__main__":
    unittest.main()
