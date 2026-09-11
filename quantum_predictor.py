"""
Quantum-Inspired Machine Learning (QIML) Fuel Prediction Engine.
Implements:
1. QuantumKernelFuelRegressor: Pure Quantum Fidelity Kernel in 2^N Hilbert space.
2. HybridQuantumClassicalFuelRegressor: Combines Quantum Entanglement Kernel with Classical RBF regularizer.
Fully serializable via Joblib and Scikit-Learn estimator compatible.
"""

import time
import numpy as np
import pennylane as qml
from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel


class QuantumKernelFuelRegressor(BaseEstimator, RegressorMixin):
    """
    Pure Quantum Kernel Support Vector / Ridge Regressor (Q-SVR).
    Computes quantum state overlap fidelity K_ij = |<psi(x_i) | psi(x_j)>|^2
    using a multi-qubit parameterized quantum circuit with CNOT entanglement and phase shifts.
    """
    def __init__(
        self,
        n_qubits: int = 6,
        alpha: float = 0.05,
        n_layers: int = 2,
        log_target: bool = True
    ):
        self.n_qubits = n_qubits
        self.alpha = alpha
        self.n_layers = n_layers
        self.log_target = log_target

        self._init_quantum_circuit()

        self.train_states_: Optional[np.ndarray] = None
        self.ridge_model_: Optional[Ridge] = None
        self.train_time_sec_: float = 0.0

    def _init_quantum_circuit(self):
        """Initializes PennyLane device and QNode."""
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x):
            # 1. Initialize uniform superposition
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)

            # 2. Entangled Quantum Feature Map
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(x[i], wires=i)
                    qml.RZ(x[i] ** 2, wires=i)

                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

                for i in range(self.n_qubits):
                    qml.PhaseShift(x[i] * x[(i + 1) % self.n_qubits], wires=i)

            return qml.state()

        self.qnode_circuit = circuit

    def __getstate__(self):
        """Custom pickling handler to omit local QNode closure."""
        state = self.__dict__.copy()
        state["dev"] = None
        state["qnode_circuit"] = None
        return state

    def __setstate__(self, state):
        """Restore and rebuild quantum circuit upon unpickling."""
        self.__dict__.update(state)
        self._init_quantum_circuit()

    def _compute_quantum_states(self, X: np.ndarray) -> np.ndarray:
        """Computes 2^N dimensional complex unit state vectors."""
        n_samples = X.shape[0]
        state_dim = 2 ** self.n_qubits
        states = np.zeros((n_samples, state_dim), dtype=np.complex128)

        for i in range(n_samples):
            psi = self.qnode_circuit(X[i])
            states[i] = np.array(psi, dtype=np.complex128)

        return states

    def _compute_fidelity_kernel(self, states_A: np.ndarray, states_B: np.ndarray) -> np.ndarray:
        """Fidelity overlap: K_ij = |<psi_i | psi_j>|^2"""
        gram = np.dot(states_A, states_B.conj().T)
        fidelity = np.abs(gram) ** 2
        return np.clip(fidelity, 0.0, 1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantumKernelFuelRegressor":
        t0 = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        y_train = np.log10(np.clip(y, 1.0, None)) if self.log_target else y

        self.train_states_ = self._compute_quantum_states(X)
        K_train = self._compute_fidelity_kernel(self.train_states_, self.train_states_)

        self.ridge_model_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.ridge_model_.fit(K_train, y_train)

        self.train_time_sec_ = time.time() - t0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.train_states_ is None or self.ridge_model_ is None:
            raise RuntimeError("Model is not fitted.")

        X = np.asarray(X, dtype=np.float64)
        test_states = self._compute_quantum_states(X)
        K_test = self._compute_fidelity_kernel(test_states, self.train_states_)

        pred_log = self.ridge_model_.predict(K_test)
        preds = np.power(10.0, pred_log) if self.log_target else pred_log
        return np.clip(preds, 0.0, None)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y, self.predict(X)))


class HybridQuantumClassicalFuelRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Quantum-Classical Kernel Regressor (HQCKL).
    Combines the high-dimensional quantum Hilbert feature map with
    classical Gaussian RBF kernel: K_hybrid = beta * K_quantum + (1 - beta) * K_classical.
    Provides quantum entanglement advantages while guaranteeing high numerical stability.
    """
    def __init__(
        self,
        n_qubits: int = 6,
        beta_quantum_weight: float = 0.60,
        alpha: float = 0.01,
        gamma_rbf: float = 0.05,
        log_target: bool = True
    ):
        self.n_qubits = n_qubits
        self.beta_quantum_weight = beta_quantum_weight
        self.alpha = alpha
        self.gamma_rbf = gamma_rbf
        self.log_target = log_target

        self.q_reg = QuantumKernelFuelRegressor(n_qubits=n_qubits, alpha=alpha, log_target=log_target)
        self.train_X_c_: Optional[np.ndarray] = None
        self.ridge_model_: Optional[Ridge] = None
        self.train_time_sec_: float = 0.0

    def fit(self, X_quantum: np.ndarray, X_classical: np.ndarray, y: np.ndarray) -> "HybridQuantumClassicalFuelRegressor":
        t0 = time.time()
        y = np.asarray(y, dtype=np.float64)
        y_train = np.log10(np.clip(y, 1.0, None)) if self.log_target else y

        self.train_X_c_ = np.asarray(X_classical, dtype=np.float64)
        self.q_reg.train_states_ = self.q_reg._compute_quantum_states(X_quantum)

        # 1. Compute Quantum Kernel
        K_q = self.q_reg._compute_fidelity_kernel(self.q_reg.train_states_, self.q_reg.train_states_)
        # 2. Compute Classical RBF Kernel
        K_c = rbf_kernel(self.train_X_c_, self.train_X_c_, gamma=self.gamma_rbf)

        # 3. Hybrid blend
        K_hybrid = (self.beta_quantum_weight * K_q) + ((1.0 - self.beta_quantum_weight) * K_c)

        self.ridge_model_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.ridge_model_.fit(K_hybrid, y_train)

        self.train_time_sec_ = time.time() - t0
        return self

    def predict(self, X_quantum: np.ndarray, X_classical: np.ndarray) -> np.ndarray:
        if self.train_X_c_ is None or self.ridge_model_ is None:
            raise RuntimeError("Model is not fitted.")

        X_q = np.asarray(X_quantum, dtype=np.float64)
        X_c = np.asarray(X_classical, dtype=np.float64)

        test_states = self.q_reg._compute_quantum_states(X_q)
        K_q_test = self.q_reg._compute_fidelity_kernel(test_states, self.q_reg.train_states_)
        K_c_test = rbf_kernel(X_c, self.train_X_c_, gamma=self.gamma_rbf)

        K_hybrid_test = (self.beta_quantum_weight * K_q_test) + ((1.0 - self.beta_quantum_weight) * K_c_test)

        pred_log = self.ridge_model_.predict(K_hybrid_test)
        preds = np.power(10.0, pred_log) if self.log_target else pred_log
        return np.clip(preds, 0.0, None)

    def score(self, X_quantum: np.ndarray, X_classical: np.ndarray, y: np.ndarray) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y, self.predict(X_quantum, X_classical)))
