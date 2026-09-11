"""
Quantum Maritime Feature Preprocessor.
Encodes physical vessel, voyage, and thermodynamic fuel attributes
into normalized qubit rotation angles [0, pi] and feature matrices.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from maritime_specs import FUEL_DATABASE, VESSEL_CLASSES


class MaritimeQuantumPreprocessor:
    """
    Transforms multi-vessel, multi-fuel maritime voyage records
    into quantum-ready state vectors and classical feature matrices.
    """
    def __init__(self, n_qubits: int = 6):
        self.n_qubits = n_qubits
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=n_qubits, random_state=42)
        self.feature_names: List[str] = []
        self.target_name: str = "observed_fuel_consumed_kg"

    def _extract_raw_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches raw tabular dataframe with thermodynamic and hydrodynamic descriptors."""
        df_feat = pd.DataFrame(index=df.index)

        # 1. Direct physical variables
        df_feat["speed_knots"] = df["speed_knots"]
        df_feat["distance_nm"] = df["distance_nm"]
        df_feat["cargo_load_factor"] = df["cargo_load_factor"]
        df_feat["displacement_ton"] = df["displacement_ton"]
        df_feat["added_resistance_pct"] = df["added_resistance_pct"]

        # 2. Fuel thermodynamic properties (LHV and Density)
        df_feat["fuel_lhv_mj_per_kg"] = df["fuel_type"].map(lambda f: FUEL_DATABASE[f].lhv_mj_per_kg)
        df_feat["fuel_density_kg_m3"] = df["fuel_type"].map(lambda f: FUEL_DATABASE[f].density_kg_per_m3)
        df_feat["fuel_cost_ton"] = df["fuel_type"].map(lambda f: FUEL_DATABASE[f].cost_per_metric_ton)

        # 3. Hydrodynamic non-linear feature interaction (V^3 * displacement^(2/3))
        df_feat["power_proxy"] = (df["displacement_ton"] ** (2.0 / 3.0)) * (df["speed_knots"] ** 3.0) / 1000.0

        # 4. One-hot encoded vessel categories
        vessel_dummies = pd.get_dummies(df["vessel_class"], prefix="vessel", drop_first=False)
        fuel_dummies = pd.get_dummies(df["fuel_type"], prefix="fuel", drop_first=False)

        full_features = pd.concat([df_feat, vessel_dummies, fuel_dummies], axis=1)
        return full_features

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits scalers and transforms input data.
        Returns:
            X_classical: Scaled high-dimensional feature matrix for classical models.
            X_quantum: Scaled angles in [0.1*pi, 0.9*pi] for N-qubit quantum state embedding.
            y: Target fuel consumption (kg).
        """
        raw_X = self._extract_raw_features(df)
        self.feature_names = list(raw_X.columns)

        y = df[self.target_name].values.astype(np.float64)

        # Scale classical features
        X_classical = self.scaler.fit_transform(raw_X.values)

        # Dimensionality reduction to n_qubits via PCA
        X_reduced = self.pca.fit_transform(X_classical)

        min_vals = X_reduced.min(axis=0)
        max_vals = X_reduced.max(axis=0)
        denom = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
        X_quantum = 0.1 * np.pi + 0.8 * np.pi * ((X_reduced - min_vals) / denom)

        self._min_vals = min_vals
        self._max_vals = max_vals
        self._denom = denom

        return X_classical, X_quantum, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transforms unseen test data using fitted scalers."""
        raw_X = self._extract_raw_features(df)
        y = df[self.target_name].values.astype(np.float64)

        X_classical = self.scaler.transform(raw_X.values)
        X_reduced = self.pca.transform(X_classical)
        X_quantum = 0.1 * np.pi + 0.8 * np.pi * np.clip((X_reduced - self._min_vals) / self._denom, 0.0, 1.0)

        return X_classical, X_quantum, y
