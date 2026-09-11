"""
Training & Comprehensive Benchmarking Pipeline.
Trains and benchmarks:
1. Hybrid Quantum-Classical Regressor (HQCKL)
2. Pure Quantum-Inspired SVR (Q-SVR)
3. Random Forest Regressor
4. Gradient Boosting Regressor
5. Classical Support Vector Regressor (RBF-SVR)
6. Linear Regression (Baseline)
Outputs metrics JSON, serialized model weights, and benchmark comparison plots.
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from quantum_preprocessor import MaritimeQuantumPreprocessor
from quantum_predictor import QuantumKernelFuelRegressor, HybridQuantumClassicalFuelRegressor


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes MAPE in percentage, ignoring zero denominators."""
    denom = np.where(y_true == 0, 1.0, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def evaluate_predictions(name: str, y_test: np.ndarray, preds: np.ndarray, train_time: float, latency_ms: float) -> Dict[str, Any]:
    """Computes comprehensive regression metrics."""
    r2 = float(r2_score(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mape = float(mean_absolute_percentage_error(y_test, preds))

    print(f"[{name:<34}] R^2: {r2:0.4f} | MAE: {mae:>10.1f} kg | RMSE: {rmse:>10.1f} kg | MAPE: {mape:5.2f}% | Latency: {latency_ms:0.2f} ms")

    return {
        "model_name": name,
        "r2_score": round(r2, 4),
        "mae_kg": round(mae, 1),
        "rmse_kg": round(rmse, 1),
        "mape_pct": round(mape, 2),
        "train_time_sec": round(train_time, 3),
        "inference_latency_ms": round(latency_ms, 3),
        "_preds_sample": preds[:200].tolist()
    }


def run_benchmark_pipeline(
    dataset_path: str = "data/maritime_voyage_dataset.csv",
    output_dir: str = "models",
    n_train_samples: int = 1200,
    n_test_samples: int = 400
):
    """Executes the complete training and benchmark suite."""
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 88)
    print("        SIH QUANTUM VS CLASSICAL MARITIME FUEL PREDICTION BENCHMARK")
    print("=" * 88)

    # 1. Load dataset
    print(f"Loading maritime dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")

    # 2. Preprocess features
    print("Transforming physical features into quantum angle embeddings...")
    preprocessor = MaritimeQuantumPreprocessor(n_qubits=6)
    X_c, X_q, y = preprocessor.fit_transform(df)

    # Train / Test split
    indices = np.arange(len(df))
    idx_train, idx_test = train_test_split(indices, test_size=0.2, random_state=42)

    idx_train = idx_train[:n_train_samples]
    idx_test = idx_test[:n_test_samples]

    X_train_c, X_test_c = X_c[idx_train], X_c[idx_test]
    X_train_q, X_test_q = X_q[idx_train], X_q[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"Training split: {len(y_train)} voyages | Testing split: {len(y_test)} voyages\n")

    results = []

    # -------------------------------------------------------------
    # Model 1: Hybrid Quantum-Classical Regressor (HQCKL)
    # -------------------------------------------------------------
    print("--> Training Model 1: Hybrid Quantum-Classical Regressor (HQCKL)...")
    hq_model = HybridQuantumClassicalFuelRegressor(n_qubits=6, beta_quantum_weight=0.60, alpha=0.01)
    hq_model.fit(X_train_q, X_train_c, y_train)

    t0 = time.time()
    hq_preds = hq_model.predict(X_test_q, X_test_c)
    hq_latency = ((time.time() - t0) / len(X_test_q)) * 1000.0

    hq_res = evaluate_predictions("Hybrid Quantum-Classical (HQCKL)", y_test, hq_preds, hq_model.train_time_sec_, hq_latency)
    results.append(hq_res)

    # -------------------------------------------------------------
    # Model 2: Pure Quantum-Inspired Kernel Regressor (Q-SVR)
    # -------------------------------------------------------------
    print("--> Training Model 2: Pure Quantum Kernel Regressor (Q-SVR)...")
    q_reg = QuantumKernelFuelRegressor(n_qubits=6, alpha=0.05, n_layers=2, log_target=True)
    q_reg.fit(X_train_q, y_train)

    t0 = time.time()
    q_preds = q_reg.predict(X_test_q)
    q_latency = ((time.time() - t0) / len(X_test_q)) * 1000.0

    q_res = evaluate_predictions("Pure Quantum Kernel (Q-SVR)", y_test, q_preds, q_reg.train_time_sec_, q_latency)
    results.append(q_res)

    # -------------------------------------------------------------
    # Model 3: Classical Random Forest Regressor
    # -------------------------------------------------------------
    print("--> Training Model 3: Classical Random Forest Regressor...")
    t0 = time.time()
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_c, y_train)
    t_rf = time.time() - t0

    t0 = time.time()
    rf_preds = rf_model.predict(X_test_c)
    rf_latency = ((time.time() - t0) / len(X_test_c)) * 1000.0

    rf_res = evaluate_predictions("Random Forest Regressor", y_test, rf_preds, t_rf, rf_latency)
    results.append(rf_res)

    # -------------------------------------------------------------
    # Model 4: Gradient Boosting Regressor
    # -------------------------------------------------------------
    print("--> Training Model 4: Gradient Boosting Regressor...")
    t0 = time.time()
    gbr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.08, max_depth=6, random_state=42)
    gbr_model.fit(X_train_c, y_train)
    t_gbr = time.time() - t0

    t0 = time.time()
    gbr_preds = gbr_model.predict(X_test_c)
    gbr_latency = ((time.time() - t0) / len(X_test_c)) * 1000.0

    gbr_res = evaluate_predictions("Gradient Boosting Regressor", y_test, gbr_preds, t_gbr, gbr_latency)
    results.append(gbr_res)

    # -------------------------------------------------------------
    # Model 5: Classical Support Vector Regressor (RBF-SVR)
    # -------------------------------------------------------------
    print("--> Training Model 5: Classical Support Vector Regressor (RBF-SVR)...")
    t0 = time.time()
    y_train_log = np.log10(y_train)
    svr_model = SVR(kernel="rbf", C=10.0, epsilon=0.05)
    svr_model.fit(X_train_c, y_train_log)
    t_svr = time.time() - t0

    t0 = time.time()
    svr_preds = np.power(10.0, svr_model.predict(X_test_c))
    svr_latency = ((time.time() - t0) / len(X_test_c)) * 1000.0

    svr_res = evaluate_predictions("Classical SVR (RBF Kernel)", y_test, svr_preds, t_svr, svr_latency)
    results.append(svr_res)

    # -------------------------------------------------------------
    # Model 6: Linear Regression Baseline
    # -------------------------------------------------------------
    print("--> Training Model 6: Linear Regression (Baseline)...")
    t0 = time.time()
    lr_model = LinearRegression()
    lr_model.fit(X_train_c, y_train)
    t_lr = time.time() - t0

    t0 = time.time()
    lr_preds = lr_model.predict(X_test_c)
    lr_latency = ((time.time() - t0) / len(X_test_c)) * 1000.0

    lr_res = evaluate_predictions("Linear Regression (Baseline)", y_test, lr_preds, t_lr, lr_latency)
    results.append(lr_res)

    # -------------------------------------------------------------
    # Save Model Weights & Preprocessor
    # -------------------------------------------------------------
    print("\nSerializing model artifacts...")
    joblib.dump(hq_model, os.path.join(output_dir, "hybrid_quantum_predictor.joblib"))
    joblib.dump(q_reg, os.path.join(output_dir, "pure_quantum_predictor.joblib"))
    joblib.dump(rf_model, os.path.join(output_dir, "random_forest_predictor.joblib"))
    joblib.dump(gbr_model, os.path.join(output_dir, "gradient_boosting_predictor.joblib"))
    joblib.dump(preprocessor, os.path.join(output_dir, "preprocessor.joblib"))
    print("Saved models to directory: models/")

    # -------------------------------------------------------------
    # Export Benchmark Results JSON
    # -------------------------------------------------------------
    clean_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    benchmark_json_path = os.path.join(output_dir, "model_benchmark_results.json")
    with open(benchmark_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "SIH Quantum vs Classical Maritime Fuel Prediction Benchmark",
            "training_samples": len(y_train),
            "test_samples": len(y_test),
            "quantum_features": {
                "n_qubits": 6,
                "ansatz": "Entangled Circular CNOT + Non-Linear Phase Modulation",
                "hilbert_dimension": 64
            },
            "models": clean_results
        }, f, indent=2)
    print(f"Benchmark summary saved to: {benchmark_json_path}")

    # -------------------------------------------------------------
    # Plot Benchmark Visualizations
    # -------------------------------------------------------------
    print("Generating publication-quality benchmark charts...")
    plot_path = os.path.join(output_dir, "benchmark_comparison.png")
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    model_labels = [r["model_name"] for r in clean_results]
    r2_scores = [r["r2_score"] for r in clean_results]
    mape_scores = [r["mape_pct"] for r in clean_results]
    train_times = [r["train_time_sec"] for r in clean_results]

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    # 1. R^2 Score
    axes[0, 0].barh(model_labels, r2_scores, color=colors)
    axes[0, 0].set_xlim(0.0, 1.05)
    axes[0, 0].set_xlabel("R² Score (Higher is Better)", fontsize=11)
    axes[0, 0].set_title("Model Prediction Accuracy (R² Score)", fontsize=13, fontweight="bold")
    for i, v in enumerate(r2_scores):
        axes[0, 0].text(v + 0.015, i, f"{v:0.4f}", va="center", fontweight="bold", fontsize=10)

    # 2. MAPE (%)
    axes[0, 1].barh(model_labels, mape_scores, color=colors)
    axes[0, 1].set_xlabel("MAPE % (Lower is Better)", fontsize=11)
    axes[0, 1].set_title("Mean Absolute Percentage Error (%)", fontsize=13, fontweight="bold")
    for i, v in enumerate(mape_scores):
        axes[0, 1].text(v + 1.0, i, f"{v:0.2f}%", va="center", fontweight="bold", fontsize=10)

    # 3. Hybrid Quantum Predicted vs Actual
    hq_sample_preds = np.array(results[0]["_preds_sample"])
    y_test_sample = y_test[:len(hq_sample_preds)]
    axes[1, 0].scatter(y_test_sample / 1000.0, hq_sample_preds / 1000.0, alpha=0.55, color="#1f77b4", label="Hybrid Quantum (HQCKL)")
    min_val = min(y_test_sample.min(), hq_sample_preds.min()) / 1000.0
    max_val = max(y_test_sample.max(), hq_sample_preds.max()) / 1000.0
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect 1:1 Alignment")
    axes[1, 0].set_xlabel("Actual Fuel Consumed (Metric Tons)", fontsize=11)
    axes[1, 0].set_ylabel("Predicted Fuel Consumed (Metric Tons)", fontsize=11)
    axes[1, 0].set_title("Hybrid Quantum: Predicted vs Actual", fontsize=13, fontweight="bold")
    axes[1, 0].legend(fontsize=10)

    # 4. Training Time
    axes[1, 1].barh(model_labels, train_times, color=colors)
    axes[1, 1].set_xlabel("Training Time in Seconds", fontsize=11)
    axes[1, 1].set_title("Computational Training Duration (sec)", fontsize=13, fontweight="bold")
    for i, v in enumerate(train_times):
        axes[1, 1].text(v + 0.1, i, f"{v:0.2f}s", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Charts successfully exported to: {plot_path}")

    print("\n" + "=" * 88)
    print("                   BENCHMARK PIPELINE EXECUTED SUCCESSFULLY")
    print("=" * 88)
    return clean_results


if __name__ == "__main__":
    run_benchmark_pipeline()
