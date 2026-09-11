"""
Execution and Benchmarking Suite for Multi-Objective Green Fleet Optimization.
Executes:
1. Conventional Baseline Fleet Evaluation
2. Multi-Objective Quantum-Inspired Evolutionary Algorithm (MO-QIGA)
3. Classical Genetic Algorithm (CGA)
4. Multi-Objective Particle Swarm Optimization (PSO)
Generates Pareto frontiers, convergence trajectories, and summary reports.
"""

import os
import json
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from fleet_environment import GreenFleetEnvironment
from quantum_fleet_optimizer import MultiObjectiveQIGA, dominates
from classical_optimizers import ClassicalGeneticAlgorithm, ParticleSwarmOptimization


def compute_hypervolume_2d(costs: List[float], emissions: List[float], ref_cost: float, ref_emiss: float) -> float:
    """
    Computes 2D hypervolume dominated by Pareto front with respect to reference nadir point.
    Costs in million USD, emissions in thousand metric tons CO2e.
    Higher hypervolume represents superior Pareto coverage and quality.
    """
    if len(costs) == 0:
        return 0.0

    # Normalize to millions / thousands
    pts = sorted([(c / 1e6, e / 1e3) for c, e in zip(costs, emissions) if c <= ref_cost and e <= ref_emiss], key=lambda x: x[0])
    if len(pts) == 0:
        return 0.0

    ref_c = ref_cost / 1e6
    ref_e = ref_emiss / 1e3

    hv = 0.0
    current_e = ref_e
    for c, e in pts:
        if e < current_e:
            hv += (ref_c - c) * (current_e - e)
            current_e = e

    return round(hv, 2)


def run_fleet_benchmarks(
    pop_size: int = 35,
    generations: int = 30,
    output_dir: str = "data",
    artifact_dir: str = r"C:\Users\Pushkar\.gemini\antigravity\brain\006f11a9-e37d-4553-ad2d-7de6f9e83732"
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 88)
    print("     SIH GREEN FLEET QUANTUM METAHEURISTIC MULTI-OBJECTIVE BENCHMARK")
    print("=" * 88)

    env = GreenFleetEnvironment(carbon_tax_per_ton=80.0)

    # -------------------------------------------------------------
    # 1. Baseline Fleet Evaluation
    # -------------------------------------------------------------
    print("\n[1/4] Evaluating Industry Baseline Fleet (Conventional Operation)...")
    base_dep = env.get_baseline_deployment()
    base_cost, base_emiss, base_delay, base_details = env.evaluate_fleet_deployment(base_dep)
    print(f"  --> Baseline Cost      : ${base_cost:,.2f}")
    print(f"  --> Baseline Emissions : {base_emiss:,.1f} kg CO2e ({base_emiss/1000:,.1f} tons)")
    print(f"  --> Baseline Delay     : {base_delay:.1f} hours")

    # -------------------------------------------------------------
    # 2. Multi-Objective Quantum-Inspired GA (MO-QIGA)
    # -------------------------------------------------------------
    print("\n[2/4] Running Multi-Objective Quantum-Inspired Evolutionary Algorithm (MO-QIGA)...")
    qiga = MultiObjectiveQIGA(env, pop_size=pop_size, generations=generations, seed=42)
    qiga_pareto, qiga_stats = qiga.run()

    # -------------------------------------------------------------
    # 3. Classical Genetic Algorithm (CGA)
    # -------------------------------------------------------------
    print("\n[3/4] Running Classical Genetic Algorithm (CGA)...")
    cga = ClassicalGeneticAlgorithm(env, pop_size=pop_size, generations=generations, seed=42)
    cga_pareto, cga_stats = cga.run()

    # -------------------------------------------------------------
    # 4. Particle Swarm Optimization (PSO)
    # -------------------------------------------------------------
    print("\n[4/4] Running Multi-Objective Particle Swarm Optimization (PSO)...")
    pso = ParticleSwarmOptimization(env, n_particles=pop_size, iterations=generations, seed=42)
    pso_pareto, pso_stats = pso.run()

    # -------------------------------------------------------------
    # Compute Hypervolume & Key Benchmark Metrics
    # -------------------------------------------------------------
    # Reference point: 1.25 * Baseline
    ref_cost = base_cost * 1.25
    ref_emiss = base_emiss * 1.25

    qiga_costs = [p["total_cost_usd"] for p in qiga_pareto]
    qiga_emiss = [p["total_emissions_kg"] for p in qiga_pareto]
    qiga_hv = compute_hypervolume_2d(qiga_costs, qiga_emiss, ref_cost, ref_emiss)

    cga_costs = [p["total_cost_usd"] for p in cga_pareto]
    cga_emiss = [p["total_emissions_kg"] for p in cga_pareto]
    cga_hv = compute_hypervolume_2d(cga_costs, cga_emiss, ref_cost, ref_emiss)

    pso_costs = [p["total_cost_usd"] for p in pso_pareto]
    pso_emiss = [p["total_emissions_kg"] for p in pso_pareto]
    pso_hv = compute_hypervolume_2d(pso_costs, pso_emiss, ref_cost, ref_emiss)

    # Find Representative Solutions on Quantum Pareto Front
    # Min Cost Solution
    q_min_cost_idx = int(np.argmin(qiga_costs))
    q_min_cost_sol = qiga_pareto[q_min_cost_idx]

    # Min Emissions Solution
    q_min_emiss_idx = int(np.argmin(qiga_emiss))
    q_min_emiss_sol = qiga_pareto[q_min_emiss_idx]

    # Balanced Compromise Solution (Closest to normalized Utopia point (0,0))
    c_norm = (np.array(qiga_costs) - min(qiga_costs)) / (max(qiga_costs) - min(qiga_costs) + 1e-6)
    e_norm = (np.array(qiga_emiss) - min(qiga_emiss)) / (max(qiga_emiss) - min(qiga_emiss) + 1e-6)
    utopia_dist = np.sqrt(c_norm**2 + e_norm**2)
    q_balanced_sol = qiga_pareto[int(np.argmin(utopia_dist))]

    # Savings relative to baseline
    cost_savings_pct = (base_cost - q_min_cost_sol["total_cost_usd"]) / base_cost * 100.0
    emiss_savings_pct = (base_emiss - q_min_emiss_sol["total_emissions_kg"]) / base_emiss * 100.0
    balanced_cost_savings = (base_cost - q_balanced_sol["total_cost_usd"]) / base_cost * 100.0
    balanced_emiss_savings = (base_emiss - q_balanced_sol["total_emissions_kg"]) / base_emiss * 100.0

    print("\n" + "=" * 88)
    print("                      OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 88)
    print(f"{'Metric':<32} | {'Baseline':<14} | {'MO-QIGA (Quantum)':<18} | {'Classical GA':<14} | {'PSO'}")
    print("-" * 88)
    print(f"{'Min Cost Found ($)':<32} | ${base_cost/1e6:5.2f}M       | ${min(qiga_costs)/1e6:5.2f}M            | ${min(cga_costs)/1e6:5.2f}M       | ${min(pso_costs)/1e6:5.2f}M")
    print(f"{'Min Emissions Found (tons CO2e)':<32} | {base_emiss/1e3:6.1f}k        | {min(qiga_emiss)/1e3:6.1f}k             | {min(cga_emiss)/1e3:6.1f}k        | {min(pso_emiss)/1e3:6.1f}k")
    print(f"{'Pareto Front Size (solutions)':<32} | {'-':<14} | {len(qiga_pareto):<18} | {len(cga_pareto):<14} | {len(pso_pareto)}")
    print(f"{'Pareto Hypervolume (Higher=Better)':<32} | {'-':<14} | {qiga_hv:<18} | {cga_hv:<14} | {pso_hv}")
    print(f"{'Max Cost Savings vs Baseline':<32} | {'-':<14} | {cost_savings_pct:5.1f}%            | -              | -")
    print(f"{'Max Emissions Cut vs Baseline':<32} | {'-':<14} | {emiss_savings_pct:5.1f}%            | -              | -")
    print("-" * 88)

    # -------------------------------------------------------------
    # Plot Pareto Frontiers & Convergence
    # -------------------------------------------------------------
    print("Generating publication-quality comparison charts...")

    # Chart 1: Pareto Frontiers Comparison
    pareto_plot_path = os.path.join(output_dir, "pareto_frontier_comparison.png")
    plt.figure(figsize=(10, 7))

    plt.scatter(base_cost / 1e6, base_emiss / 1e3, color="black", s=180, marker="X", zorder=5, label=f"Industry Baseline (${base_cost/1e6:0.2f}M, {base_emiss/1e3:0.1f}k t)")
    plt.scatter([c / 1e6 for c in cga_costs], [e / 1e3 for e in cga_emiss], color="#ff7f0e", alpha=0.6, s=70, label=f"Classical GA (HV: {cga_hv})")
    plt.scatter([c / 1e6 for c in pso_costs], [e / 1e3 for e in pso_emiss], color="#2ca02c", alpha=0.6, s=70, label=f"PSO (HV: {pso_hv})")
    plt.scatter([c / 1e6 for c in qiga_costs], [e / 1e3 for e in qiga_emiss], color="#1f77b4", alpha=0.9, s=110, edgecolor="black", linewidth=1.2, label=f"MO-QIGA (Quantum) (HV: {qiga_hv})")

    # Highlight representative points
    plt.scatter(q_balanced_sol["total_cost_usd"] / 1e6, q_balanced_sol["total_emissions_kg"] / 1e3, color="#d62728", s=160, marker="*", zorder=6, label=f"Balanced Green ({balanced_cost_savings:0.1f}% cost, {balanced_emiss_savings:0.1f}% CO2 cut)")

    plt.xlabel("Total Fleet Operational Cost ($ Millions)", fontsize=12)
    plt.ylabel("Lifecycle GHG Emissions (Thousand Tons CO2e)", fontsize=12)
    plt.title("Multi-Objective Fleet Optimization: Pareto Frontier Comparison", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(pareto_plot_path, dpi=200)
    plt.close()

    # Chart 2: Convergence Trajectory Comparison
    conv_plot_path = os.path.join(output_dir, "convergence_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    gens = range(1, generations + 1)
    # Cost Convergence
    ax1.plot(gens, [c / 1e6 for c in qiga_stats["history_best_cost"]], color="#1f77b4", linewidth=2.2, label="MO-QIGA (Quantum)")
    ax1.plot(gens, [c / 1e6 for c in cga_stats["history_best_cost"]], color="#ff7f0e", linestyle="--", linewidth=1.8, label="Classical GA")
    ax1.plot(gens, [c / 1e6 for c in pso_stats["history_best_cost"]], color="#2ca02c", linestyle=":", linewidth=1.8, label="PSO")
    ax1.set_xlabel("Generations / Iterations", fontsize=11)
    ax1.set_ylabel("Best Fleet Cost ($ Millions)", fontsize=11)
    ax1.set_title("Cost Convergence Trajectory", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Emissions Convergence
    ax2.plot(gens, [e / 1e3 for e in qiga_stats["history_best_emissions"]], color="#1f77b4", linewidth=2.2, label="MO-QIGA (Quantum)")
    ax2.plot(gens, [e / 1e3 for e in cga_stats["history_best_emissions"]], color="#ff7f0e", linestyle="--", linewidth=1.8, label="Classical GA")
    ax2.plot(gens, [e / 1e3 for e in pso_stats["history_best_emissions"]], color="#2ca02c", linestyle=":", linewidth=1.8, label="PSO")
    ax2.set_xlabel("Generations / Iterations", fontsize=11)
    ax2.set_ylabel("Best Lifecycle Emissions (k Tons CO2e)", fontsize=11)
    ax2.set_title("Emissions Convergence Trajectory", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(conv_plot_path, dpi=200)
    plt.close()

    # Copy plots to artifact directory for walkthrough report embedding
    if os.path.exists(artifact_dir):
        shutil.copy(pareto_plot_path, os.path.join(artifact_dir, "pareto_frontier_comparison.png"))
        shutil.copy(conv_plot_path, os.path.join(artifact_dir, "convergence_comparison.png"))

    # -------------------------------------------------------------
    # Export Detailed JSON
    # -------------------------------------------------------------
    results_export = {
        "benchmark_name": "SIH Green Fleet Quantum Optimization",
        "baseline_fleet": {
            "total_cost_usd": base_cost,
            "total_emissions_kg": base_emiss,
            "total_delay_hours": base_delay
        },
        "comparison_metrics": {
            "mo_qiga": {
                "algorithm": "Multi-Objective QIGA",
                "hypervolume": qiga_hv,
                "min_cost_usd": min(qiga_costs),
                "min_emissions_kg": min(qiga_emiss),
                "pareto_solutions": len(qiga_pareto),
                "runtime_sec": qiga_stats["runtime_sec"]
            },
            "classical_ga": {
                "algorithm": "Classical GA",
                "hypervolume": cga_hv,
                "min_cost_usd": min(cga_costs),
                "min_emissions_kg": min(cga_emiss),
                "pareto_solutions": len(cga_pareto),
                "runtime_sec": cga_stats["runtime_sec"]
            },
            "pso": {
                "algorithm": "Particle Swarm Optimization",
                "hypervolume": pso_hv,
                "min_cost_usd": min(pso_costs),
                "min_emissions_kg": min(pso_emiss),
                "pareto_solutions": len(pso_pareto),
                "runtime_sec": pso_stats["runtime_sec"]
            }
        },
        "quantum_representative_solutions": {
            "cost_optimal": {
                "cost_usd": q_min_cost_sol["total_cost_usd"],
                "emissions_kg": q_min_cost_sol["total_emissions_kg"],
                "cost_savings_pct": round(cost_savings_pct, 1),
                "emissions_reduction_pct": round((base_emiss - q_min_cost_sol["total_emissions_kg"]) / base_emiss * 100, 1)
            },
            "zero_carbon_optimal": {
                "cost_usd": q_min_emiss_sol["total_cost_usd"],
                "emissions_kg": q_min_emiss_sol["total_emissions_kg"],
                "cost_increase_pct": round((q_min_emiss_sol["total_cost_usd"] - base_cost) / base_cost * 100, 1),
                "emissions_reduction_pct": round(emiss_savings_pct, 1)
            },
            "balanced_compromise": {
                "cost_usd": q_balanced_sol["total_cost_usd"],
                "emissions_kg": q_balanced_sol["total_emissions_kg"],
                "cost_savings_pct": round(balanced_cost_savings, 1),
                "emissions_reduction_pct": round(balanced_emiss_savings, 1)
            }
        },
        "quantum_pareto_frontier": [
            {
                "cost_usd": round(p["total_cost_usd"], 2),
                "emissions_kg": round(p["total_emissions_kg"], 1),
                "delay_hours": round(p["total_delay_hours"], 1),
                "deployment": p["deployment"]
            }
            for p in qiga_pareto
        ]
    }

    results_json_path = os.path.join(output_dir, "fleet_optimization_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_export, f, indent=2)
    print(f"Detailed optimization results saved to: {results_json_path}")
    print("\nBenchmark Finished Successfully!")
    return results_export


if __name__ == "__main__":
    run_fleet_benchmarks(pop_size=35, generations=30)
