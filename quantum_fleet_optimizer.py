"""
Multi-Objective Quantum-Inspired Genetic Algorithm (MO-QIGA) for Green Fleet Optimization.
Uses Qubit probability amplitudes, Born rule measurement collapse,
adaptive quantum rotation gates, and quantum catastrophe diversity injection
to generate Pareto-optimal fleet solutions (Cost vs. Lifecycle Emissions vs. Delay).
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from fleet_environment import GreenFleetEnvironment, FleetRouteDemand
from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE


def dominates(sol_a: Tuple[float, float, float], sol_b: Tuple[float, float, float]) -> bool:
    """
    Pareto dominance check: sol_a dominates sol_b if it is no worse in all objectives
    and strictly better in at least one objective (Cost, Emissions, Delay).
    """
    cost_a, emiss_a, delay_a = sol_a
    cost_b, emiss_b, delay_b = sol_b

    no_worse = (cost_a <= cost_b and emiss_a <= emiss_b and delay_a <= delay_b)
    strictly_better = (cost_a < cost_b or emiss_a < emiss_b or delay_a < delay_b)
    return no_worse and strictly_better


def compute_pareto_front(population_fitness: List[Tuple[float, float, float]]) -> List[int]:
    """Returns indices of all non-dominated solutions (Pareto Front 1)."""
    n = len(population_fitness)
    pareto_indices = []

    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and dominates(population_fitness[j], population_fitness[i]):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)

    return pareto_indices


class MultiObjectiveQIGA:
    """
    Quantum-Inspired Genetic Algorithm with Qubit Representation
    and Dynamic Rotation Gate Updates for Maritime Fleet Management.
    """
    def __init__(
        self,
        env: GreenFleetEnvironment,
        pop_size: int = 30,
        generations: int = 35,
        base_rotation_step: float = 0.06,
        catastrophe_threshold: float = 0.08,
        seed: int = 42
    ):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.base_rotation_step = base_rotation_step
        self.catastrophe_threshold = catastrophe_threshold
        self.seed = seed
        np.random.seed(seed)

        self.n_routes = env.n_routes

        # Filter valid vessels and fuels per route to enforce domain validity
        self.route_candidate_vessels: List[List[str]] = []
        self.route_candidate_fuels: List[List[str]] = []
        for route in self.env.routes:
            # Only allow vessels of matching category
            cands = [v_name for v_name, v in VESSEL_CLASSES.items() if v.category == route.cargo_type]
            self.route_candidate_vessels.append(cands)
            self.route_candidate_fuels.append(list(FUEL_DATABASE.keys()))

        # Qubit Chromosome dimensions:
        # Per route: 4 decision qubits
        # q0: vessel choice, q1: speed, q2: fuel choice, q3: shore power
        self.n_genes = self.n_routes * 4

        # Initialize Qubit angles theta in [0, pi/2] (uniform superposition at pi/4)
        self.qubit_angles = np.random.uniform(0.1, np.pi/2 - 0.1, (self.pop_size, self.n_genes))

        self.pareto_archive: List[Dict[str, Any]] = []
        self.history_best_cost: List[float] = []
        self.history_best_emissions: List[float] = []
        self.history_pareto_size: List[int] = []

    def collapse_individual(self, angles: np.ndarray) -> List[Dict[str, Any]]:
        """
        Collapses a continuous qubit angle vector into a concrete fleet deployment
        using Born's probability rule: probability = sin^2(theta).
        """
        deployment = []
        for r_idx, route in enumerate(self.env.routes):
            g_offset = r_idx * 4
            v_angle = angles[g_offset]
            s_angle = angles[g_offset + 1]
            f_angle = angles[g_offset + 2]
            p_angle = angles[g_offset + 3]

            # 1. Vessel Choice
            cand_vessels = self.route_candidate_vessels[r_idx]
            v_prob = np.sin(v_angle) ** 2
            v_idx = int(np.clip(np.floor(v_prob * len(cand_vessels)), 0, len(cand_vessels) - 1))
            vessel_name = cand_vessels[v_idx]
            vessel = VESSEL_CLASSES[vessel_name]

            # 2. Cruising Speed: continuous Born rule mapping
            s_prob = np.sin(s_angle) ** 2
            speed = vessel.min_speed_knots + s_prob * (vessel.max_speed_knots - vessel.min_speed_knots)

            # 3. Fuel Choice: sampled from compatible fuels for this vessel
            compat_fuels = vessel.compatible_fuels
            f_prob = np.sin(f_angle) ** 2
            f_idx = int(np.clip(np.floor(f_prob * len(compat_fuels)), 0, len(compat_fuels) - 1))
            fuel_type = compat_fuels[f_idx]

            # 4. Shore Power (Cold-Ironing): binary quantum measurement
            p_prob = np.sin(p_angle) ** 2
            use_shore_power = bool(p_prob > 0.40) # 60% bias towards green shore power

            deployment.append({
                "vessel_class": vessel_name,
                "speed_knots": speed,
                "fuel_type": fuel_type,
                "use_shore_power": use_shore_power
            })

        return deployment

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes Multi-Objective QIGA optimization.
        Returns:
            pareto_frontier: List of non-dominated Pareto deployment dictionaries.
            stats: Summary optimization metrics and convergence trajectory.
        """
        t0 = time.time()
        print(f"--> Starting MO-QIGA Optimization (Pop: {self.pop_size}, Gen: {self.generations})...")

        for gen in range(self.generations):
            # 1. Quantum State Collapse
            current_deployments = [self.collapse_individual(self.qubit_angles[i]) for i in range(self.pop_size)]

            # 2. Fitness Evaluation on Fleet Environment
            eval_results = [self.env.evaluate_fleet_deployment(dep) for dep in current_deployments]
            fitnesses = [(r[0], r[1], r[2]) for r in eval_results] # (Cost, Emissions, Delay)

            # 3. Non-dominated sorting (Pareto Front 1)
            pareto_indices = compute_pareto_front(fitnesses)

            # Store best objective values for tracking
            costs = [f[0] for f in fitnesses]
            emissions = [f[1] for f in fitnesses]
            self.history_best_cost.append(float(min(costs)))
            self.history_best_emissions.append(float(min(emissions)))
            self.history_pareto_size.append(len(pareto_indices))

            # Update Pareto archive
            for p_idx in pareto_indices:
                dep_copy = current_deployments[p_idx]
                cost, emiss, delay, details = eval_results[p_idx]
                # Check if archive already contains or dominates this solution
                dominated_in_archive = False
                for arch in self.pareto_archive:
                    if dominates((arch["total_cost_usd"], arch["total_emissions_kg"], arch["total_delay_hours"]), (cost, emiss, delay)):
                        dominated_in_archive = True
                        break
                if not dominated_in_archive:
                    # Remove solutions in archive dominated by new solution
                    self.pareto_archive = [
                        arch for arch in self.pareto_archive
                        if not dominates((cost, emiss, delay), (arch["total_cost_usd"], arch["total_emissions_kg"], arch["total_delay_hours"]))
                    ]
                    self.pareto_archive.append({
                        "total_cost_usd": cost,
                        "total_emissions_kg": emiss,
                        "total_delay_hours": delay,
                        "deployment": dep_copy,
                        "details": details,
                        "generation": gen
                    })

            # 4. Adaptive Quantum Rotation Gate Update
            # Annealing rotation step: smaller rotations as generations advance
            step = self.base_rotation_step * (1.0 - 0.65 * (gen / self.generations))

            # Select leaders from Pareto Front
            leader_indices = pareto_indices if len(pareto_indices) > 0 else [int(np.argmin(costs))]

            for i in range(self.pop_size):
                # Pick a random non-dominated leader to guide rotation
                chosen_leader = np.random.choice(leader_indices)
                leader_angles = self.qubit_angles[chosen_leader]

                # Dynamic Quantum Rotation Gate:
                # theta_new = theta + Delta_theta * sign(theta_leader - theta)
                delta = leader_angles - self.qubit_angles[i]
                direction = np.sign(delta)
                rotation = step * direction * np.random.uniform(0.5, 1.2, size=self.n_genes)

                self.qubit_angles[i] += rotation
                # Keep qubit angles strictly bounded in [0.05, pi/2 - 0.05]
                self.qubit_angles[i] = np.clip(self.qubit_angles[i], 0.05, np.pi/2 - 0.05)

            # 5. Quantum Catastrophe & Diversity Maintenance
            # Compute average population angle variance
            angle_variance = float(np.mean(np.var(self.qubit_angles, axis=0)))
            if angle_variance < self.catastrophe_threshold and gen < self.generations - 5:
                # Apply quantum phase inversion (Hadamard-inspired mutation) to 25% of population
                n_mutate = max(2, int(self.pop_size * 0.25))
                mut_indices = np.random.choice(self.pop_size, n_mutate, replace=False)
                for m_idx in mut_indices:
                    # Invert phase: theta -> (pi/2 - theta) + quantum noise
                    self.qubit_angles[m_idx] = (np.pi/2 - self.qubit_angles[m_idx]) + np.random.normal(0, 0.15, self.n_genes)
                    self.qubit_angles[m_idx] = np.clip(self.qubit_angles[m_idx], 0.05, np.pi/2 - 0.05)

        total_runtime = time.time() - t0
        print(f"MO-QIGA Completed in {total_runtime:.2f}s | Pareto Front Size: {len(self.pareto_archive)} solutions")

        stats = {
            "algorithm": "Multi-Objective QIGA",
            "runtime_sec": round(total_runtime, 2),
            "generations": self.generations,
            "population_size": self.pop_size,
            "pareto_solutions_count": len(self.pareto_archive),
            "history_best_cost": self.history_best_cost,
            "history_best_emissions": self.history_best_emissions,
            "history_pareto_size": self.history_pareto_size
        }

        return self.pareto_archive, stats
