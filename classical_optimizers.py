"""
Classical Metaheuristic Optimization Algorithms for Benchmarking:
1. Classical Multi-Objective Genetic Algorithm (CGA)
2. Multi-Objective Particle Swarm Optimization (PSO)
3. Simulated Annealing (SA)
Directly comparable against Quantum-Inspired Evolutionary Algorithms.
"""

import time
import numpy as np
from typing import List, Dict, Any, Tuple
from fleet_environment import GreenFleetEnvironment
from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE
from quantum_fleet_optimizer import dominates, compute_pareto_front


class ClassicalGeneticAlgorithm:
    """Standard Classical Genetic Algorithm (no quantum principles) for fleet optimization."""
    def __init__(
        self,
        env: GreenFleetEnvironment,
        pop_size: int = 30,
        generations: int = 35,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        seed: int = 42
    ):
        self.env = env
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        np.random.seed(seed)

        self.n_routes = env.n_routes
        self.route_candidate_vessels = []
        for route in self.env.routes:
            cands = [v_name for v_name, v in VESSEL_CLASSES.items() if v.category == route.cargo_type]
            self.route_candidate_vessels.append(cands)

        # Classical Population Initialization
        self.population = [self._random_individual() for _ in range(self.pop_size)]
        self.pareto_archive: List[Dict[str, Any]] = []
        self.history_best_cost: List[float] = []
        self.history_best_emissions: List[float] = []

    def _random_individual(self) -> List[Dict[str, Any]]:
        ind = []
        for r_idx, route in enumerate(self.env.routes):
            cands = self.route_candidate_vessels[r_idx]
            v_name = np.random.choice(cands)
            vessel = VESSEL_CLASSES[v_name]
            speed = np.random.uniform(vessel.min_speed_knots, vessel.max_speed_knots)
            fuel = np.random.choice(vessel.compatible_fuels)
            shore = bool(np.random.rand() < 0.5)
            ind.append({"vessel_class": v_name, "speed_knots": speed, "fuel_type": fuel, "use_shore_power": shore})
        return ind

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.time()
        print(f"--> Starting Classical GA (Pop: {self.pop_size}, Gen: {self.generations})...")

        for gen in range(self.generations):
            eval_results = [self.env.evaluate_fleet_deployment(ind) for ind in self.population]
            fitnesses = [(r[0], r[1], r[2]) for r in eval_results]
            pareto_indices = compute_pareto_front(fitnesses)

            costs = [f[0] for f in fitnesses]
            emiss = [f[1] for f in fitnesses]
            self.history_best_cost.append(float(min(costs)))
            self.history_best_emissions.append(float(min(emiss)))

            # Update archive
            for p_idx in pareto_indices:
                cost, em, delay, details = eval_results[p_idx]
                sol = {"total_cost_usd": cost, "total_emissions_kg": em, "total_delay_hours": delay, "deployment": self.population[p_idx], "generation": gen}
                dominated_in_archive = any(dominates((a["total_cost_usd"], a["total_emissions_kg"], a["total_delay_hours"]), (cost, em, delay)) for a in self.pareto_archive)
                if not dominated_in_archive:
                    self.pareto_archive = [a for a in self.pareto_archive if not dominates((cost, em, delay), (a["total_cost_usd"], a["total_emissions_kg"], a["total_delay_hours"]))]
                    self.pareto_archive.append(sol)

            # Selection (Binary Tournament)
            new_pop = []
            for _ in range(self.pop_size):
                i1, i2 = np.random.randint(0, self.pop_size, 2)
                parent1 = self.population[i1] if dominates(fitnesses[i1], fitnesses[i2]) else self.population[i2]
                i3, i4 = np.random.randint(0, self.pop_size, 2)
                parent2 = self.population[i3] if dominates(fitnesses[i3], fitnesses[i4]) else self.population[i4]

                # Crossover
                if np.random.rand() < self.crossover_rate:
                    child = []
                    for r in range(self.n_routes):
                        p = parent1[r] if np.random.rand() < 0.5 else parent2[r]
                        speed = 0.5 * (parent1[r]["speed_knots"] + parent2[r]["speed_knots"])
                        child.append({"vessel_class": p["vessel_class"], "speed_knots": speed, "fuel_type": p["fuel_type"], "use_shore_power": p["use_shore_power"]})
                else:
                    child = [dict(g) for g in parent1]

                # Mutation
                for r in range(self.n_routes):
                    if np.random.rand() < self.mutation_rate:
                        vessel = VESSEL_CLASSES[child[r]["vessel_class"]]
                        child[r]["speed_knots"] = float(np.clip(child[r]["speed_knots"] + np.random.normal(0, 1.2), vessel.min_speed_knots, vessel.max_speed_knots))
                    if np.random.rand() < self.mutation_rate:
                        child[r]["use_shore_power"] = not child[r]["use_shore_power"]
                    if np.random.rand() < (self.mutation_rate * 0.5):
                        vessel = VESSEL_CLASSES[child[r]["vessel_class"]]
                        child[r]["fuel_type"] = np.random.choice(vessel.compatible_fuels)

                new_pop.append(child)

            self.population = new_pop

        runtime = time.time() - t0
        print(f"Classical GA Completed in {runtime:.2f}s | Pareto Front Size: {len(self.pareto_archive)}")

        stats = {
            "algorithm": "Classical GA",
            "runtime_sec": round(runtime, 2),
            "generations": self.generations,
            "population_size": self.pop_size,
            "pareto_solutions_count": len(self.pareto_archive),
            "history_best_cost": self.history_best_cost,
            "history_best_emissions": self.history_best_emissions
        }
        return self.pareto_archive, stats


class ParticleSwarmOptimization:
    """Multi-Objective Particle Swarm Optimization (PSO) for fleet scheduling."""
    def __init__(
        self,
        env: GreenFleetEnvironment,
        n_particles: int = 30,
        iterations: int = 35,
        w: float = 0.7,
        c1: float = 1.4,
        c2: float = 1.4,
        seed: int = 42
    ):
        self.env = env
        self.n_particles = n_particles
        self.iterations = iterations
        self.w, self.c1, self.c2 = w, c1, c2
        np.random.seed(seed)

        self.n_routes = env.n_routes
        self.dim = self.n_routes * 4

        # Continuous vector representation [0, 1]^D
        self.positions = np.random.uniform(0.0, 1.0, (n_particles, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (n_particles, self.dim))

        self.pbest_positions = self.positions.copy()
        self.pbest_fitnesses = []
        self.pareto_archive: List[Dict[str, Any]] = []
        self.history_best_cost: List[float] = []
        self.history_best_emissions: List[float] = []

    def _decode_position(self, pos: np.ndarray) -> List[Dict[str, Any]]:
        deployment = []
        for r_idx, route in enumerate(self.env.routes):
            off = r_idx * 4
            cands = [v for v, vs in VESSEL_CLASSES.items() if vs.category == route.cargo_type]
            v_idx = int(np.clip(np.floor(pos[off] * len(cands)), 0, len(cands) - 1))
            v_name = cands[v_idx]
            vessel = VESSEL_CLASSES[v_name]

            speed = vessel.min_speed_knots + pos[off + 1] * (vessel.max_speed_knots - vessel.min_speed_knots)
            compat = vessel.compatible_fuels
            f_idx = int(np.clip(np.floor(pos[off + 2] * len(compat)), 0, len(compat) - 1))
            fuel = compat[f_idx]
            shore = bool(pos[off + 3] > 0.45)

            deployment.append({"vessel_class": v_name, "speed_knots": speed, "fuel_type": fuel, "use_shore_power": shore})
        return deployment

    def run(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.time()
        print(f"--> Starting Particle Swarm Optimization (Particles: {self.n_particles}, Iter: {self.iterations})...")

        # Initial evaluation
        for i in range(self.n_particles):
            dep = self._decode_position(self.positions[i])
            c, e, d, _ = self.env.evaluate_fleet_deployment(dep)
            self.pbest_fitnesses.append((c, e, d))

        for it in range(self.iterations):
            current_deps = [self._decode_position(self.positions[i]) for i in range(self.n_particles)]
            evals = [self.env.evaluate_fleet_deployment(d) for d in current_deps]
            fitnesses = [(r[0], r[1], r[2]) for r in evals]

            costs = [f[0] for f in fitnesses]
            emiss = [f[1] for f in fitnesses]
            self.history_best_cost.append(float(min(costs)))
            self.history_best_emissions.append(float(min(emiss)))

            # Update pbest and Pareto archive
            pareto_idx = compute_pareto_front(fitnesses)
            for p in pareto_idx:
                c, e, d, _ = evals[p]
                sol = {"total_cost_usd": c, "total_emissions_kg": e, "total_delay_hours": d, "deployment": current_deps[p], "iteration": it}
                if not any(dominates((a["total_cost_usd"], a["total_emissions_kg"], a["total_delay_hours"]), (c, e, d)) for a in self.pareto_archive):
                    self.pareto_archive = [a for a in self.pareto_archive if not dominates((c, e, d), (a["total_cost_usd"], a["total_emissions_kg"], a["total_delay_hours"]))]
                    self.pareto_archive.append(sol)

            # Update pbest
            for i in range(self.n_particles):
                if dominates(fitnesses[i], self.pbest_fitnesses[i]):
                    self.pbest_positions[i] = self.positions[i].copy()
                    self.pbest_fitnesses[i] = fitnesses[i]

            # Choose gbest from Pareto front
            gbest_idx = np.random.choice(pareto_idx) if len(pareto_idx) > 0 else int(np.argmin(costs))
            gbest_pos = self.positions[gbest_idx]

            # Update velocities and positions
            r1 = np.random.uniform(0, 1, (self.n_particles, self.dim))
            r2 = np.random.uniform(0, 1, (self.n_particles, self.dim))
            self.velocities = (
                self.w * self.velocities
                + self.c1 * r1 * (self.pbest_positions - self.positions)
                + self.c2 * r2 * (gbest_pos - self.positions)
            )
            self.velocities = np.clip(self.velocities, -0.2, 0.2)
            self.positions = np.clip(self.positions + self.velocities, 0.0, 1.0)

        runtime = time.time() - t0
        print(f"PSO Completed in {runtime:.2f}s | Pareto Front Size: {len(self.pareto_archive)}")

        stats = {
            "algorithm": "Particle Swarm Optimization",
            "runtime_sec": round(runtime, 2),
            "iterations": self.iterations,
            "population_size": self.n_particles,
            "pareto_solutions_count": len(self.pareto_archive),
            "history_best_cost": self.history_best_cost,
            "history_best_emissions": self.history_best_emissions
        }
        return self.pareto_archive, stats
