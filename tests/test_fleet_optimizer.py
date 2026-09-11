"""
Unit tests for Green Fleet Environment and Quantum Fleet Optimizer.
"""

import unittest
import numpy as np
from fleet_environment import GreenFleetEnvironment, FLEET_SCENARIO_ROUTES
from maritime_specs import VESSEL_CLASSES
from quantum_fleet_optimizer import MultiObjectiveQIGA, dominates, compute_pareto_front


class TestFleetOptimizer(unittest.TestCase):

    def setUp(self):
        self.env = GreenFleetEnvironment()
        self.qiga = MultiObjectiveQIGA(self.env, pop_size=10, generations=3, seed=42)

    def test_pareto_dominance_logic(self):
        """Test Pareto dominance: lower cost, lower emissions, lower delay dominates."""
        sol_better = (1000.0, 5000.0, 0.0)
        sol_worse = (1200.0, 6000.0, 2.0)
        sol_tradeoff = (800.0, 7000.0, 0.0)

        self.assertTrue(dominates(sol_better, sol_worse))
        self.assertFalse(dominates(sol_worse, sol_better))
        # Trade-off solutions must not dominate each other
        self.assertFalse(dominates(sol_better, sol_tradeoff))
        self.assertFalse(dominates(sol_tradeoff, sol_better))

    def test_compute_pareto_front(self):
        """Verify extraction of Front 1 non-dominated set."""
        fitnesses = [
            (100.0, 200.0, 0.0), # Non-dominated (Lowest cost)
            (200.0, 100.0, 0.0), # Non-dominated (Lowest emissions)
            (300.0, 300.0, 1.0), # Dominated by both
            (150.0, 150.0, 0.0)  # Non-dominated (Balanced)
        ]
        front = compute_pareto_front(fitnesses)
        self.assertIn(0, front)
        self.assertIn(1, front)
        self.assertIn(3, front)
        self.assertNotIn(2, front)

    def test_qubit_collapse_bounds(self):
        """Qubit measurement collapse must generate valid physical speeds and matching categories."""
        dummy_angles = np.random.uniform(0.1, 1.4, self.qiga.n_genes)
        dep = self.qiga.collapse_individual(dummy_angles)

        self.assertEqual(len(dep), self.env.n_routes)

        for r_idx, dec in enumerate(dep):
            route = self.env.routes[r_idx]
            vessel = VESSEL_CLASSES[dec["vessel_class"]]

            # Category matching
            self.assertEqual(vessel.category, route.cargo_type)

            # Speed bounds
            self.assertGreaterEqual(dec["speed_knots"], vessel.min_speed_knots - 1e-4)
            self.assertLessEqual(dec["speed_knots"], vessel.max_speed_knots + 1e-4)

            # Fuel compatibility
            self.assertIn(dec["fuel_type"], vessel.compatible_fuels)


if __name__ == "__main__":
    unittest.main()
