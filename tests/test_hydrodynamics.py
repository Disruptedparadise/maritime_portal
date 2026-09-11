"""
Unit tests for maritime hydrodynamics, propulsion, multi-fuel thermodynamics,
and cold-ironing shore power calculations.
"""

import unittest
from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE
from hydrodynamics import (
    calculate_displacement,
    calculate_environmental_resistance,
    calculate_engine_sfoc,
    calculate_voyage_metrics
)


class TestMaritimeHydrodynamics(unittest.TestCase):

    def test_displacement_monotonicity(self):
        """Displacement must scale monotonically with cargo load factor."""
        vessel = VESSEL_CLASSES["Container_Panamax"]
        disp_empty = calculate_displacement(vessel, 0.0)
        disp_half = calculate_displacement(vessel, 0.5)
        disp_full = calculate_displacement(vessel, 1.0)

        self.assertEqual(disp_empty, vessel.lightweight_ton)
        self.assertAlmostEqual(disp_full, vessel.lightweight_ton + vessel.deadweight_ton)
        self.assertGreater(disp_full, disp_half)
        self.assertGreater(disp_half, disp_empty)

    def test_cubic_speed_power_relationship(self):
        """Propulsion power must roughly follow cubic speed scaling."""
        v1 = 12.0
        v2 = 18.0
        # Ratio (18/12)^3 = 1.5^3 = 3.375
        expected_ratio = (v2 / v1) ** 3

        m1 = calculate_voyage_metrics("Container_Panamax", "HFO", speed_knots=v1, distance_nm=1000, cargo_load_factor=0.7)
        m2 = calculate_voyage_metrics("Container_Panamax", "HFO", speed_knots=v2, distance_nm=1000, cargo_load_factor=0.7)

        actual_power_ratio = m2["propulsion_power_kw"] / m1["propulsion_power_kw"]
        self.assertAlmostEqual(actual_power_ratio, expected_ratio, delta=0.2)

    def test_alternative_fuel_mass_ratios(self):
        """
        Hydrogen has LHV ~120 MJ/kg vs HFO ~40.5 MJ/kg -> Requires ~3x less mass for identical propulsion.
        Methanol has LHV ~19.9 MJ/kg -> Requires ~2x more mass for identical propulsion.
        """
        res_hfo = calculate_voyage_metrics("Container_Panamax", "HFO", speed_knots=18.0, distance_nm=1000, cargo_load_factor=0.7)
        res_h2 = calculate_voyage_metrics("Container_Panamax", "Hydrogen", speed_knots=18.0, distance_nm=1000, cargo_load_factor=0.7)
        res_meth = calculate_voyage_metrics("Container_Panamax", "Methanol", speed_knots=18.0, distance_nm=1000, cargo_load_factor=0.7)

        # Hydrogen mass should be roughly 40.5 / 120 = 0.3375 of HFO mass
        h2_mass_ratio = res_h2["me_fuel_consumed_kg"] / res_hfo["me_fuel_consumed_kg"]
        self.assertAlmostEqual(h2_mass_ratio, 40.5 / 120.0, delta=0.05)

        # Methanol mass should be roughly 40.5 / 19.9 = 2.035 of HFO mass
        meth_mass_ratio = res_meth["me_fuel_consumed_kg"] / res_hfo["me_fuel_consumed_kg"]
        self.assertAlmostEqual(meth_mass_ratio, 40.5 / 19.9, delta=0.05)

    def test_zero_carbon_exhaust_for_ammonia_and_hydrogen(self):
        """Ammonia and Hydrogen should have ZERO direct Tank-to-Wake CO2 emissions."""
        res_nh3 = calculate_voyage_metrics("Container_Panamax", "Ammonia", speed_knots=18.0, distance_nm=1000, cargo_load_factor=0.7)
        res_h2 = calculate_voyage_metrics("Container_Panamax", "Hydrogen", speed_knots=18.0, distance_nm=1000, cargo_load_factor=0.7)

        self.assertEqual(res_nh3["ttw_co2_emissions_kg"], 0.0)
        self.assertEqual(res_h2["ttw_co2_emissions_kg"], 0.0)
        # But lifecycle emissions exist (from synthesis) and are low
        self.assertGreater(res_nh3["wtw_co2e_emissions_kg"], 0.0)
        self.assertGreater(res_h2["wtw_co2e_emissions_kg"], 0.0)

    def test_shore_power_cold_ironing_benefit(self):
        """Using shore power at berth should reduce port fuel consumption to zero."""
        res_with_shore = calculate_voyage_metrics(
            "Container_ULCV", "HFO", speed_knots=18.0, distance_nm=500,
            cargo_load_factor=0.8, port_dwell_time_hours=36.0, use_shore_power_at_berth=True
        )
        res_without_shore = calculate_voyage_metrics(
            "Container_ULCV", "HFO", speed_knots=18.0, distance_nm=500,
            cargo_load_factor=0.8, port_dwell_time_hours=36.0, use_shore_power_at_berth=False
        )

        self.assertEqual(res_with_shore["port_fuel_consumed_kg"], 0.0)
        self.assertGreater(res_without_shore["port_fuel_consumed_kg"], 0.0)
        self.assertEqual(res_with_shore["port_power_source"], "Shore_Power_AMP")
        self.assertEqual(res_without_shore["port_power_source"], "Auxiliary_Engine")


if __name__ == "__main__":
    unittest.main()
