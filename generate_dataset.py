"""
Comprehensive Maritime Voyage & Fleet Hydrodynamic Dataset Generator.
Generates multi-vessel, multi-fuel, environmental, and shore-power voyage data
for training quantum-inspired prediction models and benchmarking green fleet optimization.
"""

import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE, ShorePowerSpecs
from hydrodynamics import calculate_voyage_metrics


# Major real-world maritime trade routes for reference
SAMPLE_ROUTES = [
    {"route_name": "Shanghai - Rotterdam (Suez)", "distance_nm": 10525, "typical_days": 24},
    {"route_name": "Singapore - Rotterdam", "distance_nm": 8288, "typical_days": 19},
    {"route_name": "Shenzhen - Los Angeles (Transpacific)", "distance_nm": 6500, "typical_days": 14},
    {"route_name": "Hamburg - New York (Transatlantic)", "distance_nm": 3650, "typical_days": 9},
    {"route_name": "Dubai (Jebel Ali) - Mumbai (Nhava Sheva)", "distance_nm": 1070, "typical_days": 3},
    {"route_name": "Singapore - Port Klang (Feeder Hop)", "distance_nm": 210, "typical_days": 1},
    {"route_name": "Tokyo - Busan (Regional)", "distance_nm": 580, "typical_days": 2},
    {"route_name": "Hedland - Qingdao (Iron Ore Bulk)", "distance_nm": 3450, "typical_days": 11},
    {"route_name": "Ras Tanura - Singapore (Crude Tanker)", "distance_nm": 3720, "typical_days": 11},
    {"route_name": "Santos - Rotterdam (Agri-Bulk)", "distance_nm": 5480, "typical_days": 16},
]


def generate_fleet_voyage_dataset(
    n_samples: int = 5000,
    seed: int = 42,
    output_dir: str = "data"
) -> pd.DataFrame:
    """
    Generates a physically calibrated, diverse maritime dataset with 5,000+ voyages.
    """
    np.random.seed(seed)
    random.seed(seed)

    records: List[Dict[str, Any]] = []
    vessel_class_names = list(VESSEL_CLASSES.keys())

    print(f"Generating {n_samples} maritime voyage observations...")

    for i in range(n_samples):
        # 1. Select vessel class
        vessel_name = np.random.choice(vessel_class_names)
        vessel = VESSEL_CLASSES[vessel_name]

        # 2. Select compatible fuel for this vessel
        fuel_code = np.random.choice(vessel.compatible_fuels)

        # 3. Operational parameters
        # Speed sampled within vessel design window
        speed_knots = np.random.uniform(vessel.min_speed_knots, vessel.max_speed_knots)

        # Distance: 60% sample from realistic trade routes (+ noise), 40% random range (150 to 12000 nm)
        if np.random.rand() < 0.6:
            ref_route = random.choice(SAMPLE_ROUTES)
            route_name = ref_route["route_name"]
            distance_nm = max(100.0, ref_route["distance_nm"] * np.random.uniform(0.92, 1.08))
        else:
            route_name = "Custom_Charter_Route"
            distance_nm = np.random.uniform(150.0, 11000.0)

        cargo_load_factor = np.random.uniform(0.35, 0.98) # Real cargo voyages operate between 35% and 98% full

        # 4. Environmental & Weather conditions
        # Sea state / Beaufort scale distribution
        # Most days are moderate (10-20 kts), with occasional storms (>30 kts)
        wind_speed_knots = float(np.clip(np.random.gamma(shape=3.5, scale=4.0), 2.0, 48.0))
        wind_direction_deg = float(np.random.uniform(0.0, 180.0))
        # Wave height correlates with wind speed
        wave_height_m = float(np.clip(0.08 * (wind_speed_knots ** 1.1) + np.random.normal(0, 0.4), 0.5, 7.5))
        days_since_drydock = int(np.random.randint(15, 720))

        # 5. Port stay & Cold-Ironing / Shore Power
        port_dwell_time_hours = float(np.random.uniform(12.0, 60.0))
        # 50% probability that port provides AMP shore power and ship plugs in
        use_shore_power = bool(np.random.rand() < 0.50) if vessel.supports_shore_power else False

        # 6. Carbon tax policy ($/ton CO2)
        carbon_tax_usd = float(np.random.choice([0.0, 40.0, 80.0, 120.0, 180.0, 250.0]))

        # 7. Compute deterministic hydrodynamic metrics
        metrics = calculate_voyage_metrics(
            vessel_class_name=vessel_name,
            fuel_code=fuel_code,
            speed_knots=speed_knots,
            distance_nm=distance_nm,
            cargo_load_factor=cargo_load_factor,
            wind_speed_knots=wind_speed_knots,
            wind_direction_deg=wind_direction_deg,
            wave_height_m=wave_height_m,
            days_since_drydock=days_since_drydock,
            port_dwell_time_hours=port_dwell_time_hours,
            use_shore_power_at_berth=use_shore_power,
            carbon_tax_usd_per_ton=carbon_tax_usd
        )

        # 8. Add realistic sensor telemetry noise (+/- 2.5% Gaussian jitter) to fuel consumed
        sensor_noise_factor = np.random.normal(1.0, 0.025)
        observed_fuel_consumed_kg = metrics["total_fuel_consumed_kg"] * sensor_noise_factor

        # Add route metadata
        metrics["voyage_id"] = f"VOY-{i+1:05d}"
        metrics["route_name"] = route_name
        metrics["wind_speed_knots"] = round(wind_speed_knots, 1)
        metrics["wind_direction_deg"] = round(wind_direction_deg, 1)
        metrics["wave_height_m"] = round(wave_height_m, 2)
        metrics["days_since_drydock"] = days_since_drydock
        metrics["carbon_tax_usd_per_ton"] = carbon_tax_usd
        metrics["observed_fuel_consumed_kg"] = round(observed_fuel_consumed_kg, 1)

        records.append(metrics)

    df = pd.DataFrame(records)

    # Reorder columns logically
    col_order = [
        "voyage_id", "route_name", "vessel_class", "vessel_category", "fuel_type",
        "speed_knots", "distance_nm", "cargo_load_factor", "displacement_ton",
        "wind_speed_knots", "wind_direction_deg", "wave_height_m", "days_since_drydock",
        "transit_time_hours", "port_dwell_time_hours", "propulsion_power_kw",
        "engine_load_ratio", "added_resistance_pct", "is_overloaded",
        "use_shore_power", "port_power_source", "grid_electricity_kwh",
        "me_fuel_consumed_kg", "aux_fuel_transit_kg", "port_fuel_consumed_kg",
        "total_fuel_consumed_kg", "observed_fuel_consumed_kg",
        "fuel_cost_usd", "total_energy_cost_usd", "carbon_tax_usd_per_ton",
        "carbon_tax_usd", "total_voyage_cost_usd",
        "ttw_co2_emissions_kg", "wtw_co2e_emissions_kg", "cii_rating_g_co2_dwt_nm"
    ]
    df = df[col_order]

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "maritime_voyage_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully saved to: {csv_path} (Shape: {df.shape})")

    # Generate summary metadata
    summary = {
        "dataset_name": "SIH Green Maritime Hydrodynamic & Multi-Fuel Fleet Dataset",
        "total_voyages": len(df),
        "vessel_classes": df["vessel_class"].value_counts().to_dict(),
        "vessel_categories": df["vessel_category"].value_counts().to_dict(),
        "fuel_types": df["fuel_type"].value_counts().to_dict(),
        "shore_power_usage": df["use_shore_power"].value_counts().to_dict(),
        "features": {
            "speed_knots": {"min": float(df["speed_knots"].min()), "max": float(df["speed_knots"].max()), "mean": float(round(df["speed_knots"].mean(), 2))},
            "distance_nm": {"min": float(df["distance_nm"].min()), "max": float(df["distance_nm"].max()), "mean": float(round(df["distance_nm"].mean(), 2))},
            "cargo_load_factor": {"min": float(df["cargo_load_factor"].min()), "max": float(df["cargo_load_factor"].max()), "mean": float(round(df["cargo_load_factor"].mean(), 2))},
            "observed_fuel_consumed_kg": {"min": float(df["observed_fuel_consumed_kg"].min()), "max": float(df["observed_fuel_consumed_kg"].max()), "mean": float(round(df["observed_fuel_consumed_kg"].mean(), 2))},
            "wtw_co2e_emissions_kg": {"min": float(df["wtw_co2e_emissions_kg"].min()), "max": float(df["wtw_co2e_emissions_kg"].max()), "mean": float(round(df["wtw_co2e_emissions_kg"].mean(), 2))},
            "cii_rating_g_co2_dwt_nm": {"min": float(df["cii_rating_g_co2_dwt_nm"].min()), "max": float(df["cii_rating_g_co2_dwt_nm"].max()), "mean": float(round(df["cii_rating_g_co2_dwt_nm"].mean(), 3))}
        },
        "fuel_averages": {
            fuel: {
                "avg_fuel_consumed_kg": float(round(df[df["fuel_type"] == fuel]["observed_fuel_consumed_kg"].mean(), 1)),
                "avg_ttw_co2_kg": float(round(df[df["fuel_type"] == fuel]["ttw_co2_emissions_kg"].mean(), 1)),
                "avg_wtw_co2e_kg": float(round(df[df["fuel_type"] == fuel]["wtw_co2e_emissions_kg"].mean(), 1)),
                "avg_voyage_cost_usd": float(round(df[df["fuel_type"] == fuel]["total_voyage_cost_usd"].mean(), 1))
            }
            for fuel in df["fuel_type"].unique()
        }
    }

    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return df


if __name__ == "__main__":
    df = generate_fleet_voyage_dataset(n_samples=5000, seed=42)
    print("\n--- First 5 records ---")
    print(df[["voyage_id", "vessel_class", "fuel_type", "speed_knots", "distance_nm", "observed_fuel_consumed_kg", "wtw_co2e_emissions_kg", "use_shore_power"]].head())
