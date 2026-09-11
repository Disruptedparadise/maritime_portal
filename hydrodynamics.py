"""
Hydrodynamic and Maritime Propulsion Modeling Engine.
Computes displacement, effective resistance, propulsion power,
SFOC load curves, multi-fuel consumption, and shore-power hoteling emissions.
"""

import math
from typing import Dict, Any, Tuple
from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE, ShorePowerSpecs, VesselClassSpecs, FuelProperties


def calculate_displacement(vessel: VesselClassSpecs, cargo_load_factor: float) -> float:
    """
    Computes total vessel displacement in metric tons.
    Displacement = Lightweight (empty ship) + (Deadweight * cargo load factor).
    """
    cargo_load_factor = max(0.0, min(1.0, cargo_load_factor))
    return vessel.lightweight_ton + (vessel.deadweight_ton * cargo_load_factor)


def calculate_environmental_resistance(
    wind_speed_knots: float,
    wind_direction_deg: float,
    significant_wave_height_m: float,
    days_since_last_drydock: int = 180
) -> float:
    """
    Estimates fractional added resistance (delta R / R_calm) due to:
    - Aerodynamic wind drag based on wind speed & relative angle (0 = headwind, 180 = tailwind)
    - Hydrodynamic wave added resistance (sea state)
    - Hull biofouling growth over voyage days
    """
    # Wind angle factor: headwind (cos=1) increases resistance, tailwind decreases resistance slightly
    rad = math.radians(wind_direction_deg)
    wind_factor = (wind_speed_knots / 30.0) ** 2 * (0.08 * math.cos(rad) + 0.04)
    wind_factor = max(-0.03, wind_factor) # Can give small tailwind assist

    # Wave resistance (scales roughly quadratically with wave height)
    wave_factor = 0.035 * (significant_wave_height_m ** 1.8)

    # Biofouling resistance penalty (~1.5% added drag per 100 days in water without cleaning)
    fouling_factor = 0.015 * (days_since_last_drydock / 100.0)

    total_added_resistance = wind_factor + wave_factor + fouling_factor
    return max(0.0, total_added_resistance)


def calculate_engine_sfoc(base_sfoc: float, engine_load_ratio: float) -> float:
    """
    Computes Specific Fuel Oil Consumption (g/kWh) using standard marine 2-stroke diesel / dual-fuel curve.
    Engine is most thermally efficient around 70-85% MCR.
    At low loads (<40%), thermal efficiency drops (higher SFOC).
    """
    load = max(0.15, min(1.05, engine_load_ratio))
    # Quadratic curve centered at optimal 75% load + low-load penalty
    penalty = 0.50 * ((load - 0.75) ** 2) + (0.03 / (load + 0.05))
    return base_sfoc * (1.0 + penalty)


def calculate_voyage_metrics(
    vessel_class_name: str,
    fuel_code: str,
    speed_knots: float,
    distance_nm: float,
    cargo_load_factor: float,
    wind_speed_knots: float = 12.0,
    wind_direction_deg: float = 30.0,
    wave_height_m: float = 1.5,
    days_since_drydock: int = 180,
    port_dwell_time_hours: float = 24.0,
    use_shore_power_at_berth: bool = True,
    carbon_tax_usd_per_ton: float = 80.0
) -> Dict[str, Any]:
    """
    Performs full hydrodynamic and thermodynamic calculation for a voyage leg and port stay.
    """
    if vessel_class_name not in VESSEL_CLASSES:
        raise ValueError(f"Unknown vessel class: {vessel_class_name}")
    if fuel_code not in FUEL_DATABASE:
        raise ValueError(f"Unknown fuel: {fuel_code}")

    vessel = VESSEL_CLASSES[vessel_class_name]
    fuel = FUEL_DATABASE[fuel_code]
    shore_power = ShorePowerSpecs()

    # 1. Speeds and Voyage Time
    speed = max(vessel.min_speed_knots, min(vessel.max_speed_knots, speed_knots))
    transit_time_hours = distance_nm / speed

    # 2. Total displacement & Calm Water Propulsion Power (Admiralty Formula)
    displacement = calculate_displacement(vessel, cargo_load_factor)
    # Admiralty Law: P_calm = (Displacement^(2/3) * V^3) / C_admiralty
    power_calm_kw = (math.pow(displacement, 2.0 / 3.0) * math.pow(speed, 3.0)) / vessel.admiralty_coeff

    # 3. Added environmental resistance (weather, waves, fouling)
    env_resistance = calculate_environmental_resistance(
        wind_speed_knots, wind_direction_deg, wave_height_m, days_since_drydock
    )
    total_propulsion_power_kw = power_calm_kw * (1.0 + env_resistance)

    # Engine load check against MCR
    engine_load = total_propulsion_power_kw / vessel.mcr_power_kw
    is_overloaded = engine_load > 1.0
    engine_load = min(1.0, engine_load)

    # 4. Main Engine Fuel Consumption
    sfoc_me = calculate_engine_sfoc(vessel.base_sfoc_g_per_kwh, engine_load)
    # Energy equivalent scaling: Baseline LHV is 40.5 MJ/kg (HFO)
    energy_scaling = 40.5 / fuel.lhv_mj_per_kg

    # Main engine fuel mass (kg) = Power (kW) * Time (h) * SFOC (g/kWh) / 1000 * energy_scaling
    me_fuel_consumed_kg = total_propulsion_power_kw * transit_time_hours * (sfoc_me / 1000.0) * energy_scaling

    # 5. Auxiliary Engine Fuel during Transit (hoteling & reefers at sea)
    # Auxiliary engines typically use MGO/VLSFO (SFOC ~ 210 g/kWh) or the dual-fuel system
    aux_power_sea_kw = vessel.aux_power_kw * (0.65 + 0.35 * cargo_load_factor) # Reefers active when loaded
    aux_sfoc_sea = 210.0
    aux_fuel_transit_kg = aux_power_sea_kw * transit_time_hours * (aux_sfoc_sea / 1000.0) * energy_scaling

    transit_fuel_total_kg = me_fuel_consumed_kg + aux_fuel_transit_kg

    # 6. Port Berth Operations & Shore Power (Cold-Ironing)
    port_hours = max(0.0, port_dwell_time_hours)
    aux_power_port_kw = vessel.aux_power_kw * 0.75 # Hoteling, lighting, ventilation, pumps, reefers

    if use_shore_power_at_berth and vessel.supports_shore_power:
        port_fuel_consumed_kg = 0.0
        # Electricity drawn from grid (kWh) including transmission loss
        grid_electricity_kwh = (aux_power_port_kw * port_hours) / (1.0 - shore_power.efficiency_loss)
        port_energy_cost_usd = (grid_electricity_kwh * shore_power.electricity_tariff_per_kwh) + shore_power.connection_fixed_fee_usd
        port_emissions_co2_kg = grid_electricity_kwh * shore_power.grid_co2e_per_kwh
        port_power_source = "Shore_Power_AMP"
    else:
        # Auxiliary engine burned in port
        port_fuel_consumed_kg = aux_power_port_kw * port_hours * (aux_sfoc_sea / 1000.0) * energy_scaling
        grid_electricity_kwh = 0.0
        port_energy_cost_usd = (port_fuel_consumed_kg / 1000.0) * fuel.cost_per_metric_ton
        port_emissions_co2_kg = port_fuel_consumed_kg * fuel.ttw_co2_factor
        port_power_source = "Auxiliary_Engine"

    total_voyage_fuel_kg = transit_fuel_total_kg + port_fuel_consumed_kg

    # 7. Financial Costs
    bunkering_fuel_cost_usd = (total_voyage_fuel_kg / 1000.0) * fuel.cost_per_metric_ton
    total_energy_cost_usd = bunkering_fuel_cost_usd + (port_energy_cost_usd if use_shore_power_at_berth else 0.0)

    # 8. Emissions (Tank-to-Wake exhaust vs. Well-to-Wake lifecycle)
    transit_ttw_co2_kg = transit_fuel_total_kg * fuel.ttw_co2_factor
    total_ttw_co2_kg = transit_ttw_co2_kg + (port_emissions_co2_kg if not use_shore_power_at_berth else 0.0)
    
    transit_wtw_co2e_kg = transit_fuel_total_kg * fuel.wtw_co2e_factor
    total_wtw_co2e_kg = transit_wtw_co2e_kg + port_emissions_co2_kg

    # Carbon tax liability (based on lifecycle or direct emissions)
    carbon_tax_usd = (total_wtw_co2e_kg / 1000.0) * carbon_tax_usd_per_ton
    total_voyage_cost_usd = total_energy_cost_usd + carbon_tax_usd

    # 9. IMO Carbon Intensity Indicator (CII) Metric
    # CII = grams CO2 / (DWT * nautical miles)
    transport_work_ton_nm = vessel.deadweight_ton * distance_nm
    cii_grams_co2_per_dwt_nm = (total_ttw_co2_kg * 1000.0) / max(1.0, transport_work_ton_nm)

    return {
        "vessel_class": vessel_class_name,
        "vessel_category": vessel.category,
        "fuel_type": fuel_code,
        "speed_knots": round(speed, 2),
        "distance_nm": round(distance_nm, 1),
        "cargo_load_factor": round(cargo_load_factor, 3),
        "displacement_ton": round(displacement, 1),
        "transit_time_hours": round(transit_time_hours, 2),
        "port_dwell_time_hours": round(port_hours, 1),
        "propulsion_power_kw": round(total_propulsion_power_kw, 1),
        "engine_load_ratio": round(engine_load, 3),
        "is_overloaded": is_overloaded,
        "added_resistance_pct": round(env_resistance * 100.0, 2),
        "me_fuel_consumed_kg": round(me_fuel_consumed_kg, 1),
        "aux_fuel_transit_kg": round(aux_fuel_transit_kg, 1),
        "port_fuel_consumed_kg": round(port_fuel_consumed_kg, 1),
        "total_fuel_consumed_kg": round(total_voyage_fuel_kg, 1),
        "use_shore_power": use_shore_power_at_berth and vessel.supports_shore_power,
        "port_power_source": port_power_source,
        "grid_electricity_kwh": round(grid_electricity_kwh, 1),
        "fuel_cost_usd": round(bunkering_fuel_cost_usd, 2),
        "total_energy_cost_usd": round(total_energy_cost_usd, 2),
        "carbon_tax_usd": round(carbon_tax_usd, 2),
        "total_voyage_cost_usd": round(total_voyage_cost_usd, 2),
        "ttw_co2_emissions_kg": round(total_ttw_co2_kg, 1),
        "wtw_co2e_emissions_kg": round(total_wtw_co2e_kg, 1),
        "cii_rating_g_co2_dwt_nm": round(cii_grams_co2_per_dwt_nm, 3)
    }
