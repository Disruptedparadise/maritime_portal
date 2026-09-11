"""
Maritime Engineering Specifications and Data Classes
Defines vessel classes, capacities, propulsion characteristics,
fuel thermodynamic properties, and shore power parameters.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class FuelProperties:
    """Properties of maritime fuels."""
    name: str
    fuel_code: str
    lhv_mj_per_kg: float          # Lower Heating Value (MJ/kg)
    density_kg_per_m3: float      # Density (kg/m3) at 15C
    cost_per_metric_ton: float    # Base market cost (USD / ton)
    ttw_co2_factor: float         # Tank-to-Wake CO2 emission factor (kg CO2 / kg fuel)
    wtw_co2e_factor: float        # Well-to-Wake lifecycle GHG factor (kg CO2e / kg fuel)
    requires_cryo_storage: bool   # Cryogenic containment needed
    boil_off_rate_pct_day: float  # Daily boil-off rate (for cryogenic fuels)
    safety_hazard_rating: str     # Toxicity/flammability indicator


# IMO & Marine Engineering standard fuel database
FUEL_DATABASE: Dict[str, FuelProperties] = {
    "HFO": FuelProperties(
        name="Heavy Fuel Oil (VLSFO)",
        fuel_code="HFO",
        lhv_mj_per_kg=40.5,
        density_kg_per_m3=991.0,
        cost_per_metric_ton=600.0,
        ttw_co2_factor=3.114,
        wtw_co2e_factor=3.650,
        requires_cryo_storage=False,
        boil_off_rate_pct_day=0.0,
        safety_hazard_rating="Standard"
    ),
    "LNG": FuelProperties(
        name="Liquefied Natural Gas",
        fuel_code="LNG",
        lhv_mj_per_kg=49.2,
        density_kg_per_m3=450.0,
        cost_per_metric_ton=800.0,
        ttw_co2_factor=2.750,
        wtw_co2e_factor=3.200,      # Accounts for methane slip (20-yr GWP impact)
        requires_cryo_storage=True,
        boil_off_rate_pct_day=0.15,
        safety_hazard_rating="Flammable Cryogen"
    ),
    "Methanol": FuelProperties(
        name="Green e-Methanol",
        fuel_code="Methanol",
        lhv_mj_per_kg=19.9,
        density_kg_per_m3=792.0,
        cost_per_metric_ton=1050.0,
        ttw_co2_factor=1.375,
        wtw_co2e_factor=0.450,      # Net lifecycle emissions with green carbon capture
        requires_cryo_storage=False,
        boil_off_rate_pct_day=0.0,
        safety_hazard_rating="Toxic/Flammable Liquid"
    ),
    "Ammonia": FuelProperties(
        name="Green Ammonia (NH3)",
        fuel_code="Ammonia",
        lhv_mj_per_kg=18.6,
        density_kg_per_m3=682.0,
        cost_per_metric_ton=950.0,
        ttw_co2_factor=0.000,       # Zero carbon in molecule
        wtw_co2e_factor=0.120,      # Lifecycle emissions from renewable synthesis
        requires_cryo_storage=True, # Refrigerated or pressurized
        boil_off_rate_pct_day=0.05,
        safety_hazard_rating="Highly Toxic"
    ),
    "Hydrogen": FuelProperties(
        name="Green Liquid Hydrogen (LH2)",
        fuel_code="Hydrogen",
        lhv_mj_per_kg=120.0,
        density_kg_per_m3=71.0,     # Very low volumetric density
        cost_per_metric_ton=4200.0,
        ttw_co2_factor=0.000,       # Zero carbon
        wtw_co2e_factor=0.050,      # Pure green electrolysis
        requires_cryo_storage=True, # Deep cryogenic (-253 C)
        boil_off_rate_pct_day=0.40,
        safety_hazard_rating="Highly Flammable/Explosive"
    )
}


@dataclass(frozen=True)
class VesselClassSpecs:
    """Physical and operational specifications for a maritime vessel class."""
    category: str                 # "Container", "Bulk Carrier", "Tanker"
    class_name: str               # e.g., "Container_Feeder", "Container_Panamax", etc.
    capacity_teu: Optional[int]   # TEU capacity (for container ships)
    lightweight_ton: float        # Vessel empty weight (LWT in metric tons)
    deadweight_ton: float         # Maximum cargo/fuel/ballast capacity (DWT in tons)
    length_overall_m: float       # LOA (meters)
    beam_m: float                 # Width (meters)
    design_draft_m: float         # Maximum draft (meters)
    design_speed_knots: float     # Design operational speed (knots)
    min_speed_knots: float        # Minimum safe maneuvering speed
    max_speed_knots: float        # Maximum sprint speed
    mcr_power_kw: float           # Maximum Continuous Rating of Main Engine (kW)
    aux_power_kw: float           # Auxiliary Engine Power for hoteling/reefers (kW)
    admiralty_coeff: float        # Hydrodynamic hull efficiency coefficient
    base_sfoc_g_per_kwh: float    # Base Specific Fuel Oil Consumption (g/kWh at 75% MCR)
    compatible_fuels: List[str]   # Allowed fuel types (retrofit / dual-fuel capable)
    supports_shore_power: bool    # Equipped with AMP (Alternative Maritime Power) plug


VESSEL_CLASSES: Dict[str, VesselClassSpecs] = {
    # --- Container Ships ---
    "Container_Feeder": VesselClassSpecs(
        category="Container",
        class_name="Container_Feeder",
        capacity_teu=2500,
        lightweight_ton=11500.0,
        deadweight_ton=34000.0,
        length_overall_m=195.0,
        beam_m=30.2,
        design_draft_m=11.0,
        design_speed_knots=18.5,
        min_speed_knots=10.0,
        max_speed_knots=21.0,
        mcr_power_kw=18000.0,
        aux_power_kw=1800.0,       # Includes ~300 reefer plugs
        admiralty_coeff=540.0,
        base_sfoc_g_per_kwh=168.0,
        compatible_fuels=["HFO", "LNG", "Methanol"],
        supports_shore_power=True
    ),
    "Container_Panamax": VesselClassSpecs(
        category="Container",
        class_name="Container_Panamax",
        capacity_teu=5500,
        lightweight_ton=22000.0,
        deadweight_ton=68000.0,
        length_overall_m=294.0,
        beam_m=32.2,
        design_draft_m=13.5,
        design_speed_knots=21.0,
        min_speed_knots=11.0,
        max_speed_knots=24.0,
        mcr_power_kw=42000.0,
        aux_power_kw=3200.0,       # Includes ~600 reefer plugs
        admiralty_coeff=580.0,
        base_sfoc_g_per_kwh=162.0,
        compatible_fuels=["HFO", "LNG", "Methanol", "Ammonia"],
        supports_shore_power=True
    ),
    "Container_ULCV": VesselClassSpecs(
        category="Container",
        class_name="Container_ULCV",
        capacity_teu=18000,
        lightweight_ton=45000.0,
        deadweight_ton=195000.0,
        length_overall_m=399.0,
        beam_m=58.6,
        design_draft_m=16.0,
        design_speed_knots=22.0,
        min_speed_knots=12.0,
        max_speed_knots=25.0,
        mcr_power_kw=70000.0,
        aux_power_kw=5500.0,       # Large reefer capacity (1500+ plugs)
        admiralty_coeff=620.0,
        base_sfoc_g_per_kwh=158.0,
        compatible_fuels=["HFO", "LNG", "Methanol", "Ammonia", "Hydrogen"],
        supports_shore_power=True
    ),

    # --- Bulk Carriers ---
    "Bulk_Handysize": VesselClassSpecs(
        category="Bulk Carrier",
        class_name="Bulk_Handysize",
        capacity_teu=None,
        lightweight_ton=8500.0,
        deadweight_ton=38000.0,
        length_overall_m=180.0,
        beam_m=28.4,
        design_draft_m=10.2,
        design_speed_knots=14.0,
        min_speed_knots=9.0,
        max_speed_knots=16.0,
        mcr_power_kw=7200.0,
        aux_power_kw=750.0,
        admiralty_coeff=490.0,
        base_sfoc_g_per_kwh=172.0,
        compatible_fuels=["HFO", "LNG", "Methanol"],
        supports_shore_power=True
    ),
    "Bulk_Capesize": VesselClassSpecs(
        category="Bulk Carrier",
        class_name="Bulk_Capesize",
        capacity_teu=None,
        lightweight_ton=24000.0,
        deadweight_ton=180000.0,
        length_overall_m=292.0,
        beam_m=45.0,
        design_draft_m=18.2,
        design_speed_knots=14.5,
        min_speed_knots=9.5,
        max_speed_knots=16.5,
        mcr_power_kw=16500.0,
        aux_power_kw=1100.0,
        admiralty_coeff=530.0,
        base_sfoc_g_per_kwh=164.0,
        compatible_fuels=["HFO", "LNG", "Methanol", "Ammonia"],
        supports_shore_power=True
    ),

    # --- Tankers ---
    "Tanker_Aframax": VesselClassSpecs(
        category="Tanker",
        class_name="Tanker_Aframax",
        capacity_teu=None,
        lightweight_ton=19000.0,
        deadweight_ton=115000.0,
        length_overall_m=245.0,
        beam_m=42.0,
        design_draft_m=15.0,
        design_speed_knots=14.5,
        min_speed_knots=9.0,
        max_speed_knots=16.0,
        mcr_power_kw=13500.0,
        aux_power_kw=1200.0,       # Includes inert gas system & cargo pumps
        admiralty_coeff=510.0,
        base_sfoc_g_per_kwh=166.0,
        compatible_fuels=["HFO", "LNG", "Methanol"],
        supports_shore_power=True
    )
}


@dataclass(frozen=True)
class ShorePowerSpecs:
    """Port Shore Power (Cold Ironing / AMP) parameters."""
    grid_co2e_per_kwh: float = 0.380       # kg CO2e / kWh (average greening regional grid)
    electricity_tariff_per_kwh: float = 0.18 # USD / kWh
    connection_fixed_fee_usd: float = 350.0  # Hookup fee
    efficiency_loss: float = 0.05          # 5% transformer/cable transmission loss
