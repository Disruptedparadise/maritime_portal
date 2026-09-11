"""
Fleet Environment and Multi-Objective Constraint Engine.
Models realistic global maritime trade routes, cargo demand constraints,
delivery schedule deadlines (ETAs), and port cold-ironing infrastructure.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE, ShorePowerSpecs
from hydrodynamics import calculate_voyage_metrics


@dataclass(frozen=True)
class FleetRouteDemand:
    """Route specifications and customer demand."""
    route_id: str
    origin_port: str
    destination_port: str
    distance_nm: float
    cargo_type: str            # "Container", "Bulk Carrier", "Tanker"
    demand_quantity: float     # TEU (for Container) or Metric Tons (for Bulk/Tanker)
    deadline_hours: float      # Required arrival ETA deadline
    port_dwell_hours: float    # Docking/loading/unloading duration
    has_shore_power_berth: bool# Whether origin/destination port has AMP connection
    weather_severity: float    # Wind/wave severity factor (1.0 = normal, >1.0 = rough)


# 10 Major Global Commercial Shipping Corridors
FLEET_SCENARIO_ROUTES: List[FleetRouteDemand] = [
    FleetRouteDemand(
        route_id="CORR-01",
        origin_port="Shanghai",
        destination_port="Rotterdam",
        distance_nm=10525.0,
        cargo_type="Container",
        demand_quantity=14500.0,  # Requires large ULCV
        deadline_hours=580.0,     # ~24 days
        port_dwell_hours=48.0,
        has_shore_power_berth=True,
        weather_severity=1.12
    ),
    FleetRouteDemand(
        route_id="CORR-02",
        origin_port="Shenzhen",
        destination_port="Los Angeles",
        distance_nm=6500.0,
        cargo_type="Container",
        demand_quantity=4800.0,   # Panamax or ULCV
        deadline_hours=360.0,     # ~15 days
        port_dwell_hours=36.0,
        has_shore_power_berth=True,
        weather_severity=1.08
    ),
    FleetRouteDemand(
        route_id="CORR-03",
        origin_port="Singapore",
        destination_port="Port Klang",
        distance_nm=210.0,
        cargo_type="Container",
        demand_quantity=1800.0,   # Feeder
        deadline_hours=36.0,      # 1.5 days
        port_dwell_hours=18.0,
        has_shore_power_berth=False,
        weather_severity=0.98
    ),
    FleetRouteDemand(
        route_id="CORR-04",
        origin_port="Hamburg",
        destination_port="New York",
        distance_nm=3650.0,
        cargo_type="Container",
        demand_quantity=4200.0,   # Panamax
        deadline_hours=240.0,     # 10 days
        port_dwell_hours=30.0,
        has_shore_power_berth=True,
        weather_severity=1.20
    ),
    FleetRouteDemand(
        route_id="CORR-05",
        origin_port="Dubai (Jebel Ali)",
        destination_port="Mumbai (Nhava Sheva)",
        distance_nm=1070.0,
        cargo_type="Container",
        demand_quantity=2200.0,   # Feeder / Small Panamax
        deadline_hours=84.0,      # 3.5 days
        port_dwell_hours=20.0,
        has_shore_power_berth=True,
        weather_severity=1.02
    ),
    FleetRouteDemand(
        route_id="CORR-06",
        origin_port="Port Hedland",
        destination_port="Qingdao",
        distance_nm=3450.0,
        cargo_type="Bulk Carrier",
        demand_quantity=160000.0, # Capesize Iron Ore
        deadline_hours=280.0,     # ~11.5 days
        port_dwell_hours=40.0,
        has_shore_power_berth=False,
        weather_severity=1.05
    ),
    FleetRouteDemand(
        route_id="CORR-07",
        origin_port="Santos (Brazil)",
        destination_port="Rotterdam",
        distance_nm=5480.0,
        cargo_type="Bulk Carrier",
        demand_quantity=32000.0,  # Handysize Agri-Bulk
        deadline_hours=420.0,     # 17.5 days
        port_dwell_hours=36.0,
        has_shore_power_berth=True,
        weather_severity=1.10
    ),
    FleetRouteDemand(
        route_id="CORR-08",
        origin_port="Ras Tanura",
        destination_port="Singapore",
        distance_nm=3720.0,
        cargo_type="Tanker",
        demand_quantity=95000.0,  # Aframax Crude
        deadline_hours=300.0,     # 12.5 days
        port_dwell_hours=32.0,
        has_shore_power_berth=False,
        weather_severity=1.04
    ),
    FleetRouteDemand(
        route_id="CORR-09",
        origin_port="Busan",
        destination_port="Tokyo",
        distance_nm=580.0,
        cargo_type="Container",
        demand_quantity=1900.0,   # Feeder
        deadline_hours=48.0,      # 2 days
        port_dwell_hours=16.0,
        has_shore_power_berth=True,
        weather_severity=1.06
    ),
    FleetRouteDemand(
        route_id="CORR-10",
        origin_port="Houston",
        destination_port="Antwerp",
        distance_nm=4850.0,
        cargo_type="Tanker",
        demand_quantity=90000.0,  # Aframax Clean Products
        deadline_hours=380.0,     # ~16 days
        port_dwell_hours=36.0,
        has_shore_power_berth=True,
        weather_severity=1.15
    )
]


class GreenFleetEnvironment:
    """
    Evaluates multi-objective fleet deployments, calculating:
    1. Total Fleet Operational Cost (Fuel + Port + Shore Power + Carbon Levies + Delay penalties)
    2. Total Fleet Lifecycle Emissions (Well-to-Wake CO2e kg)
    3. Schedule Reliability & Delay Hours
    Enforces capacity and compatibility constraints with penalty functions.
    """
    def __init__(
        self,
        routes: List[FleetRouteDemand] = FLEET_SCENARIO_ROUTES,
        carbon_tax_per_ton: float = 80.0,
        late_penalty_usd_per_hour: float = 1200.0
    ):
        self.routes = routes
        self.n_routes = len(routes)
        self.carbon_tax_per_ton = carbon_tax_per_ton
        self.late_penalty_usd_per_hour = late_penalty_usd_per_hour

        self.vessel_classes_list = list(VESSEL_CLASSES.keys())
        self.fuel_options = list(FUEL_DATABASE.keys())

    def evaluate_fleet_deployment(
        self,
        deployment: List[Dict[str, Any]]
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Evaluates a fleet deployment across all routes.
        Each element in deployment has:
        - vessel_class: str
        - speed_knots: float
        - fuel_type: str
        - use_shore_power: bool

        Returns:
            (total_cost_usd, total_wtw_co2e_kg, total_delay_hours, details_dict)
        """
        if len(deployment) != self.n_routes:
            raise ValueError(f"Expected {self.n_routes} route decisions, got {len(deployment)}")

        total_cost = 0.0
        total_emissions = 0.0
        total_delay = 0.0
        total_penalties = 0.0
        route_summaries = []

        for i, route in enumerate(self.routes):
            decision = deployment[i]
            vessel_name = decision["vessel_class"]
            speed = float(decision["speed_knots"])
            fuel_type = decision["fuel_type"]
            shore_power = bool(decision["use_shore_power"])

            vessel = VESSEL_CLASSES[vessel_name]

            # -------------------------------------------------------------
            # Constraint 1: Category & Capacity Feasibility
            # -------------------------------------------------------------
            capacity_penalty = 0.0
            if vessel.category != route.cargo_type:
                # Incompatible vessel type (e.g. Tanker on Container route)
                capacity_penalty += 800000.0

            if route.cargo_type == "Container":
                max_capacity = float(vessel.capacity_teu or 0)
                if max_capacity < route.demand_quantity:
                    # Ship cannot hold required TEU
                    shortfall = route.demand_quantity - max_capacity
                    capacity_penalty += shortfall * 400.0 # Heavy cargo spill penalty
                cargo_load_factor = min(1.0, route.demand_quantity / max(1.0, max_capacity))
            else:
                max_capacity = vessel.deadweight_ton
                if max_capacity < route.demand_quantity:
                    shortfall = route.demand_quantity - max_capacity
                    capacity_penalty += shortfall * 20.0
                cargo_load_factor = min(1.0, route.demand_quantity / max(1.0, max_capacity))

            # -------------------------------------------------------------
            # Constraint 2: Fuel Compatibility
            # -------------------------------------------------------------
            fuel_penalty = 0.0
            if fuel_type not in vessel.compatible_fuels:
                fuel_penalty += 500000.0 # Unretrofitted engine penalty

            # -------------------------------------------------------------
            # Constraint 3: Shore Power Availability at Port
            # -------------------------------------------------------------
            effective_shore_power = shore_power and route.has_shore_power_berth and vessel.supports_shore_power

            # -------------------------------------------------------------
            # Physics Calculation via Hydrodynamics Engine
            # -------------------------------------------------------------
            # Environmental weather scaling
            wind_spd = 14.0 * route.weather_severity
            wave_ht = 1.6 * route.weather_severity

            # Clamp speed to vessel envelope
            speed_clamped = max(vessel.min_speed_knots, min(vessel.max_speed_knots, speed))

            metrics = calculate_voyage_metrics(
                vessel_class_name=vessel_name,
                fuel_code=fuel_type,
                speed_knots=speed_clamped,
                distance_nm=route.distance_nm,
                cargo_load_factor=cargo_load_factor,
                wind_speed_knots=wind_spd,
                wind_direction_deg=45.0,
                wave_height_m=wave_ht,
                port_dwell_time_hours=route.port_dwell_hours,
                use_shore_power_at_berth=effective_shore_power,
                carbon_tax_usd_per_ton=self.carbon_tax_per_ton
            )

            # -------------------------------------------------------------
            # Constraint 4: Schedule Reliability & Delivery Deadline (ETA)
            # -------------------------------------------------------------
            total_trip_hours = metrics["transit_time_hours"] + metrics["port_dwell_time_hours"]
            delay_hours = max(0.0, total_trip_hours - route.deadline_hours)
            delay_cost = delay_hours * self.late_penalty_usd_per_hour
            # Additional quadratic penalty for severe schedule blowouts
            if delay_hours > 24.0:
                delay_cost += ((delay_hours - 24.0) ** 2) * 50.0

            route_cost = metrics["total_voyage_cost_usd"] + delay_cost + capacity_penalty + fuel_penalty
            route_emissions = metrics["wtw_co2e_emissions_kg"]

            total_cost += route_cost
            total_emissions += route_emissions
            total_delay += delay_hours
            total_penalties += (capacity_penalty + fuel_penalty)

            route_summaries.append({
                "route_id": route.route_id,
                "origin_dest": f"{route.origin_port} -> {route.destination_port}",
                "vessel": vessel_name,
                "speed_knots": speed_clamped,
                "fuel": fuel_type,
                "shore_power_active": effective_shore_power,
                "trip_hours": round(total_trip_hours, 1),
                "deadline_hours": route.deadline_hours,
                "delay_hours": round(delay_hours, 1),
                "cost_usd": round(route_cost, 1),
                "emissions_kg": round(route_emissions, 1),
                "is_feasible": (capacity_penalty == 0 and fuel_penalty == 0 and delay_hours <= 12.0)
            })

        summary = {
            "total_cost_usd": round(total_cost, 2),
            "total_emissions_kg": round(total_emissions, 1),
            "total_delay_hours": round(total_delay, 1),
            "total_penalties_usd": round(total_penalties, 2),
            "is_fully_feasible": (total_penalties == 0.0),
            "routes": route_summaries
        }

        return total_cost, total_emissions, total_delay, summary

    def get_baseline_deployment(self) -> List[Dict[str, Any]]:
        """
        Generates standard industry baseline fleet deployment:
        - Heavy Fuel Oil (HFO) for all vessels
        - Design speeds (no slow-steaming optimization)
        - No Shore Power (auxiliary engines running in port)
        - Standard default vessel category matching
        """
        baseline = []
        for route in self.routes:
            # Pick a default compliant vessel
            if route.cargo_type == "Container":
                if route.demand_quantity > 8000:
                    v = "Container_ULCV"
                elif route.demand_quantity > 3000:
                    v = "Container_Panamax"
                else:
                    v = "Container_Feeder"
            elif route.cargo_type == "Bulk Carrier":
                v = "Bulk_Capesize" if route.demand_quantity > 50000 else "Bulk_Handysize"
            else:
                v = "Tanker_Aframax"

            vessel = VESSEL_CLASSES[v]
            baseline.append({
                "vessel_class": v,
                "speed_knots": vessel.design_speed_knots,
                "fuel_type": "HFO",
                "use_shore_power": False
            })
        return baseline
