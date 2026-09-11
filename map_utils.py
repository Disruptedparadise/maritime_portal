"""
Geospatial Maritime Mapping Utilities.
Renders interactive global shipping routes, port infrastructure,
and cold-ironing shore power berths using Folium.
"""

import folium
from typing import List, Dict, Any

PORT_COORDINATES = {
    "Shanghai": (31.2304, 121.4737),
    "Rotterdam": (51.9244, 4.4777),
    "Shenzhen": (22.5431, 114.0579),
    "Los Angeles": (33.7432, -118.2673),
    "Singapore": (1.3521, 103.8198),
    "Port Klang": (3.0039, 101.3924),
    "Hamburg": (53.5511, 9.9937),
    "New York": (40.7128, -74.0060),
    "Dubai (Jebel Ali)": (24.9857, 55.0273),
    "Mumbai (Nhava Sheva)": (18.9499, 72.9515),
    "Port Hedland": (-20.3122, 118.5760),
    "Qingdao": (36.0671, 120.3826),
    "Santos (Brazil)": (-23.9618, -46.3322),
    "Ras Tanura": (26.6745, 50.1633),
    "Busan": (35.1796, 129.0756),
    "Tokyo": (35.6762, 139.6503),
    "Houston": (29.7604, -95.3698),
    "Antwerp": (51.2194, 4.4025)
}


def create_global_shipping_map(route_summaries: List[Dict[str, Any]]) -> folium.Map:
    """
    Creates an interactive Folium map centered globally,
    drawing shipping lanes, port icons, and deployment data.
    """
    m = folium.Map(
        location=[22.0, 30.0],
        zoom_start=2,
        tiles="CartoDB positron"
    )

    ports_drawn = set()

    for r in route_summaries:
        orig_name, dest_name = r["origin_dest"].split(" -> ")
        orig_coord = PORT_COORDINATES.get(orig_name.strip())
        dest_coord = PORT_COORDINATES.get(dest_name.strip())

        if not orig_coord or not dest_coord:
            continue

        # Color based on fuel type
        fuel = r.get("fuel", "HFO")
        fuel_colors = {
            "HFO": "#e74c3c",      # Red
            "LNG": "#3498db",      # Blue
            "Methanol": "#f39c12", # Orange
            "Ammonia": "#2ecc71",  # Green
            "Hydrogen": "#9b59b6"  # Purple
        }
        line_color = fuel_colors.get(fuel, "#34495e")

        # Draw shipping route polyline
        folium.PolyLine(
            locations=[orig_coord, dest_coord],
            color=line_color,
            weight=3.5,
            opacity=0.8,
            tooltip=(
                f"<b>{r['route_id']}</b>: {orig_name} → {dest_name}<br>"
                f"Vessel: {r.get('vessel', 'N/A')}<br>"
                f"Speed: {r.get('speed_knots', 0)} kts | Fuel: {fuel}<br>"
                f"Shore Power: {'Active' if r.get('shore_power_active') else 'Off'}<br>"
                f"Voyage Cost: ${r.get('cost_usd', 0):,.0f}"
            )
        ).add_to(m)

        # Draw Port Markers
        for p_name, p_coord in [(orig_name, orig_coord), (dest_name, dest_coord)]:
            if p_name not in ports_drawn:
                ports_drawn.add(p_name)
                folium.CircleMarker(
                    location=p_coord,
                    radius=6,
                    color="#2c3e50",
                    fill=True,
                    fill_color="#1abc9c",
                    fill_opacity=0.9,
                    popup=f"<b>Port:</b> {p_name}"
                ).add_to(m)

    return m
