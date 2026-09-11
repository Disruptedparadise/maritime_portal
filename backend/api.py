"""
Green Fleet Optimizer - FastAPI Backend
Serves the Maritime Portal static files and exposes the optimization API.
Run: python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from maritime_specs import VESSEL_CLASSES, FUEL_DATABASE
from hydrodynamics import calculate_voyage_metrics

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parent.parent   # c:\antigravity_sih
sys.path.insert(0, str(ROOT))
DATA_DIR   = ROOT / "data"
MODELS_DIR = ROOT / "models"
PORTAL_DIR = ROOT / "web_portal"

# ---------------------------------------------------------------------------
# Lifespan: load everything once at startup
# ---------------------------------------------------------------------------
_state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load JSON data
    with open(MODELS_DIR / "model_benchmark_results.json") as f:
        _state["benchmark"] = json.load(f)
    with open(DATA_DIR / "fleet_optimization_results.json") as f:
        _state["fleet"] = json.load(f)

    # Load ML models (graceful fallback if joblib fails)
    try:
        _state["preprocessor"] = joblib.load(MODELS_DIR / "preprocessor.joblib")
        _state["hq_model"]     = joblib.load(MODELS_DIR / "hybrid_quantum_predictor.joblib")
        _state["gb_model"]     = joblib.load(MODELS_DIR / "gradient_boosting_predictor.joblib")
        _state["models_loaded"] = True
        print("[OK] ML models loaded.")
    except Exception as e:
        print(f"[WARN] Could not load ML models: {e}")
        _state["models_loaded"] = False

    yield
    _state.clear()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Green Fleet Optimizer API",
    description="Quantum-Inspired Maritime Fleet Optimization - SIH 2025",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Maritime Portal static files at /portal
if PORTAL_DIR.exists():
    app.mount("/portal", StaticFiles(directory=str(PORTAL_DIR), html=True), name="portal")

# ---------------------------------------------------------------------------
# Pydantic request schema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    vessel_class:       str   = "Container Feeder"
    fuel_type:          str   = "HFO"
    speed_knots:        float = 14.0
    distance_nm:        float = 5000.0
    cargo_load_factor:  float = 0.75
    wave_height_m:      float = 1.5
    carbon_tax_usd_ton: float = 50.0

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
TRADE_ROUTES = [
    {"id":0,"name":"Shanghai-Rotterdam",    "origin":"Shanghai",     "dest":"Rotterdam",   "distance_nm":11800,"origin_ll":[31.2,121.5], "dest_ll":[51.9,4.5]},
    {"id":1,"name":"Shenzhen-Los Angeles",  "origin":"Shenzhen",     "dest":"Los Angeles", "distance_nm":6400, "origin_ll":[22.5,114.1], "dest_ll":[33.7,-118.2]},
    {"id":2,"name":"Singapore-Port Klang",  "origin":"Singapore",    "dest":"Port Klang",  "distance_nm":170,  "origin_ll":[1.3,103.8],  "dest_ll":[3.0,101.4]},
    {"id":3,"name":"Hamburg-New York",      "origin":"Hamburg",      "dest":"New York",    "distance_nm":3700, "origin_ll":[53.5,10.0],  "dest_ll":[40.7,-74.0]},
    {"id":4,"name":"Dubai-Mumbai",          "origin":"Dubai",        "dest":"Mumbai",      "distance_nm":1200, "origin_ll":[25.2,55.3],  "dest_ll":[19.1,72.9]},
    {"id":5,"name":"Port Hedland-Qingdao",  "origin":"Port Hedland", "dest":"Qingdao",     "distance_nm":4500, "origin_ll":[-20.3,118.6],"dest_ll":[36.1,120.4]},
    {"id":6,"name":"Santos-Rotterdam",      "origin":"Santos",       "dest":"Rotterdam",   "distance_nm":5700, "origin_ll":[-23.9,-46.3],"dest_ll":[51.9,4.5]},
    {"id":7,"name":"Ras Tanura-Singapore",  "origin":"Ras Tanura",   "dest":"Singapore",   "distance_nm":3600, "origin_ll":[26.6,50.2],  "dest_ll":[1.3,103.8]},
    {"id":8,"name":"Busan-Tokyo",           "origin":"Busan",        "dest":"Tokyo",       "distance_nm":620,  "origin_ll":[35.1,129.0], "dest_ll":[35.7,139.7]},
    {"id":9,"name":"Houston-Antwerp",       "origin":"Houston",      "dest":"Antwerp",     "distance_nm":5000, "origin_ll":[29.8,-95.0], "dest_ll":[51.2,4.4]},
]

FUEL_COST_PER_KG  = {"HFO":0.65,"LNG":0.85,"Methanol":1.10,"Ammonia":1.40,"Hydrogen":2.50}
FUEL_EMISS_FACTOR = {"HFO":3.17,"LNG":2.75,"Methanol":0.80,"Ammonia":0.05,"Hydrogen":0.0}
VESSEL_BASE_RATE  = {"Container Feeder":6.5,"Panamax Container":14.2,"ULCV":32.1,
                     "Bulk Handysize":4.8,"Capesize Bulk":28.4,"Aframax Tanker":18.6}

@app.get("/")
async def root():
    return {"message": "Green Fleet Optimizer API is running", "docs": "/docs"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "models_loaded": _state.get("models_loaded", False)}

@app.get("/api/benchmark")
async def get_benchmark():
    return _state["benchmark"]

@app.get("/api/baseline")
async def get_baseline():
    return _state["fleet"]["baseline_fleet"]

@app.get("/api/pareto")
async def get_pareto():
    return _state["fleet"]["quantum_pareto_frontier"]

@app.get("/api/routes")
async def get_routes():
    return TRADE_ROUTES

@app.get("/api/optimize")
async def get_optimization():
    fleet = _state["fleet"]
    return {
        "baseline":               fleet["baseline_fleet"],
        "comparison_metrics":     fleet["comparison_metrics"],
        "quantum_representative": fleet["quantum_representative_solutions"],
        "pareto_count":           len(fleet["quantum_pareto_frontier"]),
    }

@app.post("/api/predict")
async def predict_fuel(req: PredictRequest):
    """Predict fuel consumption using the Hybrid Quantum-Classical model."""
    vessel_norm = {
        "Container Feeder": "Container_Feeder",
        "Panamax Container": "Container_Panamax",
        "ULCV": "Container_ULCV",
        "Bulk Handysize": "Bulk_Handysize",
        "Capesize Bulk": "Bulk_Capesize",
        "Aframax Tanker": "Tanker_Aframax",
    }.get(req.vessel_class, req.vessel_class)

    fuel_code = req.fuel_type if req.fuel_type in FUEL_DATABASE else "HFO"
    fuel_props = FUEL_DATABASE[fuel_code]

    if _state.get("models_loaded"):
        try:
            pre = _state["preprocessor"]
            hq = _state["hq_model"]

            m = calculate_voyage_metrics(
                vessel_class_name=vessel_norm,
                fuel_code=fuel_code,
                speed_knots=req.speed_knots,
                distance_nm=req.distance_nm,
                cargo_load_factor=req.cargo_load_factor,
                wave_height_m=req.wave_height_m,
                carbon_tax_usd_per_ton=req.carbon_tax_usd_ton
            )

            displacement = m["displacement_ton"]
            power_proxy = (displacement ** (2.0 / 3.0)) * (req.speed_knots ** 3.0) / 1000.0

            row = {
                "speed_knots": req.speed_knots,
                "distance_nm": req.distance_nm,
                "cargo_load_factor": req.cargo_load_factor,
                "displacement_ton": displacement,
                "added_resistance_pct": m["added_resistance_pct"],
                "fuel_lhv_mj_per_kg": fuel_props.lhv_mj_per_kg,
                "fuel_density_kg_m3": fuel_props.density_kg_per_m3,
                "fuel_cost_ton": fuel_props.cost_per_metric_ton,
                "power_proxy": power_proxy,
            }

            for col in pre.feature_names:
                if col.startswith("vessel_"):
                    row[col] = 1.0 if col == f"vessel_{vessel_norm}" else 0.0
                elif col.startswith("fuel_"):
                    row[col] = 1.0 if col == f"fuel_{fuel_code}" else 0.0

            df_row = pd.DataFrame([row])[pre.feature_names]
            Xc = pre.scaler.transform(df_row.values)
            X_reduced = pre.pca.transform(Xc)
            denom = np.where((pre._max_vals - pre._min_vals) == 0, 1.0, pre._max_vals - pre._min_vals)
            Xq = 0.1 * np.pi + 0.8 * np.pi * np.clip((X_reduced - pre._min_vals) / denom, 0.0, 1.0)

            preds = hq.predict(Xq, Xc)
            fuel_kg = float(preds[0])
            model_used = "Hybrid Quantum-Classical (HQCKL)"
            r2 = 0.9291
            quantum_angles = [round(float(a), 4) for a in Xq[0]]
            prop_power = round(m.get("propulsion_power_kw", 0.0), 1)
            admiralty_kg = round(VESSEL_BASE_RATE.get(req.vessel_class, 6.5) * req.distance_nm * ((req.speed_knots / 14.0) ** 3) * (1.0 + 0.08 * req.wave_height_m), 1)
            cii_g = round(float(m.get("cii_rating_g_co2_dwt_nm", 0.0)), 3)
        except Exception as e:
            base_rate = VESSEL_BASE_RATE.get(req.vessel_class, 6.5)
            fuel_kg = base_rate * req.distance_nm * ((req.speed_knots / 14.0) ** 3) * (1.0 + 0.08 * req.wave_height_m)
            model_used = f"Admiralty Formula (fallback: {e})"
            r2 = None
            quantum_angles = [0.0] * 6
            prop_power = 0.0
            displacement = 0.0
            admiralty_kg = round(fuel_kg, 1)
            cii_g = 6.2
    else:
        base_rate = VESSEL_BASE_RATE.get(req.vessel_class, 6.5)
        fuel_kg = base_rate * req.distance_nm * ((req.speed_knots / 14.0) ** 3) * (1.0 + 0.08 * req.wave_height_m)
        model_used = "Admiralty Formula (fallback - models not loaded)"
        r2 = None
        quantum_angles = [0.0] * 6
        prop_power = 0.0
        displacement = 0.0
        admiralty_kg = round(fuel_kg, 1)
        cii_g = 6.2

    cost_usd = fuel_kg * (fuel_props.cost_per_metric_ton / 1000.0)
    emissions_kg = fuel_kg * fuel_props.wtw_co2e_factor
    carbon_cost = (emissions_kg / 1000.0) * req.carbon_tax_usd_ton

    if cii_g <= 4.5:
        cii_grade = "A"
    elif cii_g <= 6.5:
        cii_grade = "B"
    elif cii_g <= 8.5:
        cii_grade = "C"
    elif cii_g <= 11.0:
        cii_grade = "D"
    else:
        cii_grade = "E"

    return {
        "fuel_consumed_kg": round(fuel_kg, 1),
        "fuel_consumed_tons": round(fuel_kg / 1000.0, 2),
        "cost_usd": round(cost_usd, 2),
        "carbon_cost_usd": round(carbon_cost, 2),
        "total_cost_usd": round(cost_usd + carbon_cost, 2),
        "emissions_kg_co2e": round(emissions_kg, 1),
        "emissions_tons_co2e": round(emissions_kg / 1000.0, 2),
        "model_used": model_used,
        "r2_score": r2,
        "quantum_angles": quantum_angles,
        "displacement_ton": round(displacement, 1),
        "propulsion_power_kw": prop_power,
        "admiralty_baseline_kg": admiralty_kg,
        "cii_rating_g_co2_dwt_nm": cii_g,
        "cii_grade": cii_grade,
        "fuel_lhv_mj_per_kg": fuel_props.lhv_mj_per_kg,
        "fuel_cost_per_ton": fuel_props.cost_per_metric_ton,
        "fuel_wtw_factor": fuel_props.wtw_co2e_factor
    }

# Catch-all for portal SPA routing
@app.get("/portal/{full_path:path}")
async def serve_portal(full_path: str):
    idx = PORTAL_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    raise HTTPException(status_code=404, detail="Portal not built yet")
