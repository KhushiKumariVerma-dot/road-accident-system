import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from utils import (
    load_model_package,
    load_sensor_data,
    save_sensor_reading,
    load_dataset,
    predict_risk as utils_predict_risk,
    explain_risk as utils_explain_risk,
    RISK_MAP
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
GEOJSON_FILE = DATA_DIR / "road_accident_locations.geojson"

app = FastAPI(
    title="Road Accident Risk Prediction API",
    description="High-frequency IoT Road Telemetry & XGBoost Machine Learning Accident Risk Evaluation Engine",
    version="1.0.0"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorData(BaseModel):
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude coordinate of road segment", example=28.6139)
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude coordinate of road segment", example=77.2090)
    traffic_volume: int = Field(..., ge=0, le=10000, description="Vehicle count per hour", example=850)
    average_speed: float = Field(..., ge=0.0, le=250.0, description="Average vehicle speed in km/h", example=45.0)
    weather_condition: str = Field(..., description="Weather condition ('Clear', 'Rain', 'Fog')", example="Rain")
    visibility: float = Field(..., ge=0.0, le=50.0, description="Atmospheric visibility in km", example=3.5)
    road_condition: str = Field(..., description="Surface condition ('Dry', 'Wet', 'Damaged')", example="Wet")
    historical_accidents: int = Field(..., ge=0, description="Recorded past accidents on this road segment", example=14)


@app.get("/")
def root():
    pkg = load_model_package()
    return {
        "status": "online",
        "service": "Road Accident Risk Prediction API",
        "version": "1.0.0",
        "documentation": "/docs",
        "model_accuracy": f"{pkg.get('accuracy', 0.0) * 100:.2f}%",
        "endpoints": [
            "GET /health",
            "POST /predict",
            "GET /sensor/latest",
            "POST /sensor/simulate",
            "GET /model/info",
            "GET /stats",
            "GET /geojson"
        ]
    }


@app.get("/health")
def health_check():
    try:
        pkg = load_model_package()
        return {
            "status": "healthy",
            "model_loaded": pkg is not None,
            "model_type": type(pkg["model"]).__name__,
            "accuracy": pkg.get("accuracy", None)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/predict")
def predict_endpoint(sensor: SensorData):
    """Predict accident risk for a given road telemetry payload."""
    pkg = load_model_package()
    model = pkg["model"]
    weather_map = pkg["weather_map"]
    road_map = pkg["road_map"]

    sensor_dict = sensor.model_dump()

    # Validate categorical mappings
    if sensor_dict["weather_condition"] not in weather_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid weather_condition '{sensor_dict['weather_condition']}'. Expected one of: {list(weather_map.keys())}"
        )

    if sensor_dict["road_condition"] not in road_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid road_condition '{sensor_dict['road_condition']}'. Expected one of: {list(road_map.keys())}"
        )

    risk_label, prob_dict = utils_predict_risk(sensor_dict, model, weather_map, road_map, return_proba=True)
    explanation = utils_explain_risk(sensor_dict, risk_label)

    return {
        "risk": risk_label,
        "probabilities": prob_dict,
        "explanation": explanation,
        "input_telemetry": sensor_dict
    }


@app.get("/sensor/latest")
def get_latest_sensor():
    """Retrieve the latest telemetry reading recorded by the IoT sensor network."""
    data = load_sensor_data()
    risk_label, prob_dict = utils_predict_risk(data, return_proba=True)
    explanation = utils_explain_risk(data, risk_label)
    return {
        "sensor_data": data,
        "current_risk": risk_label,
        "probabilities": prob_dict,
        "explanation": explanation
    }


@app.post("/sensor/simulate")
def simulate_sensor_reading():
    """Generate and record a fresh randomized IoT road sensor reading."""
    new_data = save_sensor_reading()
    risk_label, prob_dict = utils_predict_risk(new_data, return_proba=True)
    explanation = utils_explain_risk(new_data, risk_label)
    return {
        "message": "New sensor telemetry generated and stored successfully",
        "sensor_data": new_data,
        "predicted_risk": risk_label,
        "probabilities": prob_dict,
        "explanation": explanation
    }


@app.get("/model/info")
def get_model_info():
    """Retrieve metadata, accuracy, and feature importance rankings of the trained model."""
    pkg = load_model_package()
    return {
        "model_name": "XGBoost Road Risk Classifier",
        "objective": "multi:softprob",
        "accuracy": pkg.get("accuracy", 0.0),
        "classes": list(pkg.get("risk_map", {}).keys()),
        "features": pkg.get("feature_names", []),
        "feature_importances": pkg.get("feature_importances", {}),
        "training_samples": pkg.get("train_samples", 0),
        "test_samples": pkg.get("test_samples", 0)
    }


@app.get("/stats")
def get_dataset_stats():
    """Summary statistics for historical road safety data."""
    df = load_dataset("road_accident")
    risk_counts = df["risk"].value_counts().to_dict()
    return {
        "total_records": len(df),
        "risk_distribution": risk_counts,
        "avg_traffic_volume": round(float(df["traffic_volume"].mean()), 2),
        "avg_speed_kmh": round(float(df["average_speed"].mean()), 2),
        "avg_visibility_km": round(float(df["visibility"].mean()), 2),
        "weather_breakdown": df["weather_condition"].value_counts().to_dict(),
        "road_condition_breakdown": df["road_condition"].value_counts().to_dict()
    }


@app.get("/geojson")
def get_geojson(risk: Optional[str] = Query(None, description="Filter by risk level: Low, Medium, High")):
    """Return a GeoJSON FeatureCollection of road risk hotspots for map rendering."""
    if not GEOJSON_FILE.exists():
        from geospatial_analysis import generate_geojson
        generate_geojson()

    try:
        with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)

        if risk:
            risk_cap = risk.capitalize()
            filtered_features = [
                feat for feat in geojson_data.get("features", [])
                if feat.get("properties", {}).get("risk") == risk_cap
            ]
            geojson_data["features"] = filtered_features

        return JSONResponse(content=geojson_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read GeoJSON: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
