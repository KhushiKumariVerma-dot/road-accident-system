import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
MODEL_FILE = BASE_DIR / "road_risk_model.joblib"
SENSOR_FILE = DATA_DIR / "latest_sensor_data.json"
ROAD_ACCIDENT_FILE = DATA_DIR / "road_accident_data.csv"
CRASH_REPORT_CLEAN_FILE = DATA_DIR / "crash_data_clean.csv"
CRASH_REPORT_RAW_FILE = DATA_DIR / "CrashDataReport.csv"
GEOJSON_FILE = DATA_DIR / "road_accident_locations.geojson"

RISK_MAP = {0: "Low", 1: "Medium", 2: "High"}
INV_RISK_MAP = {"Low": 0, "Medium": 1, "High": 2}

_CACHED_MODEL_PKG = None


def load_model_package() -> dict:
    """Load the full trained model package from disk (cached in memory)."""
    global _CACHED_MODEL_PKG
    if _CACHED_MODEL_PKG is None:
        if not MODEL_FILE.exists():
            from train_ml_model import train_model
            _CACHED_MODEL_PKG = train_model()
        else:
            _CACHED_MODEL_PKG = joblib.load(MODEL_FILE)
    return _CACHED_MODEL_PKG


def load_model() -> Tuple[Any, Dict[str, int], Dict[str, int]]:
    """Load the XGBoost model and mapping dictionaries from joblib."""
    pkg = load_model_package()
    return pkg["model"], pkg["weather_map"], pkg["road_map"]


def load_sensor_data() -> dict:
    """Load the latest IoT sensor telemetry from JSON, or generate one if missing."""
    if not SENSOR_FILE.exists():
        from iot_sensor_simulator import save_sensor_data
        return save_sensor_data()

    try:
        with open(SENSOR_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Error reading sensor file, generating fresh reading: %s", e)
        from iot_sensor_simulator import save_sensor_data
        return save_sensor_data()


def save_sensor_reading(data: Optional[dict] = None) -> dict:
    """Generate and persist a new IoT sensor reading."""
    from iot_sensor_simulator import save_sensor_data
    return save_sensor_data(data)


def load_dataset(dataset_type: str = "road_accident") -> pd.DataFrame:
    """
    Load project datasets.
    - 'road_accident': Synthetic Delhi-NCR 500-segment dataset for risk ML & IoT.
    - 'crash_report': Cleaned Tempe/Arizona 58K-record real crash database.
    """
    if dataset_type == "crash_report":
        if CRASH_REPORT_CLEAN_FILE.exists():
            return pd.read_csv(CRASH_REPORT_CLEAN_FILE, low_memory=False)
        elif CRASH_REPORT_RAW_FILE.exists():
            from prepare_crash_data import prepare_crash_data
            return prepare_crash_data()
        else:
            raise FileNotFoundError(f"Crash report file not found at {CRASH_REPORT_CLEAN_FILE}")

    # Default: 'road_accident'
    if not ROAD_ACCIDENT_FILE.exists():
        from road_data_simulator import generate_road_dataset
        return generate_road_dataset(500)

    df = pd.read_csv(ROAD_ACCIDENT_FILE)
    if "location_name" not in df.columns:
        from road_data_simulator import assign_landmark
        df["location_name"] = [
            assign_landmark(lat, lon, i)
            for i, (lat, lon) in enumerate(zip(df["latitude"], df["longitude"]))
        ]
    return df


def predict_risk(
    sensor: dict,
    model=None,
    weather_map=None,
    road_map=None,
    return_proba: bool = False
):
    """
    Predict risk level (Low, Medium, High) for given sensor data.
    If return_proba=True, returns (risk_label, prob_dict).
    """
    if model is None or weather_map is None or road_map is None:
        model, weather_map, road_map = load_model()

    # Safe mapping with fallbacks
    weather_val = weather_map.get(sensor.get("weather_condition", "Clear"), 0)
    road_val = road_map.get(sensor.get("road_condition", "Dry"), 0)

    input_data = pd.DataFrame([{
        "latitude": float(sensor["latitude"]),
        "longitude": float(sensor["longitude"]),
        "traffic_volume": int(sensor["traffic_volume"]),
        "average_speed": float(sensor["average_speed"]),
        "weather_condition": weather_val,
        "visibility": float(sensor["visibility"]),
        "road_condition": road_val,
        "historical_accidents": int(sensor["historical_accidents"])
    }])

    pred = int(model.predict(input_data)[0])
    risk_label = RISK_MAP.get(pred, "Unknown")

    if return_proba:
        try:
            probs = model.predict_proba(input_data)[0]
            prob_dict = {
                "Low": float(probs[0]),
                "Medium": float(probs[1]),
                "High": float(probs[2])
            }
        except Exception:
            prob_dict = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
            prob_dict[risk_label] = 1.0
        return risk_label, prob_dict

    return risk_label


def explain_risk(sensor: dict, risk: Optional[str] = None) -> str:
    """Generate a detailed, human-readable explanation and recommendations for predicted risk."""
    reasons = []
    actions = []

    traffic = sensor.get("traffic_volume", 0)
    speed = sensor.get("average_speed", 0)
    vis = sensor.get("visibility", 10.0)
    weather = sensor.get("weather_condition", "Clear")
    road = sensor.get("road_condition", "Dry")
    accidents = sensor.get("historical_accidents", 0)

    if traffic > 800:
        reasons.append(f"heavy traffic congestion ({traffic} vehicles/hr)")
        actions.append("reroute non-essential traffic or enable congestion alerts")
    elif traffic > 500:
        reasons.append(f"moderate traffic flow ({traffic} vehicles/hr)")

    if speed > 55:
        reasons.append(f"high vehicle velocity ({speed} km/h)")
        actions.append("enforce dynamic speed limit moderation")
    elif speed < 25 and traffic > 700:
        reasons.append(f"stop-and-go bottleneck traffic ({speed} km/h)")

    if vis < 3.0:
        reasons.append(f"severely restricted visibility ({vis} km)")
        actions.append("activate fog/visibility hazard warning beacons")
    elif vis < 6.0:
        reasons.append(f"reduced visibility ({vis} km)")

    if weather in ["Rain", "Heavy Rain"]:
        reasons.append(f"precipitation reducing tire traction ({weather})")
        actions.append("broadcast wet asphalt caution to motorists")
    elif weather in ["Fog", "Dense Fog"]:
        reasons.append(f"atmospheric fog impairing driver line-of-sight ({weather})")

    if road == "Wet":
        reasons.append("wet road surface with reduced braking grip")
        actions.append("advise increased following distance")
    elif road == "Damaged":
        reasons.append("surface degradation, potholes, or structural road damage")
        actions.append("dispatch municipal highway repair crew")

    if accidents >= 12:
        reasons.append(f"historically hazardous black-spot zone ({accidents} past crashes recorded)")
        actions.append("increase automated traffic monitoring & patrol frequency")
    elif accidents >= 6:
        reasons.append(f"moderate accident history ({accidents} past crashes recorded)")

    if not reasons:
        explanation = "All telemetry metrics (traffic, speed, surface, weather) are within safe baseline operating tolerances."
    else:
        explanation = "Key risk drivers: " + "; ".join(reasons) + "."

    if actions:
        explanation += "\nRecommended interventions: " + "; ".join(actions) + "."

    return explanation
