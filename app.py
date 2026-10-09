from pathlib import Path
from flask import Flask, request, jsonify
from utils import (
    load_model_package,
    load_sensor_data,
    save_sensor_reading,
    load_dataset,
    predict_risk,
    explain_risk,
    GEOJSON_FILE
)
import json

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,PUT,POST,DELETE,OPTIONS"
    return response


@app.route("/", methods=["GET"])
def home():
    pkg = load_model_package()
    return jsonify({
        "status": "running",
        "service": "Road Accident Risk Prediction Flask API",
        "model": "XGBoost Classifier",
        "accuracy": f"{pkg.get('accuracy', 0.0) * 100:.2f}%",
        "routes": [
            "/health",
            "/predict (POST)",
            "/sensor/latest (GET)",
            "/sensor/simulate (POST)",
            "/model/info (GET)",
            "/stats (GET)",
            "/geojson (GET)"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    try:
        pkg = load_model_package()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "accuracy": pkg.get("accuracy", 0.0)
        })
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "JSON payload required"}), 400

        required_fields = [
            "latitude",
            "longitude",
            "traffic_volume",
            "average_speed",
            "weather_condition",
            "visibility",
            "road_condition",
            "historical_accidents"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": "Missing required fields", "fields": missing}), 400

        pkg = load_model_package()
        model = pkg["model"]
        weather_map = pkg["weather_map"]
        road_map = pkg["road_map"]

        if data["weather_condition"] not in weather_map:
            return jsonify({
                "error": f"Invalid weather_condition '{data['weather_condition']}'. Expected: {list(weather_map.keys())}"
            }), 400

        if data["road_condition"] not in road_map:
            return jsonify({
                "error": f"Invalid road_condition '{data['road_condition']}'. Expected: {list(road_map.keys())}"
            }), 400

        risk_label, prob_dict = predict_risk(data, model, weather_map, road_map, return_proba=True)
        explanation = explain_risk(data, risk_label)

        return jsonify({
            "predicted_risk": risk_label,
            "probabilities": prob_dict,
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/sensor/latest", methods=["GET"])
def get_latest_sensor():
    sensor_data = load_sensor_data()
    risk_label, prob_dict = predict_risk(sensor_data, return_proba=True)
    explanation = explain_risk(sensor_data, risk_label)
    return jsonify({
        "sensor_data": sensor_data,
        "predicted_risk": risk_label,
        "probabilities": prob_dict,
        "explanation": explanation
    })


@app.route("/sensor/simulate", methods=["POST", "GET"])
def simulate_sensor():
    new_data = save_sensor_reading()
    risk_label, prob_dict = predict_risk(new_data, return_proba=True)
    explanation = explain_risk(new_data, risk_label)
    return jsonify({
        "message": "New sensor reading generated",
        "sensor_data": new_data,
        "predicted_risk": risk_label,
        "probabilities": prob_dict,
        "explanation": explanation
    })


@app.route("/model/info", methods=["GET"])
def model_info():
    pkg = load_model_package()
    return jsonify({
        "model_name": "XGBoost Road Risk Classifier",
        "accuracy": pkg.get("accuracy", 0.0),
        "features": pkg.get("feature_names", []),
        "feature_importances": pkg.get("feature_importances", {}),
        "classes": list(pkg.get("risk_map", {}).keys())
    })


@app.route("/stats", methods=["GET"])
def stats():
    df = load_dataset("road_accident")
    return jsonify({
        "total_records": len(df),
        "risk_distribution": df["risk"].value_counts().to_dict(),
        "avg_traffic_volume": round(float(df["traffic_volume"].mean()), 2),
        "avg_speed_kmh": round(float(df["average_speed"].mean()), 2),
        "avg_visibility_km": round(float(df["visibility"].mean()), 2)
    })


@app.route("/geojson", methods=["GET"])
def get_geojson():
    if not GEOJSON_FILE.exists():
        from geospatial_analysis import generate_geojson
        generate_geojson()

    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    risk_filter = request.args.get("risk")
    if risk_filter:
        risk_filter = risk_filter.capitalize()
        geojson_data["features"] = [
            feat for feat in geojson_data.get("features", [])
            if feat.get("properties", {}).get("risk") == risk_filter
        ]

    return jsonify(geojson_data)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)