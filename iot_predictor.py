import json
import sys
from pathlib import Path

# Safe encoding on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from utils import load_sensor_data, predict_risk, explain_risk

BASE_DIR = Path(__file__).resolve().parent


def run_prediction():
    sensor_data = load_sensor_data()
    risk, probs = predict_risk(sensor_data, return_proba=True)
    explanation = explain_risk(sensor_data, risk)

    print("=" * 60)
    print("[*] IoT Road Sensor Risk Evaluation")
    print("=" * 60)
    print("Latest Sensor Telemetry:")
    print(json.dumps(sensor_data, indent=2))
    print("-" * 60)
    print(f"Predicted Accident Risk: {risk.upper()}")
    print(f"Confidence Breakdown: Low={probs['Low']*100:.1f}%, Medium={probs['Medium']*100:.1f}%, High={probs['High']*100:.1f}%")
    print("-" * 60)
    print("Analysis & Reasoning:")
    print(explanation)
    print("=" * 60)
    return {
        "sensor": sensor_data,
        "risk": risk,
        "probabilities": probs,
        "explanation": explanation
    }


if __name__ == "__main__":
    run_prediction()