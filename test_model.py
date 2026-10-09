import sys
import json
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from utils import (
    load_model_package,
    load_sensor_data,
    save_sensor_reading,
    load_dataset,
    predict_risk,
    explain_risk,
    GEOJSON_FILE,
    RISK_MAP
)

BASE_DIR = Path(__file__).resolve().parent


def run_tests():
    print("=" * 65)
    print("🚦 Running Road Accident Risk Project Verification Test Suite")
    print("=" * 65)
    passed = 0
    total = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, total
        total += 1
        if condition:
            print(f"  [PASS] Test {total}: {name}")
            passed += 1
        else:
            print(f"  [FAIL] Test {total}: {name} -> {details}")

    # Test 1: Load Model Package
    pkg = load_model_package()
    assert_test(
        "Model Package Loading",
        pkg is not None and "model" in pkg,
        "Failed to load trained model package"
    )

    # Test 2: Model Accuracy >= 80%
    accuracy = pkg.get("accuracy", 0.0)
    assert_test(
        f"Model Test Accuracy Verification ({accuracy * 100:.1f}%)",
        accuracy >= 0.80,
        f"Accuracy {accuracy} is below expected threshold 0.80"
    )

    # Test 3: Model Feature Names Count
    features = pkg.get("feature_names", [])
    assert_test(
        f"Model Features Registered ({len(features)} features)",
        len(features) == 8,
        f"Expected 8 features, got {features}"
    )

    # Test 4: High Risk Scenario Prediction
    high_scenario = {
        "latitude": 28.60,
        "longitude": 77.20,
        "traffic_volume": 1100,
        "average_speed": 22.0,
        "weather_condition": "Rain",
        "visibility": 2.0,
        "road_condition": "Damaged",
        "historical_accidents": 18
    }
    risk_high, probs_high = predict_risk(high_scenario, return_proba=True)
    assert_test(
        f"High Risk Prediction (Predicted: {risk_high}, P(High)={probs_high['High']*100:.1f}%)",
        risk_high == "High",
        f"Expected 'High', got {risk_high}"
    )

    # Test 5: Low Risk Scenario Prediction
    low_scenario = {
        "latitude": 28.50,
        "longitude": 77.10,
        "traffic_volume": 120,
        "average_speed": 40.0,
        "weather_condition": "Clear",
        "visibility": 9.5,
        "road_condition": "Dry",
        "historical_accidents": 0
    }
    risk_low, probs_low = predict_risk(low_scenario, return_proba=True)
    assert_test(
        f"Low Risk Prediction (Predicted: {risk_low}, P(Low)={probs_low['Low']*100:.1f}%)",
        risk_low == "Low",
        f"Expected 'Low', got {risk_low}"
    )

    # Test 6: Probability Distribution Normalization
    prob_sum = sum(probs_high.values())
    assert_test(
        f"Prediction Probabilities Sum to 1.0 (Sum: {prob_sum:.4f})",
        abs(prob_sum - 1.0) < 1e-4,
        f"Probability sum is not 1.0: {prob_sum}"
    )

    # Test 7: Explanation Engine
    exp = explain_risk(high_scenario, risk_high)
    assert_test(
        "Risk Explanation Generation",
        len(exp) > 20 and "Key risk drivers" in exp,
        f"Unexpected explanation format: {exp}"
    )

    # Test 8: Sensor Simulator Generation
    sensor = save_sensor_reading()
    assert_test(
        "IoT Sensor Reading Generation & Persistence",
        isinstance(sensor, dict) and "timestamp" in sensor and "traffic_volume" in sensor,
        "Sensor data structure is invalid"
    )

    # Test 9: Road Accident Dataset Loading
    df_road = load_dataset("road_accident")
    assert_test(
        f"Road Accident Dataset Integrity ({len(df_road)} records)",
        len(df_road) >= 500 and "risk" in df_road.columns,
        "Dataset missing or lacks required columns"
    )

    # Test 10: Clean Crash Database Loading
    try:
        df_crash = load_dataset("crash_report")
        assert_test(
            f"Historical Crash Database ({len(df_crash):,} records)",
            len(df_crash) > 1000 and "accident_risk" in df_crash.columns,
            "Crash report dataset missing or lacks accident_risk column"
        )
    except Exception as e:
        assert_test("Historical Crash Database Loading", False, str(e))

    # Test 11: GeoJSON Hotspots File
    if not GEOJSON_FILE.exists():
        from geospatial_analysis import generate_geojson
        generate_geojson()

    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)
    assert_test(
        f"GeoJSON Hotspots File ({len(geojson_data.get('features', []))} locations)",
        geojson_data.get("type") == "FeatureCollection" and len(geojson_data.get("features", [])) > 0,
        "GeoJSON file format invalid or empty"
    )

    print("=" * 65)
    print(f"Test Results: {passed}/{total} Passed ({(passed/total)*100:.1f}%)")
    print("=" * 65)

    if passed == total:
        print("ALL TESTS PASSED SUCCESSFULLY! The system is fully operational.")
        return 0
    else:
        print(f"FAILED: {total - passed} tests failed.")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)