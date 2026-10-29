import os
import sys
import types
from pathlib import Path

# Ensure xgboost compatibility when Windows Application Control restricts certain sklearn DLLs
try:
    import sklearn.model_selection
except Exception:
    mock_ms = types.ModuleType("sklearn.model_selection")
    mock_ms.StratifiedKFold = object
    sys.modules["sklearn.model_selection"] = mock_ms

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
DATA_FILE = DATA_DIR / "road_accident_data.csv"
MODEL_FILE = BASE_DIR / "road_risk_model.joblib"


def train_model():
    print("=" * 50)
    print("Road Risk Model Training Pipeline")
    print("=" * 50)

    # If dataset doesn't exist, generate it
    if not DATA_FILE.exists():
        print(f"Dataset not found at {DATA_FILE}. Generating synthetic dataset...")
        from road_data_simulator import generate_road_dataset
        generate_road_dataset(500)

    print(f"Loading training data from {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE)

    weather_map = {
        "Clear": 0,
        "Rain": 1,
        "Fog": 2
    }

    road_map = {
        "Dry": 0,
        "Wet": 1,
        "Damaged": 2
    }

    risk_map = {
        "Low": 0,
        "Medium": 1,
        "High": 2
    }

    inv_risk_map = {v: k for k, v in risk_map.items()}

    # Convert categorical variables
    df["weather_condition"] = df["weather_condition"].map(weather_map)
    df["road_condition"] = df["road_condition"].map(road_map)
    df["risk"] = df["risk"].map(risk_map)

    feature_names = [
        "latitude",
        "longitude",
        "traffic_volume",
        "average_speed",
        "weather_condition",
        "visibility",
        "road_condition",
        "historical_accidents"
    ]

    X = df[feature_names]
    y = df["risk"]

    # 80% train, 20% test split
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split = int(len(df) * 0.8)

    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Initialize and train XGBoost classifier
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate predictions
    predictions = model.predict(X_test)
    accuracy = float(np.mean(predictions == y_test))

    print(f"\nModel Test Accuracy: {accuracy * 100:.2f}%")

    # Feature importances
    importances = dict(zip(feature_names, [float(v) for v in model.feature_importances_]))
    print("\nFeature Importances:")
    for feature, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feature:22s}: {imp * 100:.2f}%")

    # Package model metadata
    model_package = {
        "model": model,
        "weather_map": weather_map,
        "road_map": road_map,
        "risk_map": risk_map,
        "inv_risk_map": inv_risk_map,
        "feature_names": feature_names,
        "feature_importances": importances,
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }

    joblib.dump(model_package, MODEL_FILE)
    print(f"\nTrained model and mappings saved to: {MODEL_FILE}")
    print("=" * 50)
    return model_package


if __name__ == "__main__":
    train_model()