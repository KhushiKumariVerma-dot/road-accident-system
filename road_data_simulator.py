import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_FILE = DATA_DIR / "road_accident_data.csv"


def generate_road_dataset(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic road accident dataset for training and analysis."""
    np.random.seed(random_seed)
    n = n_samples

    df = pd.DataFrame({
        "latitude": np.random.uniform(28.45, 28.75, n).round(4),
        "longitude": np.random.uniform(77.00, 77.45, n).round(4),
        "traffic_volume": np.random.randint(100, 1201, n),
        "average_speed": np.random.randint(15, 71, n),
        "weather_condition": np.random.choice(["Clear", "Rain", "Fog"], n),
        "visibility": np.random.uniform(1.5, 10.0, n).round(1),
        "road_condition": np.random.choice(["Dry", "Wet", "Damaged"], n),
        "historical_accidents": np.random.randint(0, 21, n)
    })

    # Synthetic risk scoring formula
    score = (
        df["traffic_volume"] / 300.0
        + np.maximum(50 - df["average_speed"], 0) / 15.0
        + (10.0 - df["visibility"]) / 3.0
        + df["historical_accidents"] / 5.0
        + df["weather_condition"].map({"Clear": 0.0, "Rain": 1.0, "Fog": 1.5})
        + df["road_condition"].map({"Dry": 0.0, "Wet": 0.8, "Damaged": 1.2})
    )

    df["risk"] = pd.cut(
        score,
        bins=[-np.inf, 5.0, 8.0, np.inf],
        labels=["Low", "Medium", "High"]
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    return df


if __name__ == "__main__":
    df = generate_road_dataset(500)
    print(f"Dataset created: {len(df)} records saved to {OUTPUT_FILE}")
    print("\nRisk Distribution:")
    print(df["risk"].value_counts())