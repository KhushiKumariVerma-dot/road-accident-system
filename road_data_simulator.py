import os
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
OUTPUT_FILE = DATA_DIR / "road_accident_data.csv"

# Major Delhi-NCR Corridor and Landmark Anchors
LANDMARKS = [
    # North Delhi
    (28.71, 77.19, "GT Karnal Road / Azadpur Flyover"),
    (28.72, 77.22, "Model Town / Ring Road Junction"),
    (28.74, 77.16, "Rohini Sector 16 / Outer Ring Road"),
    (28.70, 77.28, "Kashmere Gate ISBT / Ring Road"),
    (28.73, 77.32, "Shastri Park / Grand Trunk Road"),
    # Central Delhi
    (28.63, 77.22, "Connaught Place / Outer Circle"),
    (28.62, 77.24, "ITO Intersection / Vikas Marg"),
    (28.61, 77.20, "India Gate / C-Hexagon"),
    (28.64, 77.19, "Karol Bagh / Pusa Road"),
    (28.65, 77.23, "Chandni Chowk / Red Fort Corridor"),
    # South Delhi
    (28.57, 77.21, "Ring Road / AIIMS - Safdarjung Junction"),
    (28.56, 77.24, "Lajpat Nagar / Moolchand Underpass"),
    (28.54, 77.25, "Outer Ring Road / Nehru Place Flyover"),
    (28.53, 77.20, "Hauz Khas / Sri Aurobindo Marg"),
    (28.52, 77.21, "Saket / Press Enclave Road"),
    (28.50, 77.28, "Kalka Mandir / Maa Anandmayee Marg"),
    # West Delhi & Dwarka
    (28.60, 77.08, "Dwarka Expressway / Sector 21 Junction"),
    (28.62, 77.08, "Janakpuri District Centre / Najafgarh Rd"),
    (28.65, 77.12, "Rajouri Garden / Ring Road"),
    (28.67, 77.14, "Punjabi Bagh Chowk / Rohtak Road"),
    (28.59, 77.04, "Dwarka Sector 10 / Metro Corridor"),
    # East Delhi & Noida
    (28.59, 77.31, "DND Flyway / Mayur Vihar Toll"),
    (28.57, 77.34, "Noida Sector 18 / Atta Market Corridor"),
    (28.54, 77.38, "Noida-Gr Noida Expressway (Sec 128)"),
    (28.62, 77.37, "Indirapuram / NH-24 Ghaziabad Link"),
    (28.60, 77.30, "Akshardham / Nizamuddin Bridge Link"),
    # Gurgaon & NH-48 Corridor
    (28.50, 77.08, "NH-48 / Cyber City - Shankar Chowk"),
    (28.48, 77.09, "MG Road / IFFCO Chowk Gurgaon"),
    (28.46, 77.10, "Golf Course Road / DLF Phase 5"),
    (28.47, 77.05, "Sohna Road / Subhash Chowk"),
    (28.46, 77.03, "Hero Honda Chowk / NH-48"),
    # Faridabad Corridor
    (28.49, 77.30, "Mathura Road / Badarpur Border"),
    (28.46, 77.31, "Bata Chowk / NH-19 Faridabad"),
    (28.47, 77.33, "Surajkund Road / Green Valley"),
    (28.50, 77.33, "Sarita Vihar / Apollo Hospital Corridor")
]


def assign_landmark(lat: float, lon: float, idx: int) -> str:
    dists = [(lat - l_lat) ** 2 + (lon - l_lon) ** 2 for l_lat, l_lon, _ in LANDMARKS]
    best_idx = int(np.argmin(dists))
    landmark_title = LANDMARKS[best_idx][2]
    return f"{landmark_title} (Node #{idx + 1:03d})"


def generate_road_dataset(n_samples: int = 500, random_seed: int = 42) -> pd.DataFrame:
    """Generate synthetic road accident dataset for training and analysis."""
    np.random.seed(random_seed)
    n = n_samples

    lats = np.random.uniform(28.45, 28.75, n).round(4)
    lons = np.random.uniform(77.00, 77.45, n).round(4)

    location_names = [assign_landmark(lats[i], lons[i], i) for i in range(n)]

    df = pd.DataFrame({
        "location_name": location_names,
        "latitude": lats,
        "longitude": lons,
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
    print("\nSample locations:")
    print(df[["location_name", "risk", "traffic_volume", "average_speed"]].head(5))