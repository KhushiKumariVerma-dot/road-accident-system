import json
import sys
from pathlib import Path
import pandas as pd

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CSV_FILE = DATA_DIR / "road_accident_data.csv"
GEOJSON_FILE = DATA_DIR / "road_accident_locations.geojson"


def generate_geojson():
    if not CSV_FILE.exists():
        from road_data_simulator import generate_road_dataset
        generate_road_dataset(500)

    df = pd.read_csv(CSV_FILE)

    features = []
    for idx, row in df.iterrows():
        loc_name = row.get("location_name", f"Road Segment #{idx+1}")
        feature = {
            "type": "Feature",
            "id": str(idx),
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])]
            },
            "properties": {
                "location_name": str(loc_name),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "traffic_volume": int(row["traffic_volume"]),
                "average_speed": float(row["average_speed"]),
                "weather_condition": str(row["weather_condition"]),
                "visibility": float(row["visibility"]),
                "road_condition": str(row["road_condition"]),
                "historical_accidents": int(row["historical_accidents"]),
                "risk": str(row["risk"])
            }
        }
        features.append(feature)

    geojson_obj = {
        "type": "FeatureCollection",
        "features": features
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GEOJSON_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, indent=2)

    print("Geospatial analysis completed:")
    print(f"Total road accident points processed: {len(features)}")
    print(f"GeoJSON saved to: {GEOJSON_FILE}")
    return geojson_obj


if __name__ == "__main__":
    generate_geojson()