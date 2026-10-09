import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
SENSOR_FILE = DATA_DIR / "latest_sensor_data.json"

# Possible weather conditions
WEATHER = ["Clear", "Rain", "Fog"]

# Possible road conditions
ROAD = ["Dry", "Wet", "Damaged"]


def generate_sensor_data() -> dict:
    """Generate a realistic random IoT road sensor data packet."""
    return {
        "timestamp": datetime.now().isoformat(),
        "latitude": round(random.uniform(28.45, 28.75), 4),
        "longitude": round(random.uniform(77.00, 77.45), 4),
        "traffic_volume": random.randint(100, 1200),
        "average_speed": random.randint(15, 70),
        "weather_condition": random.choice(WEATHER),
        "visibility": round(random.uniform(1.5, 10.0), 1),
        "road_condition": random.choice(ROAD),
        "historical_accidents": random.randint(0, 20)
    }


def save_sensor_data(data: dict = None) -> dict:
    """Save sensor data to the latest_sensor_data.json file."""
    if data is None:
        data = generate_sensor_data()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SENSOR_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Road Sensor Simulator")
    parser.add_argument("--once", action="store_true", help="Generate single sensor reading and exit")
    parser.add_argument("--interval", type=float, default=5.0, help="Interval in seconds between readings")
    args = parser.parse_args()

    if args.once:
        data = save_sensor_data()
        print("Generated single IoT sensor reading:")
        print(json.dumps(data, indent=2))
    else:
        print("IoT Road Sensor Simulator Started")
        print(f"Broadcasting telemetry every {args.interval}s. Press CTRL+C to stop.\n")
        try:
            while True:
                data = save_sensor_data()
                print(f"[{data['timestamp']}] Sensor Telemetry Updated:")
                print(f"  Coordinates: ({data['latitude']}, {data['longitude']})")
                print(f"  Traffic: {data['traffic_volume']} veh/hr | Speed: {data['average_speed']} km/h | Visibility: {data['visibility']} km")
                print(f"  Weather: {data['weather_condition']} | Road: {data['road_condition']} | Past Accidents: {data['historical_accidents']}")
                print("-" * 60)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nSensor simulator stopped.")