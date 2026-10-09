from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
INPUT_FILE = DATA_DIR / "CrashDataReport.csv"
OUTPUT_FILE = DATA_DIR / "crash_data_clean.csv"


def prepare_crash_data():
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} does not exist.")
        return None

    print(f"Loading raw crash data from {INPUT_FILE} (this may take a few seconds)...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Convert injury/fatality columns to numeric
    df["Totalinjuries"] = pd.to_numeric(df["Totalinjuries"], errors="coerce").fillna(0)
    df["Totalfatalities"] = pd.to_numeric(df["Totalfatalities"], errors="coerce").fillna(0)

    # Remove records with unknown injury severity
    if "Injuryseverity" in df.columns:
        df = df[~df["Injuryseverity"].isin(["Unknown"])]
        df = df.dropna(subset=["Injuryseverity"])

        # Create 3-level accident risk
        risk_mapping = {
            "No Injury": "Low",
            "Possible Injury": "Low",
            "Suspected Minor Injury": "Medium",
            "Non Incapacitating Injury": "Medium",
            "Incapacitating Injury": "High",
            "Suspected Serious Injury": "High",
            "Fatal": "High"
        }
        df["accident_risk"] = df["Injuryseverity"].map(risk_mapping)
        df = df.dropna(subset=["accident_risk"])

    # Clean coordinates if present
    if "Latitude" in df.columns and "Longitude" in df.columns:
        df = df.dropna(subset=["Latitude", "Longitude"])
        df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]

    df.to_csv(OUTPUT_FILE, index=False)

    print("Cleaning completed successfully!")
    print(f"Total valid records: {len(df):,}")
    if "accident_risk" in df.columns:
        print("\nRisk distribution:")
        print(df["accident_risk"].value_counts())
    print(f"\nSaved clean dataset to: {OUTPUT_FILE}")
    return df


if __name__ == "__main__":
    prepare_crash_data()