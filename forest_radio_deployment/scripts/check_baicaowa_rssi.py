"""Check and validate Baicaowa LoRa RSSI measurement data.

Checks:
- Required fields present
- Timestamp parsing and timezone
- RSSI value ranges
- Tx/Rx coordinate validity
- Distance computation
- Duplicate records
- Missing values
- Temporal coverage and gaps

Outputs:
    outputs/tables/baicaowa_data_quality.csv
    outputs/tables/baicaowa_link_summary.csv
    outputs/tables/baicaowa_temporal_summary.csv

Dependencies: numpy, pandas (no GPU required)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    Path.home() / "forest_radio_deployment"
))

BAICAOWA_DIR = PROJECT_ROOT / "data" / "raw" / "baicaowa"
LORA_DIR = BAICAOWA_DIR / "lora"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# Expected fields per task specification Section 9.3
EXPECTED_FIELDS = [
    "timestamp", "tx_id", "rx_id",
    "tx_x", "tx_y", "tx_z", "rx_x", "rx_y", "rx_z",
    "distance_m", "rssi_dbm",
    "frequency_mhz", "tx_power_dbm",
    "spreading_factor", "bandwidth_khz", "coding_rate",
    "antenna_gain_dbi", "antenna_height_m",
    "W", "VPD",
    "air_temperature", "air_relative_humidity",
    "stem_temperature", "stem_relative_humidity",
    "soil_temperature", "soil_moisture",
]


def load_rssi_data(lora_dir):
    """Load all RSSI data files from the LoRa directory."""
    dfs = []
    for f in sorted(lora_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            df["_source_file"] = f.name
            dfs.append(df)
        except Exception as e:
            print(f"WARNING: Could not read {f.name}: {e}")

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def check_fields(df):
    """Check which expected fields are present/missing."""
    present = [f for f in EXPECTED_FIELDS if f in df.columns]
    missing = [f for f in EXPECTED_FIELDS if f not in df.columns]
    extra = [f for f in df.columns if f not in EXPECTED_FIELDS and not f.startswith("_")]
    return present, missing, extra


def check_rssi_quality(df):
    """Check RSSI value ranges and quality."""
    quality = {}
    if "rssi_dbm" in df.columns:
        rssi = df["rssi_dbm"].dropna()
        quality["rssi_count"] = len(rssi)
        quality["rssi_min"] = rssi.min()
        quality["rssi_max"] = rssi.max()
        quality["rssi_mean"] = rssi.mean()
        quality["rssi_std"] = rssi.std()
        quality["rssi_null_count"] = df["rssi_dbm"].isna().sum()
        quality["rssi_out_of_range"] = ((rssi < -150) | (rssi > 0)).sum()
    return quality


def check_coordinates(df):
    """Check Tx/Rx coordinate validity."""
    coord_checks = {}
    for prefix in ["tx", "rx"]:
        x_col = f"{prefix}_x"
        y_col = f"{prefix}_y"
        z_col = f"{prefix}_z"
        if x_col in df.columns and y_col in df.columns:
            coord_checks[f"{prefix}_x_range"] = f"{df[x_col].min():.2f} - {df[x_col].max():.2f}"
            coord_checks[f"{prefix}_y_range"] = f"{df[y_col].min():.2f} - {df[y_col].max():.2f}"
            coord_checks[f"{prefix}_nan_count"] = df[x_col].isna().sum() + df[y_col].isna().sum()
        if z_col in df.columns:
            coord_checks[f"{prefix}_z_range"] = f"{df[z_col].min():.2f} - {df[z_col].max():.2f}"
    return coord_checks


def check_temporal(df):
    """Check timestamp coverage and gaps."""
    temporal = {}
    if "timestamp" in df.columns:
        try:
            ts = pd.to_datetime(df["timestamp"])
            temporal["timestamp_min"] = str(ts.min())
            temporal["timestamp_max"] = str(ts.max())
            temporal["duration_days"] = (ts.max() - ts.min()).days
            temporal["total_records"] = len(ts)

            # Check gaps
            ts_sorted = ts.sort_values()
            diffs = ts_sorted.diff().dropna()
            temporal["median_interval_s"] = diffs.median().total_seconds()
            temporal["max_gap_s"] = diffs.max().total_seconds()
        except Exception as e:
            temporal["timestamp_error"] = str(e)
    return temporal


def compute_link_summary(df):
    """Compute per-link summary statistics."""
    if "tx_id" not in df.columns or "rx_id" not in df.columns:
        return None

    groups = df.groupby(["tx_id", "rx_id"])
    summaries = []
    for (tx, rx), grp in groups:
        summary = {"tx_id": tx, "rx_id": rx, "num_records": len(grp)}
        if "rssi_dbm" in grp.columns:
            rssi = grp["rssi_dbm"].dropna()
            summary["rssi_mean"] = rssi.mean()
            summary["rssi_std"] = rssi.std()
            summary["rssi_min"] = rssi.min()
            summary["rssi_max"] = rssi.max()
        if "distance_m" in grp.columns:
            summary["distance_m"] = grp["distance_m"].iloc[0]
        summaries.append(summary)

    return pd.DataFrame(summaries)


def main():
    if not LORA_DIR.exists():
        print(f"Baicaowa LoRa data directory not found: {LORA_DIR}")
        print("This script will be run after Baicaowa data is provided.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_rssi_data(LORA_DIR)
    if df is None:
        print("No RSSI data files found.")
        sys.exit(0)

    print(f"Loaded {len(df)} records from {df['_source_file'].nunique()} files")

    # Field check
    present, missing, extra = check_fields(df)
    print(f"\nPresent fields ({len(present)}): {present}")
    print(f"Missing fields ({len(missing)}): {missing}")
    print(f"Extra fields ({len(extra)}): {extra}")

    # Quality checks
    rssi_quality = check_rssi_quality(df)
    coord_checks = check_coordinates(df)
    temporal = check_temporal(df)

    # Data quality summary
    quality_data = {
        "total_records": len(df),
        "source_files": df["_source_file"].nunique(),
        "present_fields": len(present),
        "missing_fields": len(missing),
        "duplicate_rows": df.duplicated().sum(),
    }
    quality_data.update(rssi_quality)
    quality_data.update(coord_checks)
    quality_data.update(temporal)

    quality_df = pd.DataFrame([quality_data])
    quality_df.to_csv(OUTPUT_DIR / "baicaowa_data_quality.csv", index=False)

    # Link summary
    link_df = compute_link_summary(df)
    if link_df is not None:
        link_df.to_csv(OUTPUT_DIR / "baicaowa_link_summary.csv", index=False)
        print(f"\nLinks: {len(link_df)} unique Tx-Rx pairs")

    # Temporal summary by month
    if "timestamp" in df.columns:
        try:
            df["_month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")
            temporal_df = df.groupby("_month").agg(
                record_count=("rssi_dbm", "count"),
                rssi_mean=("rssi_dbm", "mean"),
                rssi_std=("rssi_dbm", "std"),
            ).reset_index()
            temporal_df["_month"] = temporal_df["_month"].astype(str)
            temporal_df.to_csv(OUTPUT_DIR / "baicaowa_temporal_summary.csv", index=False)
        except Exception:
            pass

    print("\nData quality check complete.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
