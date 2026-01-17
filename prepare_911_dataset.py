#!/usr/bin/env python3
"""
prepare_911_dataset.py

Build a leak-proof ML dataset for the "911 Alt-Response Triage" project.

Inputs
------
1) Seattle Real-Time Fire 911 calls (CSV or XLSX) exported from Socrata.
   Required columns (case-insensitive): Incident Number, Datetime, Type, Latitude, Longitude, Address

2) (Optional) Units-by-day CSV produced by scraping the SFD daily Real-Time 911 pages.
   Required columns: incident_id, level, units_raw

Outputs
-------
- care_dataset.csv / care_dataset.parquet : feature matrix with label
- care_meta.json : simple data dictionary
- quick_summary.txt : label distribution & basic counts

Usage
-----
python prepare_911_dataset.py --calls path/to/calls.csv --outdir out_dir \
    [--units path/to/units_by_day.csv] [--label_scheme type|units]

Default label scheme is "type".
For "units", you must provide --units.

Notes
-----
- If label_scheme == "type": we derive the label from the 'Type' text.
  To avoid leakage, the script will automatically drop 'Type' and any Type-derived features.
- If label_scheme == "units": label is derived from (level, units_raw). We keep all other features.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------

CALLS_REQUIRED = ["Incident Number", "Datetime", "Type", "Latitude", "Longitude", "Address"]

def _find_col(df: pd.DataFrame, name: str) -> Optional[str]:
    target = name.strip().lower()
    for c in df.columns:
        if c.strip().lower() == target:
            return c
    return None

def load_calls(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # validate required columns (case-insensitive)
    missing = [c for c in CALLS_REQUIRED if _find_col(df, c) is None]
    if missing:
        raise ValueError(f"Missing required columns in calls file: {missing}")
    # standardize names
    rename_map = {}
    for std in CALLS_REQUIRED:
        rename_map[_find_col(df, std)] = std
    df = df.rename(columns=rename_map)

    # normalize and select
    df = df[CALLS_REQUIRED].copy()
    df["Incident Number"] = df["Incident Number"].astype(str).str.strip().str.upper()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df["Type"] = df["Type"].astype(str).str.strip()

    # numeric coords
    for col in ["Latitude", "Longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_units(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # expected: incident_id, level, units_raw
    cols = {c.lower(): c for c in df.columns}
    # permissive mapping
    i_col = None
    for cand in ["incident_id", "incident", "incidentnumber", "incident_number"]:
        if cand in cols:
            i_col = cols[cand]; break
    if i_col is None:
        raise ValueError("Units file must include an incident_id column.")
    lvl_col = None
    for cand in ["level", "alarmlevel", "alarm_level"]:
        if cand in cols:
            lvl_col = cols[cand]; break
    units_col = None
    for cand in ["units_raw", "units", "apparatus"]:
        if cand in cols:
            units_col = cols[cand]; break
    if units_col is None:
        raise ValueError("Units file must include a units_raw column.")

    out = pd.DataFrame({
        "incident_id": df[i_col].astype(str).str.strip().str.upper(),
        "level": pd.to_numeric(df[lvl_col], errors="coerce") if lvl_col else pd.Series([np.nan]*len(df)),
        "units_raw": df[units_col].astype(str).fillna("")
    })
    return out.drop_duplicates(subset=["incident_id"])


def engineer_time_space(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["Datetime"].dt.year
    out["month"] = out["Datetime"].dt.month
    out["dow"] = out["Datetime"].dt.dayofweek   # 0=Mon
    out["hour"] = out["Datetime"].dt.hour
    out["is_weekend"] = out["dow"].isin([5, 6]).astype(int)

    # simple spatial bins (avoid external geohash deps)
    out["lat_bin"] = out["Latitude"].round(3)
    out["lon_bin"] = out["Longitude"].round(3)
    return out


def parse_units(df: pd.DataFrame) -> pd.DataFrame:
    """Derive counts/flags from units_raw."""
    def count(pattern, s):
        if not isinstance(s, str): return 0
        return len(re.findall(pattern, s, flags=re.I))

    d = df.copy()
    d["actual_engine"] = d["units_raw"].apply(lambda s: count(r"\bE\d+\b", s))
    d["actual_ladder"] = d["units_raw"].apply(lambda s: count(r"\bL\d+\b", s))
    d["actual_medic"]  = d["units_raw"].apply(lambda s: count(r"\bM\d+\b", s))
    d["actual_aid"]    = d["units_raw"].apply(lambda s: count(r"\bA\d+\b", s))
    d["has_specialty"] = d["units_raw"].str.contains(r"HAZ|DECON|RESCUE|FIREBOAT|REHAB|AIR|MCI|PATROL|MVU",
                                                     case=True, regex=True, na=False).astype(int)
    d["actual_total_units"] = d[["actual_engine","actual_ladder","actual_medic","actual_aid"]].sum(axis=1)
    return d


# Types considered "low-acuity-like" for TYPE-based label (you can adjust)
POSITIVE_TYPES = {
    "TRIAGED INCIDENT",
    "LOW ACUITY RESPONSE",
    "AID RESPONSE",
    "HANG-UP- AID",
    "INVESTIGATE OUT OF SERVICE"
}

def derive_label_type(df: pd.DataFrame) -> pd.Series:
    t = df["Type"].astype(str).str.upper().str.strip()
    is_pos = t.isin(POSITIVE_TYPES)
    # everything that is obviously high acuity is negative by default
    return is_pos.astype(int)


def derive_label_units(df: pd.DataFrame) -> pd.Series:
    """
    Example rule: CARE-eligible if
    - level <= 1 (or NaN treated as 1) AND
    - no Medic present AND
    - small footprint (<= 2 total units) AND
    - no specialty flags
    """
    lvl = df["level"].fillna(1)
    cond = (
        (lvl <= 1) &
        (df["actual_medic"] == 0) &
        (df["actual_total_units"] <= 2) &
        (df["has_specialty"] == 0)
    )
    return cond.astype(int)


def build_dataset(calls_path: Path,
                  outdir: Path,
                  units_path: Optional[Path] = None,
                  label_scheme: str = "type") -> Tuple[pd.DataFrame, Dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    calls = load_calls(calls_path)

    # time/space features first
    calls_feat = engineer_time_space(calls)

    # Optional: join units
    if units_path:
        units = load_units(units_path)
        calls_feat = calls_feat.merge(units, left_on="Incident Number", right_on="incident_id", how="left")
        calls_feat.drop(columns=["incident_id"], inplace=True, errors="ignore")
        # derive unit counts/flags
        calls_feat = parse_units(calls_feat)
    else:
        # create empty columns to keep schema consistent
        for col in ["level","units_raw","actual_engine","actual_ladder","actual_medic","actual_aid","has_specialty","actual_total_units"]:
            calls_feat[col] = np.nan if col in ["level"] else 0

    # Labels
    if label_scheme == "type":
        label = derive_label_type(calls_feat)
        # drop leaky columns tied to Type
        drop_cols = ["Type"]
    elif label_scheme == "units":
        if units_path is None:
            raise ValueError("label_scheme=units requires --units file.")
        label = derive_label_units(calls_feat)
        drop_cols = []  # we can keep Type as a feature for ablations if you want, but default to keep it for inspection
    else:
        raise ValueError("label_scheme must be 'type' or 'units'")

    calls_feat["label"] = label

    # Final feature selection (safe core)
    keep = [
        # identifiers
        "Incident Number", "Datetime", "Address",
        # coordinates & bins
        "Latitude", "Longitude", "lat_bin", "lon_bin",
        # time
        "year", "month", "dow", "hour", "is_weekend",
        # units-derived (safe even without units file; they are zeros)
        "level", "actual_engine", "actual_ladder", "actual_medic", "actual_aid",
        "has_specialty", "actual_total_units",
        # target
        "label"
    ]
    # include units_raw for inspection but not recommended as a direct ML input
    if "units_raw" in calls_feat.columns:
        keep.append("units_raw")

    # Drop leaky columns if type-based label
    calls_feat = calls_feat[keep].copy()

    # Save outputs
    out_csv = outdir / "care_dataset.csv"
    out_parq = outdir / "care_dataset.parquet"
    calls_feat.to_csv(out_csv, index=False)
    try:
        calls_feat.to_parquet(out_parq, index=False)
    except Exception as e:
        # parquet optional
        pass

    # Simple metadata
    meta = {
        "source_calls_file": str(calls_path),
        "source_units_file": str(units_path) if units_path else None,
        "rows": int(calls_feat.shape[0]),
        "label_scheme": label_scheme,
        "positive_definition": "Type ∈ {Triaged Incident, Low Acuity Response}" if label_scheme=="type" else
                               "level<=1 AND no medic AND <=2 total units AND no specialty",
        "columns": {c: str(calls_feat[c].dtype) for c in calls_feat.columns}
    }
    with open(outdir / "care_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Quick summary
    pos = int(calls_feat["label"].sum())
    neg = int(calls_feat.shape[0] - pos)
    with open(outdir / "quick_summary.txt", "w") as f:
        f.write(f"Rows: {calls_feat.shape[0]}\n")
        f.write(f"Positives: {pos} ({pos/calls_feat.shape[0]:.2%})\n")
        f.write(f"Negatives: {neg} ({neg/calls_feat.shape[0]:.2%})\n")
        f.write("Head:\n")
        f.write(calls_feat.head(10).to_string())

    return calls_feat, meta


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True, type=Path, help="Path to Seattle 911 calls CSV/XLSX")
    ap.add_argument("--units", required=False, type=Path, help="Path to units-by-day CSV (optional)")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory")
    ap.add_argument("--label_scheme", choices=["type","units"], default="type", help="How to derive labels")
    return ap.parse_args()


def main():
    args = parse_args()
    _df, _meta = build_dataset(args.calls, args.outdir, args.units, args.label_scheme)
    print("Saved dataset to:", args.outdir)
    print(json.dumps(_meta, indent=2))


if __name__ == "__main__":
    main()
