from pathlib import Path
import os
import logging
import pandas as pd
from fredapi import Fred

SERIES_MAP = {
    "US IG OAS": "BAMLC0A0CM",
    "US HY OAS": "BAMLH0A0HYM2",
    "US BBB OAS": "BAMLC0A4CBBBEY",
}

SEED_PATH = Path("./seed-data/ice_bofa_history.csv")
STORE_DIR = Path(os.getenv("ICE_STORE_DIR", "./cache-directory"))
STORE_PATH = STORE_DIR / "ice_bofa_history.parquet"

def _load_seed() -> pd.DataFrame:
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_PATH}")
    df = pd.read_csv(SEED_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")

def _load_store() -> pd.DataFrame:
    if STORE_PATH.exists():
        df = pd.read_parquet(STORE_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        return df.sort_values("Date")
    return _load_seed()

def _save_store(df: pd.DataFrame) -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    df = df.sort_values("Date")
    df.to_parquet(STORE_PATH, index=False)

def refresh_ice_bofa_store(fred_api_key: str) -> pd.DataFrame:
    df = _load_store()
    fred = Fred(api_key=fred_api_key)

    overlap_start = max(pd.Timestamp("2023-04-24"), df["Date"].max() - pd.Timedelta(days=45))
    updates = []

    for col, fred_id in SERIES_MAP.items():
        try:
            s = fred.get_series(fred_id)
            s.index = pd.to_datetime(s.index)
            s = s[s.index >= overlap_start].dropna()
            if not s.empty:
                updates.append(s.rename(col))
        except Exception as e:
            logging.warning(f"Could not refresh {fred_id}: {e}")

    if updates:
        upd = pd.concat(updates, axis=1).reset_index().rename(columns={"index": "Date"})
        merged = pd.concat([df, upd], ignore_index=True)
        merged = merged.sort_values("Date")
        merged = merged.drop_duplicates(subset=["Date"], keep="last")
        _save_store(merged)
        return merged

    _save_store(df)
    return df

def get_ice_bofa_series(fred_api_key: str, start_date=None) -> dict:
    df = refresh_ice_bofa_store(fred_api_key)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if start_date is not None:
        df = df[df.index >= pd.to_datetime(start_date)]

    return {
        "US IG OAS": df["US IG OAS"].rename("US IG OAS"),
        "US HY OAS": df["US HY OAS"].rename("US HY OAS"),
        "US BBB OAS": df["US BBB OAS"].rename("US BBB OAS"),
    }