"""
preprocess.py
-------------
Preprocessing pipeline for the raw energy dataset.

Steps:
  1. Load raw parquet
  2. Parse & validate timestamps
  3. Handle missing values (forward-fill gaps ≤ 3h, else interpolate)
  4. Feature engineering:
       - Time features (hour, day_of_week, month, is_weekend, season)
       - Lag features  (t-1, t-2, t-3, t-24, t-48)
       - Rolling stats (mean/std over 3h, 6h, 24h windows)
       - Day/night flag
  5. Train/val/test split (70/15/15 — no shuffle, respects time order)
  6. StandardScaler fit on train only → saved for inference
  7. Save processed splits + scaler
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
RAW_PATH  = ROOT / "data" / "raw"  / "energy_raw.parquet"
PROC_PATH = ROOT / "data" / "processed"
PROC_PATH.mkdir(parents=True, exist_ok=True)

# ── Feature groups ────────────────────────────────────────────────────────────
TARGET_COLS = ["solar_power_kw", "wind_power_kw", "total_power_kw"]

LAG_COLS    = ["solar_power_kw", "wind_power_kw", "total_power_kw",
               "ghi_w_m2", "wind_speed_m_s"]

SCALE_COLS  = ["ghi_w_m2", "ambient_temp_c", "wind_speed_m_s",
               "solar_power_kw", "wind_power_kw", "total_power_kw",
               "solar_capacity_kw", "wind_capacity_kw"]


# ── Step helpers ──────────────────────────────────────────────────────────────

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    # Encode location as integer category for model use
    df["location_code"] = df["location_id"].astype("category").cat.codes
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)
    print(f"[✓] Loaded {len(df):,} rows | {df['location_id'].nunique()} locations | "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Per-location gap filling — forward-fill ≤ 3h, else interpolate."""
    filled_frames = []
    for loc_id, grp in df.groupby("location_id"):
        grp = grp.set_index("timestamp").asfreq("h")
        n_missing = grp[TARGET_COLS].isna().sum().sum()
        if n_missing:
            grp[TARGET_COLS] = grp[TARGET_COLS].ffill(limit=3)
            grp[TARGET_COLS] = grp[TARGET_COLS].interpolate(method="time", limit_direction="both")
        filled_frames.append(grp.reset_index())
    df_out = pd.concat(filled_frames, ignore_index=True)
    return df_out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"]        = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek           # 0=Mon … 6=Sun
    df["month"]       = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"]  = (ts.dt.dayofweek >= 5).astype(int)

    # Season (meteorological)
    df["season"] = ts.dt.month.map(
        {12: 0, 1: 0, 2: 0,   # Winter
         3:  1, 4: 1, 5: 1,   # Spring
         6:  2, 7: 2, 8: 2,   # Summer
         9:  3, 10:3, 11:3}   # Autumn
    )

    # Cyclical encoding for hour & month
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]    = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Day/Night flag (solar hour 6–18)
    df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] < 18)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute lag features within each location to avoid cross-location leakage."""
    lag_frames = []
    for _, grp in df.groupby("location_id"):
        grp = grp.copy()
        for col in LAG_COLS:
            for lag in [1, 2, 3, 24, 48]:
                grp[f"{col}_lag{lag}"] = grp[col].shift(lag)
        lag_frames.append(grp)
    return pd.concat(lag_frames, ignore_index=True)


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling stats within each location."""
    roll_frames = []
    for _, grp in df.groupby("location_id"):
        grp = grp.copy()
        for col in ["solar_power_kw", "wind_power_kw", "total_power_kw"]:
            for window in [3, 6, 24]:
                roll = grp[col].rolling(window, min_periods=1)
                grp[f"{col}_roll{window}h_mean"] = roll.mean().round(4)
                grp[f"{col}_roll{window}h_std"]  = roll.std().round(4).fillna(0)
        roll_frames.append(grp)
    return pd.concat(roll_frames, ignore_index=True)


def split_data(df: pd.DataFrame, train=0.70, val=0.15):
    n      = len(df)
    i_val  = int(n * train)
    i_test = int(n * (train + val))
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


def scale_and_save(train: pd.DataFrame,
                   val: pd.DataFrame,
                   test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    # Fit ONLY on train
    avail = [c for c in SCALE_COLS if c in train.columns]
    train[avail] = scaler.fit_transform(train[avail])
    val[avail]   = scaler.transform(val[avail])
    test[avail]  = scaler.transform(test[avail])

    scaler_path = PROC_PATH / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "cols": avail}, f)
    print(f"[✓] Scaler saved → {scaler_path}")
    return train, val, test


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(raw_path: Path = RAW_PATH) -> dict[str, pd.DataFrame]:
    df = load_raw(raw_path)
    df = handle_missing(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # Drop rows with NaNs introduced by lagging (first 48 rows)
    df = df.dropna().reset_index(drop=True)
    print(f"[✓] After feature engineering: {len(df):,} rows, {df.shape[1]} columns")

    train, val, test = split_data(df)
    train, val, test = scale_and_save(train, val, test)

    splits = {"train": train, "val": val, "test": test}
    for name, split in splits.items():
        out = PROC_PATH / f"{name}.parquet"
        split.to_parquet(out, index=False)
        print(f"[✓] {name:5s}: {len(split):,} rows → {out}")

    # Save feature list for model training
    feature_meta = {
        "target_cols":  TARGET_COLS,
        "scale_cols":   [c for c in SCALE_COLS if c in train.columns],
        "n_features":   train.shape[1],
        "feature_names": list(train.columns),
        "split_sizes": {k: len(v) for k, v in splits.items()},
    }
    (PROC_PATH / "feature_meta.json").write_text(json.dumps(feature_meta, indent=2))
    print(f"[✓] Feature meta saved ({len(feature_meta['feature_names'])} features)")

    return splits


if __name__ == "__main__":
    splits = run_pipeline()
    print("\nTrain sample:")
    print(splits["train"][["timestamp", "solar_power_kw", "wind_power_kw",
                             "total_power_kw", "hour_sin", "anomaly"]].head(10))
