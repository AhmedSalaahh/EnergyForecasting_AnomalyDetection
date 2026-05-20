"""
generate_dataset.py
-------------------
Generates a realistic synthetic solar + wind power dataset across:
  - 10 geographically diverse locations (US, EU, AU, Middle East, Asia)
  - 6 years of hourly data (2018-2023)
  → ~525,600 rows total

Each location has unique climate parameters:
  - Solar capacity, latitude-adjusted irradiance, cloud cover tendency
  - Wind regime (Weibull shape/scale), hub height, seasonal pattern
  - Local ambient temperature profile

Anomaly types injected per location:
  1. Spike       - sudden extreme value (sensor malfunction)
  2. Dropout     - reading drops to 0 unexpectedly
  3. Stuck       - sensor freezes on one value for N hours
  4. Noise burst - high-frequency noise for a window
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

# ── Reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(42)

# ── Config ───────────────────────────────────────────────────────────────────
START      = "2018-01-01"
END        = "2023-12-31 23:00:00"
FREQ       = "h"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Location profiles ────────────────────────────────────────────────────────
# Each dict defines climate character for one site
LOCATIONS = [
    {
        "id": "US_AZ_Phoenix",
        "lat": 33.4,
        "solar_capacity_kw": 200,
        "wind_capacity_kw":  80,
        "ghi_peak": 1050,       # high sun, desert
        "cloud_alpha": 1.5,     # less cloud (lower alpha → clearer)
        "temp_base": 24, "temp_amp": 12,
        "wind_scale": 5.5, "wind_shape": 2.1,
        "sunrise_h": 6, "sunset_h": 19,
    },
    {
        "id": "US_TX_Abilene",
        "lat": 32.4,
        "solar_capacity_kw": 150,
        "wind_capacity_kw":  250,
        "ghi_peak": 950,
        "cloud_alpha": 2.0,
        "temp_base": 18, "temp_amp": 11,
        "wind_scale": 9.5, "wind_shape": 2.3,  # windy plains
        "sunrise_h": 7, "sunset_h": 20,
    },
    {
        "id": "DE_Bavaria",
        "lat": 48.1,
        "solar_capacity_kw": 120,
        "wind_capacity_kw":  180,
        "ghi_peak": 780,
        "cloud_alpha": 3.5,     # cloudier
        "temp_base": 10, "temp_amp": 10,
        "wind_scale": 7.0, "wind_shape": 1.9,
        "sunrise_h": 7, "sunset_h": 18,
    },
    {
        "id": "ES_Andalusia",
        "lat": 37.4,
        "solar_capacity_kw": 300,
        "wind_capacity_kw":  200,
        "ghi_peak": 1020,
        "cloud_alpha": 1.8,
        "temp_base": 19, "temp_amp": 10,
        "wind_scale": 8.0, "wind_shape": 2.2,
        "sunrise_h": 7, "sunset_h": 20,
    },
    {
        "id": "AU_Queensland",
        "lat": -27.5,           # southern hemisphere → inverted seasons
        "solar_capacity_kw": 250,
        "wind_capacity_kw":  120,
        "ghi_peak": 1000,
        "cloud_alpha": 1.7,
        "temp_base": 22, "temp_amp": 8,
        "wind_scale": 6.5, "wind_shape": 2.0,
        "sunrise_h": 6, "sunset_h": 18,
    },
    {
        "id": "SA_Riyadh",
        "lat": 24.7,
        "solar_capacity_kw": 400,
        "wind_capacity_kw":  60,
        "ghi_peak": 1080,
        "cloud_alpha": 1.2,     # very clear
        "temp_base": 28, "temp_amp": 14,
        "wind_scale": 5.0, "wind_shape": 2.5,
        "sunrise_h": 6, "sunset_h": 18,
    },
    {
        "id": "IN_Rajasthan",
        "lat": 27.0,
        "solar_capacity_kw": 350,
        "wind_capacity_kw":  160,
        "ghi_peak": 1000,
        "cloud_alpha": 2.2,     # monsoon seasons
        "temp_base": 27, "temp_amp": 12,
        "wind_scale": 7.5, "wind_shape": 2.0,
        "sunrise_h": 6, "sunset_h": 19,
    },
    {
        "id": "CN_Xinjiang",
        "lat": 42.0,
        "solar_capacity_kw": 200,
        "wind_capacity_kw":  300,
        "ghi_peak": 900,
        "cloud_alpha": 2.0,
        "temp_base": 8, "temp_amp": 18,  # continental extremes
        "wind_scale": 10.0, "wind_shape": 2.4,
        "sunrise_h": 7, "sunset_h": 19,
    },
    {
        "id": "UK_Scotland",
        "lat": 57.0,
        "solar_capacity_kw": 80,
        "wind_capacity_kw":  350,         # dominant wind site
        "ghi_peak": 650,
        "cloud_alpha": 4.0,               # very cloudy
        "temp_base": 8, "temp_amp": 6,
        "wind_scale": 11.5, "wind_shape": 1.8,
        "sunrise_h": 8, "sunset_h": 17,
    },
    {
        "id": "BR_CearA",
        "lat": -3.7,                      # near equator, Brazil
        "solar_capacity_kw": 220,
        "wind_capacity_kw":  200,
        "ghi_peak": 980,
        "cloud_alpha": 2.5,
        "temp_base": 28, "temp_amp": 4,   # low seasonal variation
        "wind_scale": 8.5, "wind_shape": 3.0,
        "sunrise_h": 6, "sunset_h": 18,
    },
]


# ── Physics helpers (location-aware) ─────────────────────────────────────────

def solar_irradiance(ts: pd.DatetimeIndex, loc: dict) -> np.ndarray:
    """Clear-sky GHI proxy (W/m²) with location-specific seasonal + diurnal."""
    doy  = ts.dayofyear
    hour = ts.hour
    lat  = loc["lat"]

    # Southern hemisphere: invert seasonal phase
    phase = 172 if lat >= 0 else (172 + 183) % 365
    season  = 0.5 + 0.5 * np.cos(2 * np.pi * (doy - phase) / 365)

    sr, ss = loc["sunrise_h"], loc["sunset_h"]
    day_len = ss - sr
    diurnal = np.maximum(0, np.sin(np.pi * (hour - sr) / day_len))

    ghi = loc["ghi_peak"] * season * diurnal

    # Cloud cover — location-tuned beta distribution
    a = loc["cloud_alpha"]
    cloud = RNG.beta(a, 2, size=len(ts))
    ghi  *= (1 - 0.65 * cloud)
    return np.clip(ghi, 0, loc["ghi_peak"])


def solar_power(ghi: np.ndarray, temp_c: np.ndarray, loc: dict) -> np.ndarray:
    """PV output (kW). Efficiency drops ~0.4 %/°C above 25 °C."""
    cap      = loc["solar_capacity_kw"]
    eta_temp = 1 - 0.004 * np.maximum(0, temp_c - 25)
    pr       = 0.80
    return np.clip(cap * (ghi / 1000) * pr * eta_temp, 0, cap)


def ambient_temp(ts: pd.DatetimeIndex, loc: dict) -> np.ndarray:
    """Ambient temperature (°C) with location seasonal + diurnal variation."""
    doy  = ts.dayofyear
    hour = ts.hour
    lat  = loc["lat"]

    phase   = 80 if lat >= 0 else (80 + 183) % 365
    base    = loc["temp_base"] + loc["temp_amp"] * np.sin(2 * np.pi * (doy - phase) / 365)
    diurnal = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
    noise   = RNG.normal(0, 1.5, size=len(ts))
    return base + diurnal + noise


def wind_speed(ts: pd.DatetimeIndex, loc: dict) -> np.ndarray:
    """Wind speed at hub height (m/s) — location-tuned Weibull + diurnal."""
    doy  = ts.dayofyear
    hour = ts.hour
    lat  = loc["lat"]

    phase    = 15 if lat >= 0 else (15 + 183) % 365
    seasonal = 1 + 0.3  * np.cos(2 * np.pi * (doy - phase) / 365)
    diurnal  = 1 + 0.15 * np.sin(2 * np.pi * (hour - 14) / 24)
    scale    = loc["wind_scale"] * seasonal * diurnal
    raw      = RNG.weibull(loc["wind_shape"], size=len(ts)) * scale
    return np.clip(raw, 0, 35)


def wind_power(ws: np.ndarray, loc: dict) -> np.ndarray:
    """Turbine power curve: cut-in=3, rated=12, cut-out=25 m/s."""
    cap = loc["wind_capacity_kw"]
    out = np.zeros_like(ws)
    ci, cr, co = 3.0, 12.0, 25.0
    mask_ramp = (ws >= ci) & (ws < cr)
    mask_full = (ws >= cr) & (ws <= co)
    out[mask_ramp] = cap * ((ws[mask_ramp] - ci) / (cr - ci)) ** 3
    out[mask_full] = cap
    return out


# ── Anomaly injection ─────────────────────────────────────────────────────────

def inject_anomalies(df: pd.DataFrame,
                     anomaly_rate: float = 0.02) -> pd.DataFrame:
    """
    Randomly inject 4 anomaly types into solar_power and wind_power columns.
    Returns a copy with an `anomaly` flag column and `anomaly_type` label.
    Uses vectorised operations to keep speed acceptable at 500k+ rows.
    """
    df = df.copy()
    df["anomaly"]      = 0
    df["anomaly_type"] = "none"

    n        = len(df)
    n_events = max(1, int(n * anomaly_rate))
    cols     = ["solar_power_kw", "wind_power_kw"]
    loc_rng  = np.random.default_rng(RNG.integers(0, 2**32))

    type_counts = {
        "spike":       int(n_events * 0.30),
        "dropout":     int(n_events * 0.30),
        "stuck":       int(n_events * 0.25),
        "noise_burst": int(n_events * 0.15),
    }

    solar_max = df["solar_power_kw"].quantile(0.95)
    wind_max  = df["wind_power_kw"].quantile(0.95)
    col_max   = {"solar_power_kw": solar_max, "wind_power_kw": wind_max}

    for atype, count in type_counts.items():
        idxs = loc_rng.integers(0, n, size=count)
        col_choices = loc_rng.choice(cols, size=count)

        for idx, col in zip(idxs, col_choices):
            if atype == "spike":
                df.iat[idx, df.columns.get_loc(col)] = col_max[col] * loc_rng.uniform(2.5, 5.0)
                df.iat[idx, df.columns.get_loc("anomaly")]      = 1
                df.iat[idx, df.columns.get_loc("anomaly_type")] = "spike"

            elif atype == "dropout":
                window = min(int(loc_rng.integers(1, 6)), n - idx)
                end    = idx + window
                df.iloc[idx:end, df.columns.get_loc(col)]          = 0.0
                df.iloc[idx:end, df.columns.get_loc("anomaly")]     = 1
                df.iloc[idx:end, df.columns.get_loc("anomaly_type")] = "dropout"

            elif atype == "stuck":
                stuck_val = df.iat[idx, df.columns.get_loc(col)]
                window    = min(int(loc_rng.integers(3, 12)), n - idx)
                end       = idx + window
                df.iloc[idx:end, df.columns.get_loc(col)]          = stuck_val
                df.iloc[idx:end, df.columns.get_loc("anomaly")]     = 1
                df.iloc[idx:end, df.columns.get_loc("anomaly_type")] = "stuck"

            elif atype == "noise_burst":
                window   = min(int(loc_rng.integers(2, 8)), n - idx)
                end      = idx + window
                factors  = loc_rng.uniform(0.1, 3.0, size=window)
                df.iloc[idx:end, df.columns.get_loc(col)] *= factors
                df.iloc[idx:end, df.columns.get_loc("anomaly")]     = 1
                df.iloc[idx:end, df.columns.get_loc("anomaly_type")] = "noise_burst"

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_location(loc: dict, ts: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate one location's full time-series."""
    ghi  = solar_irradiance(ts, loc)
    temp = ambient_temp(ts, loc)
    ws   = wind_speed(ts, loc)

    df = pd.DataFrame({
        "timestamp":         ts,
        "location_id":       loc["id"],
        "latitude":          loc["lat"],
        "solar_capacity_kw": loc["solar_capacity_kw"],
        "wind_capacity_kw":  loc["wind_capacity_kw"],
        "ghi_w_m2":          ghi.round(2),
        "ambient_temp_c":    temp.round(2),
        "wind_speed_m_s":    ws.round(2),
        "solar_power_kw":    solar_power(ghi, temp, loc).round(2),
        "wind_power_kw":     wind_power(ws, loc).round(2),
    })
    df["total_power_kw"] = (df["solar_power_kw"] + df["wind_power_kw"]).round(2)
    df = inject_anomalies(df, anomaly_rate=0.025)
    return df


def generate(save: bool = True) -> pd.DataFrame:
    ts = pd.date_range(START, END, freq=FREQ)
    print(f"Generating {len(LOCATIONS)} locations × {len(ts):,} hours = "
          f"{len(LOCATIONS)*len(ts):,} rows expected...")

    frames = []
    for loc in LOCATIONS:
        loc_rng = np.random.default_rng(abs(hash(loc["id"])) % (2**32))
        # Give each location its own RNG seed for reproducibility
        df = generate_location(loc, ts)
        frames.append(df)
        print(f"  [✓] {loc['id']:25s}  {len(df):,} rows  |  "
              f"anomalies: {df['anomaly'].sum()} ({df['anomaly'].mean()*100:.1f}%)")

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    if save:
        path = OUTPUT_DIR / "energy_raw.parquet"
        full.to_parquet(path, index=False)
        print(f"\n[✓] Saved {len(full):,} rows → {path}")

        meta = {
            "total_rows":        len(full),
            "locations":         [l["id"] for l in LOCATIONS],
            "n_locations":       len(LOCATIONS),
            "start":             str(ts[0]),
            "end":               str(ts[-1]),
            "freq":              FREQ,
            "hours_per_location": len(ts),
            "anomaly_count":     int(full["anomaly"].sum()),
            "anomaly_rate_pct":  round(full["anomaly"].mean() * 100, 2),
            "columns":           list(full.columns),
        }
        (OUTPUT_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"[✓] Total anomalies: {meta['anomaly_count']:,} ({meta['anomaly_rate_pct']}%)")
        print(f"[✓] Columns: {meta['columns']}")

    return full


if __name__ == "__main__":
    df = generate()
    print(f"\nFinal shape: {df.shape}")
    print(df.groupby("location_id")[["solar_power_kw","wind_power_kw","anomaly"]].mean().round(3))
