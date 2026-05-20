"""
eda.py
------
Exploratory Data Analysis for the raw energy dataset.
Produces a set of publication-quality plots saved to outputs/eda/

Run:  python src/data/eda.py
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "energy_raw.parquet"
OUT_DIR  = ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = {"solar": "#F4A300", "wind": "#2196F3", "total": "#4CAF50", "anomaly": "#E53935"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def save_fig(name: str, fig=None, tight=True):
    if fig is None:
        fig = plt.gcf()
    if tight:
        fig.tight_layout()
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path.name}")


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_overview(df: pd.DataFrame):
    """Time-series overview: 30-day window to keep it readable."""
    sample = df[df["timestamp"] < df["timestamp"].iloc[0] + pd.Timedelta(days=30)]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("30-Day Overview — Solar, Wind & Total Power", fontsize=14, fontweight="bold")

    for ax, col, color, label in zip(
        axes,
        ["solar_power_kw", "wind_power_kw", "total_power_kw"],
        [PALETTE["solar"], PALETTE["wind"], PALETTE["total"]],
        ["Solar Power (kW)", "Wind Power (kW)", "Total Power (kW)"]
    ):
        ax.plot(sample["timestamp"], sample[col], color=color, lw=0.8, alpha=0.9)
        # Mark anomalies
        anom = sample[sample["anomaly"] == 1]
        ax.scatter(anom["timestamp"], anom[col], color=PALETTE["anomaly"],
                   s=20, zorder=5, label="Anomaly", alpha=0.8)
        ax.set_ylabel(label, fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.autofmt_xdate()
    save_fig("01_overview_30d", fig)


def plot_seasonal_profiles(df: pd.DataFrame):
    """Mean hourly profile per season for solar and wind."""
    df2 = df.copy()
    df2["hour"]   = df2["timestamp"].dt.hour
    df2["season"] = df2["timestamp"].dt.month.map(
        {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3})
    season_names  = {0:"Winter",1:"Spring",2:"Summer",3:"Autumn"}
    season_colors = {0:"#42A5F5",1:"#66BB6A",2:"#FFA726",3:"#AB47BC"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Mean Hourly Power Profile by Season", fontsize=13, fontweight="bold")

    for s in range(4):
        sub = df2[df2["season"] == s]
        solar_mean = sub.groupby("hour")["solar_power_kw"].mean()
        wind_mean  = sub.groupby("hour")["wind_power_kw"].mean()
        ax1.plot(solar_mean.index, solar_mean.values,
                 label=season_names[s], color=season_colors[s], lw=2)
        ax2.plot(wind_mean.index, wind_mean.values,
                 label=season_names[s], color=season_colors[s], lw=2)

    for ax, title in zip([ax1, ax2], ["Solar Power (kW)", "Wind Power (kW)"]):
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(range(0, 24, 3))

    save_fig("02_seasonal_profiles", fig)


def plot_distributions(df: pd.DataFrame):
    """KDE + histogram for the three power columns, split by anomaly flag."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Power Output Distributions (Normal vs Anomaly)", fontsize=13, fontweight="bold")

    for ax, col, color, label in zip(
        axes,
        ["solar_power_kw", "wind_power_kw", "total_power_kw"],
        [PALETTE["solar"], PALETTE["wind"], PALETTE["total"]],
        ["Solar (kW)", "Wind (kW)", "Total (kW)"]
    ):
        normal = df[df["anomaly"] == 0][col]
        anomal = df[df["anomaly"] == 1][col]
        ax.hist(normal, bins=60, density=True, alpha=0.6, color=color, label="Normal")
        ax.hist(anomal, bins=40, density=True, alpha=0.5, color=PALETTE["anomaly"], label="Anomaly")
        ax.set_title(label)
        ax.set_xlabel("kW")
        ax.legend(fontsize=9)

    save_fig("03_distributions", fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    """Pearson correlation matrix for numeric features."""
    num_cols = ["ghi_w_m2", "ambient_temp_c", "wind_speed_m_s",
                "solar_power_kw", "wind_power_kw", "total_power_kw"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, square=True, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    save_fig("04_correlation_heatmap", fig)


def plot_anomaly_breakdown(df: pd.DataFrame):
    """Anomaly type counts + time distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Anomaly Analysis", fontsize=13, fontweight="bold")

    # Count by type
    counts = df[df["anomaly"] == 1]["anomaly_type"].value_counts()
    bars = ax1.bar(counts.index, counts.values,
                   color=[PALETTE["anomaly"], "#FF7043", "#FFA726", "#AB47BC"])
    ax1.bar_label(bars, padding=3, fontsize=10)
    ax1.set_title("Anomaly Count by Type")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Anomaly Type")

    # Anomalies per month
    anom = df[df["anomaly"] == 1].copy()
    anom["month"] = anom["timestamp"].dt.to_period("M").astype(str)
    monthly = anom.groupby("month").size()
    ax2.bar(range(len(monthly)), monthly.values, color=PALETTE["anomaly"], alpha=0.8)
    ax2.set_xticks(range(0, len(monthly), 2))
    ax2.set_xticklabels(monthly.index[::2], rotation=45, ha="right", fontsize=8)
    ax2.set_title("Monthly Anomaly Count")
    ax2.set_ylabel("Count")

    save_fig("05_anomaly_breakdown", fig)


def plot_monthly_total_energy(df: pd.DataFrame):
    """Total energy generation per month (MWh)."""
    df2 = df.copy()
    df2["month"]        = df2["timestamp"].dt.to_period("M")
    df2["solar_mwh"]    = df2["solar_power_kw"] / 1000   # kW·h → MWh
    df2["wind_mwh"]     = df2["wind_power_kw"]  / 1000
    monthly = df2.groupby("month")[["solar_mwh", "wind_mwh"]].sum()

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(monthly))
    w = 0.4
    ax.bar(x - w/2, monthly["solar_mwh"], width=w, label="Solar", color=PALETTE["solar"], alpha=0.85)
    ax.bar(x + w/2, monthly["wind_mwh"],  width=w, label="Wind",  color=PALETTE["wind"],  alpha=0.85)
    ax.set_xticks(x[::2])
    ax.set_xticklabels([str(p) for p in monthly.index[::2]], rotation=45, ha="right", fontsize=8)
    ax.set_title("Monthly Energy Generation (MWh)", fontsize=13, fontweight="bold")
    ax.set_ylabel("MWh")
    ax.legend()
    save_fig("06_monthly_energy", fig)


def generate_summary_stats(df: pd.DataFrame) -> dict:
    stats = {
        "total_rows":         len(df),
        "date_range":         f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}",
        "anomaly_count":      int(df["anomaly"].sum()),
        "anomaly_rate_pct":   round(df["anomaly"].mean() * 100, 2),
        "anomaly_by_type":    df[df["anomaly"]==1]["anomaly_type"].value_counts().to_dict(),
        "power_stats": {
            col: {
                "mean":   round(float(df[col].mean()), 2),
                "std":    round(float(df[col].std()), 2),
                "min":    round(float(df[col].min()), 2),
                "max":    round(float(df[col].max()), 2),
                "median": round(float(df[col].median()), 2),
            }
            for col in ["solar_power_kw", "wind_power_kw", "total_power_kw"]
        }
    }
    path = OUT_DIR / "summary_stats.json"
    path.write_text(json.dumps(stats, indent=2))
    print(f"  [✓] summary_stats.json")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def run_eda():
    print("Running EDA...")
    df = load()

    print("Generating plots:")
    plot_overview(df)
    plot_seasonal_profiles(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_anomaly_breakdown(df)
    plot_monthly_total_energy(df)

    stats = generate_summary_stats(df)
    print(f"\nDataset summary:")
    print(f"  Rows:          {stats['total_rows']:,}")
    print(f"  Date range:    {stats['date_range']}")
    print(f"  Anomalies:     {stats['anomaly_count']} ({stats['anomaly_rate_pct']}%)")
    print(f"\nAll plots saved to: {OUT_DIR}")


if __name__ == "__main__":
    run_eda()
