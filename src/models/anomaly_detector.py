"""
anomaly_detector.py
--------------------
Isolation Forest anomaly detector for the multi-location energy dataset.

Pipeline:
  1. Load processed train/val/test splits
  2. Build sensor feature matrix
  3. Train one global Isolation Forest (contamination auto-tuned)
  4. Tune contamination parameter on val set via F1 sweep
  5. Evaluate on test set: precision, recall, F1, confusion matrix
  6. Flag anomalies across full dataset and save cleaned version
  7. Persist model
  8. Produce evaluation plots

Key design decisions:
  - Features: raw sensor signals + rolling stats only (NOT power output targets)
  - One model trained globally then evaluated per-location
  - Contamination tuned on val set F1 (realistic for production use)
"""

import json
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, f1_score
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROC      = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "models"
OUT_DIR   = ROOT / "outputs" / "anomaly"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature matrix definition ─────────────────────────────────────────────────
# These are SENSOR features only — no power output targets.
# The IF must detect anomalies from sensor behaviour, not from knowing ground truth.
SENSOR_FEATURES = [
    # Raw sensor readings
    "ghi_w_m2", "ambient_temp_c", "wind_speed_m_s",
    # Raw power readings
    "solar_power_kw", "wind_power_kw",
    # Rate-of-change — spikes appear as huge deltas
    "solar_delta", "wind_delta", "ghi_delta",
    # Rolling std (6h) — stuck sensors → std collapses toward 0
    "solar_power_kw_rstd6", "wind_power_kw_rstd6", "ghi_w_m2_rstd6",
    # Rolling mean (6h) — context for interpreting current reading
    "solar_power_kw_rmean6", "wind_power_kw_rmean6",
    # Physics consistency ratios
    "ghi_solar_ratio",   # solar output per unit GHI — spikes break this
    "solar_cap_util",    # fraction of rated solar capacity being used
    "wind_cap_util",     # fraction of rated wind capacity being used
    # Stuck-value binary flags (delta ≈ 0 for 3+ consecutive hours)
    "solar_zero_delta", "wind_zero_delta",
    # Time context
    "hour", "month", "is_day",
    # Location context
    "location_code", "solar_capacity_kw", "wind_capacity_kw",
]


# ── Load data ─────────────────────────────────────────────────────────────────

def add_if_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features for the Isolation Forest.
    Works on RAW (unscaled) data so anomaly magnitudes are preserved.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["location_id", "timestamp"]).reset_index(drop=True)

    # Rate of change — spikes show as huge deltas
    for col in ["solar_power_kw", "wind_power_kw", "ghi_w_m2"]:
        delta_name = col.split("_")[0] + "_delta"
        df[delta_name] = df.groupby("location_id")[col].diff().abs().fillna(0)

    # Rolling std (6h window) — stuck sensors collapse to 0
    for col in ["solar_power_kw", "wind_power_kw", "ghi_w_m2"]:
        df[f"{col}_rstd6"] = df.groupby("location_id")[col].transform(
            lambda x: x.rolling(6, min_periods=2).std().fillna(0))
        df[f"{col}_rmean6"] = df.groupby("location_id")[col].transform(
            lambda x: x.rolling(6, min_periods=1).mean().fillna(0))

    # Solar output per unit GHI (should be stable)
    df["hour"]  = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["is_day"] = ((df["hour"] >= 6) & (df["hour"] < 20)).astype(int)
    df["ghi_solar_ratio"] = np.where(
        df["ghi_w_m2"] > 10, df["solar_power_kw"] / df["ghi_w_m2"].clip(1), 0)
    df["solar_cap_util"] = df["solar_power_kw"] / df["solar_capacity_kw"].clip(1)
    df["wind_cap_util"]  = df["wind_power_kw"]  / df["wind_capacity_kw"].clip(1)

    # Stuck-value binary flags: cumulative delta ≈ 0 for 3+ consecutive hours
    df["solar_zero_delta"] = (df.groupby("location_id")["solar_power_kw"].transform(
        lambda x: x.diff().abs().rolling(3, min_periods=1).sum()) < 0.01).astype(float)
    df["wind_zero_delta"]  = (df.groupby("location_id")["wind_power_kw"].transform(
        lambda x: x.diff().abs().rolling(3, min_periods=1).sum()) < 0.01).astype(float)

    df["location_code"] = df["location_id"].astype("category").cat.codes
    return df


def load_splits():
    """Load raw data, add IF features, and split into train/val/test."""
    raw = pd.read_parquet(ROOT / "data" / "raw" / "energy_raw.parquet")
    raw = add_if_features(raw).dropna()

    n = len(raw)
    i_val  = int(n * 0.70)
    i_test = int(n * 0.85)
    train  = raw.iloc[:i_val].copy()
    val    = raw.iloc[i_val:i_test].copy()
    test   = raw.iloc[i_test:].copy()

    print(f"[✓] Loaded raw + features  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


def get_features(df: pd.DataFrame) -> np.ndarray:
    """Extract and return the sensor feature matrix. Drop rows with NaN."""
    avail = [c for c in SENSOR_FEATURES if c in df.columns]
    return df[avail].fillna(0).values, avail


# ── Training ──────────────────────────────────────────────────────────────────

def train_isolation_forest(X_train: np.ndarray,
                            contamination: float = 0.09) -> IsolationForest:
    model = IsolationForest(
        n_estimators=400,
        max_samples="auto",
        contamination=contamination,
        max_features=0.8,       # subsample features per tree — improves diversity
        bootstrap=False,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train)
    print(f"[✓] Isolation Forest trained  (n_estimators=400, contamination={contamination}, max_features=0.8)")
    return model


# ── Contamination tuning ──────────────────────────────────────────────────────

def tune_contamination(X_val: np.ndarray,
                        y_val: np.ndarray,
                        contamination_grid: list[float] | None = None) -> tuple[float, dict]:
    """
    Sweep contamination values, retrain, pick best val F1.
    Returns best contamination + full sweep results.
    """
    if contamination_grid is None:
        contamination_grid = [0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]

    results = []
    print("\nTuning contamination parameter on validation set:")
    print(f"  {'contamination':>14}  {'precision':>10}  {'recall':>8}  {'F1':>8}")

    for c in contamination_grid:
        m = IsolationForest(
            n_estimators=200, contamination=c,
            n_jobs=-1, random_state=42
        )
        m.fit(X_val)
        # IF returns -1 for anomaly, +1 for normal → convert to 0/1
        preds = (m.predict(X_val) == -1).astype(int)
        scores = -m.score_samples(X_val)   # higher = more anomalous

        ap  = average_precision_score(y_val, scores)
        f1  = f1_score(y_val, preds, zero_division=0)
        prec = (preds * y_val).sum() / (preds.sum() + 1e-9)
        rec  = (preds * y_val).sum() / (y_val.sum() + 1e-9)

        results.append({"contamination": c, "precision": prec, "recall": rec,
                         "f1": f1, "avg_precision": ap})
        print(f"  {c:>14.2f}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}")

    best = max(results, key=lambda r: r["f1"])
    print(f"\n[✓] Best contamination: {best['contamination']}  (F1={best['f1']:.4f})")
    return best["contamination"], results


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: IsolationForest,
             X_test: np.ndarray,
             y_test: np.ndarray,
             df_test: pd.DataFrame) -> dict:
    """Full evaluation on test set."""
    preds  = (model.predict(X_test) == -1).astype(int)
    scores = -model.score_samples(X_test)   # higher = more anomalous

    report  = classification_report(y_test, preds, output_dict=True)
    cm      = confusion_matrix(y_test, preds)
    ap      = average_precision_score(y_test, scores)
    prec_c, rec_c, thr_c = precision_recall_curve(y_test, scores)

    print("\n── Test Set Evaluation ────────────────────────────────")
    print(classification_report(y_test, preds, target_names=["Normal","Anomaly"]))

    # Per-location breakdown
    df_eval = df_test.copy()
    df_eval["pred_anomaly"] = preds
    df_eval["anomaly_score"] = scores

    per_loc = []
    for loc, grp in df_eval.groupby("location_id"):
        gt   = grp["anomaly"].values
        pr   = grp["pred_anomaly"].values
        f1   = f1_score(gt, pr, zero_division=0)
        prec = (pr * gt).sum() / (pr.sum() + 1e-9)
        rec  = (pr * gt).sum() / (gt.sum() + 1e-9)
        per_loc.append({"location": loc, "f1": round(f1,4),
                        "precision": round(prec,4), "recall": round(rec,4),
                        "n_anomalies_true": int(gt.sum()),
                        "n_anomalies_pred": int(pr.sum())})
        print(f"  {loc:25s}  F1={f1:.4f}  P={prec:.4f}  R={rec:.4f}")

    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "avg_precision_score": round(float(ap), 4),
        "per_location": per_loc,
        "pr_curve": {"precision": prec_c.tolist(), "recall": rec_c.tolist()},
    }
    return results, df_eval


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_tuning_curve(sweep_results: list[dict]):
    """Contamination sweep F1 curve."""
    df = pd.DataFrame(sweep_results)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["contamination"], df["f1"],        "o-", label="F1",        lw=2)
    ax.plot(df["contamination"], df["precision"],  "s--", label="Precision", lw=1.5)
    ax.plot(df["contamination"], df["recall"],     "^--", label="Recall",    lw=1.5)
    best = df.loc[df["f1"].idxmax()]
    ax.axvline(best["contamination"], color="red", lw=1, linestyle=":", label=f"Best ({best['contamination']:.2f})")
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Score")
    ax.set_title("Contamination Parameter Tuning (Validation Set)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_contamination_tuning.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] 01_contamination_tuning.png")


def plot_confusion_matrix(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Normal","Pred Anomaly"],
                yticklabels=["True Normal","True Anomaly"], ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontweight="bold")
    # Add rates
    tn, fp, fn, tp = cm.ravel()
    ax.set_xlabel(f"Precision={tp/(tp+fp+1e-9):.3f}   Recall={tp/(tp+fn+1e-9):.3f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] 02_confusion_matrix.png")


def plot_pr_curve(pr_data: dict, ap: float):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(pr_data["recall"], pr_data["precision"], lw=2, color="#E53935")
    ax.fill_between(pr_data["recall"], pr_data["precision"], alpha=0.15, color="#E53935")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve  (AP={ap:.3f})", fontweight="bold")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] 03_pr_curve.png")


def plot_per_location(per_loc: list[dict]):
    df = pd.DataFrame(per_loc).sort_values("f1", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#E53935" if f < 0.5 else "#43A047" if f >= 0.7 else "#FB8C00"
              for f in df["f1"]]
    bars = ax.barh(df["location"], df["f1"], color=colors)
    ax.bar_label(bars, [f"{v:.3f}" for v in df["f1"]], padding=3, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title("Anomaly Detection F1 per Location", fontweight="bold")
    ax.set_xlim([0, 1.1])
    ax.axvline(df["f1"].mean(), color="navy", lw=1.5, linestyle="--",
               label=f"Mean F1 = {df['f1'].mean():.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_per_location_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] 04_per_location_f1.png")


def plot_score_distribution(df_eval: pd.DataFrame):
    """Anomaly score distributions for normal vs anomalous points."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Isolation Forest Anomaly Score Distribution", fontweight="bold", fontsize=13)

    normal = df_eval[df_eval["anomaly"] == 0]["anomaly_score"]
    anom   = df_eval[df_eval["anomaly"] == 1]["anomaly_score"]

    # Global
    axes[0].hist(normal, bins=80, density=True, alpha=0.6, color="#42A5F5", label="Normal")
    axes[0].hist(anom,   bins=60, density=True, alpha=0.6, color="#E53935", label="Anomaly")
    axes[0].set_title("Global Score Distribution")
    axes[0].set_xlabel("Anomaly Score")
    axes[0].legend()

    # Box plot per anomaly type
    df_anom = df_eval[df_eval["anomaly"] == 1]
    order   = df_anom.groupby("anomaly_type")["anomaly_score"].median().sort_values(ascending=False).index
    df_anom.boxplot(column="anomaly_score", by="anomaly_type",
                    ax=axes[1], positions=range(len(order)),
                    patch_artist=True,
                    boxprops=dict(facecolor="#FFCCBC"),
                    medianprops=dict(color="#E53935", lw=2))
    axes[1].set_xticklabels(order)
    axes[1].set_title("Score by Anomaly Type")
    axes[1].set_xlabel("Anomaly Type")
    axes[1].set_ylabel("Anomaly Score")
    plt.suptitle("")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] 05_score_distribution.png")


def plot_anomaly_timeline(df_eval: pd.DataFrame):
    """Sample 1 location × 60 days showing detections on the time-series."""
    loc    = "US_AZ_Phoenix"
    sample = df_eval[df_eval["location_id"] == loc].copy()
    sample = sample.sort_values("timestamp")
    # Pick a 60-day window with known anomalies
    anom_dates = sample[sample["anomaly"] == 1]["timestamp"]
    if len(anom_dates) == 0:
        return
    center = anom_dates.iloc[len(anom_dates)//2]
    start  = center - pd.Timedelta(days=30)
    end    = center + pd.Timedelta(days=30)
    window = sample[(sample["timestamp"] >= start) & (sample["timestamp"] <= end)]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(f"Anomaly Detection Timeline — {loc} (60-day window)", fontweight="bold")

    for ax, col, color, label in zip(
        axes,
        ["solar_power_kw", "wind_power_kw", "anomaly_score"],
        ["#FFA726", "#42A5F5", "#AB47BC"],
        ["Solar Power (kW)", "Wind Power (kW)", "Anomaly Score"]
    ):
        ax.plot(window["timestamp"], window[col], color=color, lw=0.8, alpha=0.85)

        # True positives (correctly detected anomalies)
        tp = window[(window["anomaly"] == 1) & (window["pred_anomaly"] == 1)]
        ax.scatter(tp["timestamp"], tp[col], color="#E53935", s=25, zorder=5,
                   label="True Positive", alpha=0.9)
        # False negatives (missed anomalies)
        fn = window[(window["anomaly"] == 1) & (window["pred_anomaly"] == 0)]
        ax.scatter(fn["timestamp"], fn[col], color="#FF7043", s=25, marker="x", zorder=5,
                   label="False Negative", alpha=0.9)
        # False positives
        fp = window[(window["anomaly"] == 0) & (window["pred_anomaly"] == 1)]
        ax.scatter(fp["timestamp"], fp[col], color="#9C27B0", s=15, marker="^", zorder=5,
                   label="False Positive", alpha=0.6)

        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_detection_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[✓] 06_detection_timeline.png")


# ── Save cleaned dataset ──────────────────────────────────────────────────────

def save_cleaned_dataset(model: IsolationForest, feature_names: list[str]):
    """
    Apply the trained IF to the PROCESSED splits (used by LSTM),
    add pred_anomaly + anomaly_score columns, impute flagged rows,
    and save cleaned versions.
    """
    for split_name in ["train", "val", "test"]:
        # Load processed split (has all LSTM features)
        df_proc = pd.read_parquet(PROC / f"{split_name}.parquet")
        df_proc["timestamp"] = pd.to_datetime(df_proc["timestamp"])

        # Load corresponding raw rows to compute IF features
        raw_all = pd.read_parquet(ROOT / "data" / "raw" / "energy_raw.parquet")
        raw_feat = add_if_features(raw_all).dropna()

        # Match rows by location + timestamp
        merge_key = df_proc[["location_id", "timestamp"]].copy()
        merge_key["_idx"] = np.arange(len(df_proc))
        merged = merge_key.merge(
            raw_feat[["location_id", "timestamp"] + [f for f in feature_names if f in raw_feat.columns]],
            on=["location_id", "timestamp"], how="left"
        ).fillna(0)

        X = merged[[f for f in feature_names if f in merged.columns]].values
        preds  = (model.predict(X) == -1).astype(int)
        scores = -model.score_samples(X)

        df_proc["pred_anomaly"]  = preds
        df_proc["anomaly_score"] = scores

        # Impute flagged power readings with forward-fill per location
        df_clean = df_proc.copy()
        mask = df_clean["pred_anomaly"] == 1
        for col in ["solar_power_kw", "wind_power_kw", "total_power_kw"]:
            df_clean.loc[mask, col] = np.nan
            df_clean[col] = df_clean.groupby("location_id")[col].transform(
                lambda s: s.ffill().bfill())

        out = PROC / f"{split_name}_cleaned.parquet"
        df_clean.to_parquet(out, index=False)
        flagged = int(mask.sum())
        print(f"[✓] {split_name:5s} cleaned → {out.name}  "
              f"({flagged:,} rows flagged & imputed)")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── 1. Load
    train, val, test = load_splits()
    X_train, feat_names = get_features(train)
    X_val,   _          = get_features(val)
    X_test,  _          = get_features(test)
    y_train = train["anomaly"].values
    y_val   = val["anomaly"].values
    y_test  = test["anomaly"].values

    print(f"\nFeature matrix: {X_train.shape[1]} features")
    print(f"  {feat_names}")

    # ── 2. Initial training
    model = train_isolation_forest(X_train, contamination=0.08)

    # ── 3. Tune contamination on val
    best_c, sweep = tune_contamination(X_val, y_val)
    plot_tuning_curve(sweep)

    # ── 4. Retrain with best contamination on train+val combined
    print(f"\nRetraining with best contamination={best_c} on train+val...")
    X_trainval = np.vstack([X_train, X_val])
    model = train_isolation_forest(X_trainval, contamination=best_c)

    # ── 5. Evaluate on test
    eval_results, df_eval = evaluate(model, X_test, y_test, test)
    cm = np.array(eval_results["confusion_matrix"])

    # ── 6. Plots
    print("\nGenerating evaluation plots:")
    plot_confusion_matrix(cm)
    plot_pr_curve(eval_results["pr_curve"], eval_results["avg_precision_score"])
    plot_per_location(eval_results["per_location"])
    plot_score_distribution(df_eval)
    plot_anomaly_timeline(df_eval)

    # ── 7. Save cleaned dataset for LSTM
    print("\nSaving cleaned dataset splits:")
    save_cleaned_dataset(model, feat_names)

    # ── 8. Persist model
    model_payload = {
        "model":          model,
        "feature_names":  feat_names,
        "best_contamination": best_c,
        "eval_results":   eval_results,
    }
    with open(MODEL_DIR / "isolation_forest.pkl", "wb") as f:
        pickle.dump(model_payload, f)
    print(f"\n[✓] Model saved → {MODEL_DIR / 'isolation_forest.pkl'}")

    # ── 9. Save eval JSON
    eval_out = {k: v for k, v in eval_results.items() if k != "pr_curve"}
    eval_out["best_contamination"] = best_c
    eval_out["feature_names"] = feat_names
    (OUT_DIR / "eval_results.json").write_text(json.dumps(eval_out, indent=2))
    print(f"[✓] Eval results saved → {OUT_DIR / 'eval_results.json'}")

    print("\n── Summary ────────────────────────────────────────────")
    r = eval_results["classification_report"]
    # key is "1" (anomaly class) in sklearn's output when using integer labels
    anom_key = "1" if "1" in r else list(r.keys())[1]
    print(f"  Anomaly  Precision : {r[anom_key]['precision']:.4f}")
    print(f"  Anomaly  Recall    : {r[anom_key]['recall']:.4f}")
    print(f"  Anomaly  F1        : {r[anom_key]['f1-score']:.4f}")
    print(f"  Avg Precision Score: {eval_results['avg_precision_score']:.4f}")
    print(f"\nAll outputs → {OUT_DIR}")


if __name__ == "__main__":
    run()
