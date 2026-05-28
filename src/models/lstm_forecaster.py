"""
lstm_forecaster.py
------------------
LSTM-based multi-step energy forecaster implemented in pure NumPy.

Architecture:
  - Single-layer LSTM  (hidden_dim=32)
  - Sequence length:   24h lookback  (one full day)
  - Forecast horizon:  24h ahead     (one step at a time, autoregressive eval)
  - Targets:           solar_power_kw, wind_power_kw, total_power_kw
  - Optimizer:         Adam with gradient clipping

Benchmark experiment:
  Train A — raw data (unfiltered):  LSTM trained on original processed splits
  Train B — cleaned data (filtered): LSTM trained on IF-cleaned splits
  Compare MAE / RMSE / MAPE on the shared held-out test set
  → quantifies the uplift from anomaly filtering

Outputs:
  outputs/lstm/
    01_training_curves.png       loss curves for both models
    02_benchmark_comparison.png  MAE/RMSE bar chart A vs B
    03_forecast_vs_actual.png    sample predictions overlaid on ground truth
    04_error_distribution.png    residual distributions A vs B
    eval_results.json            all numeric results
  data/models/
    lstm_raw.npz                 weights for model A
    lstm_clean.npz               weights for model B
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
PROC      = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "data" / "models"
OUT_DIR   = ROOT / "outputs" / "lstm"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN    = 24          # lookback window (hours)
HIDDEN_DIM = 32          # LSTM hidden units
N_TARGETS  = 3           # solar, wind, total
BATCH_SIZE = 256
N_EPOCHS   = 15
LR         = 1e-3        # Adam learning rate
BETA1      = 0.9
BETA2      = 0.999
EPS        = 1e-8
GRAD_CLIP  = 5.0
TRAIN_ROWS = 40_000      # keeps memory under 200MB per split
VAL_ROWS   = 10_000
TEST_ROWS  = 10_000
SEED       = 42
DTYPE      = np.float32  # float32 to halve memory vs float64

# ── Feature selection ─────────────────────────────────────────────────────────
# Rich but compact feature set for the LSTM input at each timestep
INPUT_FEATURES = [
    # Power signals (scaled) — the primary sequence
    "solar_power_kw", "wind_power_kw", "total_power_kw",
    # Weather drivers
    "ghi_w_m2", "ambient_temp_c", "wind_speed_m_s",
    # Cyclical time encodings
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    # Context
    "is_daytime", "season", "is_weekend",
    # Lag features
    "solar_power_kw_lag1",  "solar_power_kw_lag24",
    "wind_power_kw_lag1",   "wind_power_kw_lag24",
    "total_power_kw_lag1",  "total_power_kw_lag24",
    # Rolling means
    "solar_power_kw_roll3h_mean", "solar_power_kw_roll24h_mean",
    "wind_power_kw_roll3h_mean",  "wind_power_kw_roll24h_mean",
    # Location identity
    "location_code",
]
INPUT_DIM = len(INPUT_FEATURES)
TARGET_FEATURES = ["solar_power_kw", "wind_power_kw", "total_power_kw"]

RNG = np.random.default_rng(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# LSTM Cell and training functions implemented in pure NumPy
# ─────────────────────────────────────────────────────────────────────────────

class LSTMParams:
    """All trainable parameters + Adam moment accumulators."""

    def __init__(self, input_dim: int, hidden_dim: int, n_targets: int):
        scale = 0.02
        self.Wx  = RNG.standard_normal((input_dim,  4 * hidden_dim)).astype(DTYPE) * scale
        self.Wh  = RNG.standard_normal((hidden_dim, 4 * hidden_dim)).astype(DTYPE) * scale
        self.b   = np.zeros(4 * hidden_dim, dtype=DTYPE)
        self.b[hidden_dim:2*hidden_dim] = 1.0
        self.Wy  = RNG.standard_normal((hidden_dim, n_targets)).astype(DTYPE) * scale
        self.by  = np.zeros(n_targets, dtype=DTYPE)
        self._init_moments()

    def _init_moments(self):
        for attr in ["Wx", "Wh", "b", "Wy", "by"]:
            w = getattr(self, attr)
            setattr(self, f"m_{attr}", np.zeros_like(w))
            setattr(self, f"v_{attr}", np.zeros_like(w))

    def param_names(self):
        return ["Wx", "Wh", "b", "Wy", "by"]

    def arrays(self):
        return {n: getattr(self, n) for n in self.param_names()}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))


def forward(params: LSTMParams,
            x_seq: np.ndarray) -> tuple[np.ndarray, list, np.ndarray, np.ndarray]:
    batch = x_seq.shape[0]
    hd    = params.Wx.shape[1] // 4
    h = np.zeros((batch, hd), dtype=DTYPE)
    c = np.zeros((batch, hd), dtype=DTYPE)
    cache = []

    for t in range(x_seq.shape[1]):
        x_t  = x_seq[:, t, :]
        raw  = x_t @ params.Wx + h @ params.Wh + params.b

        i_a = sigmoid(raw[:, :hd])
        f_a = sigmoid(raw[:, hd:2*hd])
        g_a = np.tanh(raw[:, 2*hd:3*hd])
        o_a = sigmoid(raw[:, 3*hd:])

        c_new  = f_a * c + i_a * g_a
        tanh_c = np.tanh(c_new)
        h_new  = o_a * tanh_c

        cache.append((x_t, h.copy(), c.copy(), i_a, f_a, g_a, o_a, c_new, tanh_c))
        h, c = h_new, c_new

    pred = h @ params.Wy + params.by
    return pred, cache, h, c


def backward(params: LSTMParams,
             x_seq: np.ndarray,
             pred:  np.ndarray,
             y:     np.ndarray,
             cache: list) -> dict:
    batch = x_seq.shape[0]
    hd    = params.Wx.shape[1] // 4

    # Output gradients
    d_out = (2.0 / batch) * (pred - y).astype(DTYPE)
    _, h_last, _, _, _, _, _, _, _ = cache[-1]
    # recompute final h
    _,_,_,i_a,f_a,g_a,o_a,c_new,tanh_c = cache[-1]
    h_last_actual = o_a * tanh_c

    dWy  = h_last_actual.T @ d_out
    dby  = d_out.sum(axis=0)
    dh   = d_out @ params.Wy.T
    dc   = np.zeros((batch, hd), dtype=DTYPE)

    dWx = np.zeros_like(params.Wx)
    dWh = np.zeros_like(params.Wh)
    db  = np.zeros_like(params.b)

    for t in reversed(range(len(cache))):
        x_t, h_prev, c_prev, i_a, f_a, g_a, o_a, c_new, tanh_c = cache[t]

        do = dh * tanh_c
        dc += dh * o_a * (1.0 - tanh_c**2)
        di = dc * g_a
        df = dc * c_prev
        dg = dc * i_a
        dc = dc * f_a

        d_raw = np.concatenate([
            di * i_a * (1 - i_a),
            df * f_a * (1 - f_a),
            dg * (1 - g_a**2),
            do * o_a * (1 - o_a),
        ], axis=1)

        dWx += x_t.T @ d_raw
        dWh += h_prev.T @ d_raw
        db  += d_raw.sum(axis=0)
        dh   = d_raw @ params.Wh.T

    return {"Wx": dWx, "Wh": dWh, "b": db, "Wy": dWy, "by": dby}


def adam_step(params: LSTMParams, grads: dict, lr: float, t: int):
    """In-place Adam parameter update."""
    for name in params.param_names():
        g  = grads[name]
        # Gradient clipping
        gnorm = np.linalg.norm(g)
        if gnorm > GRAD_CLIP:
            g = g * (GRAD_CLIP / gnorm)

        m  = getattr(params, f"m_{name}")
        v  = getattr(params, f"v_{name}")
        m  = BETA1 * m + (1 - BETA1) * g
        v  = BETA2 * v + (1 - BETA2) * g**2
        m_hat = m / (1 - BETA1**t)
        v_hat = v / (1 - BETA2**t)
        setattr(params, name,   getattr(params, name) - lr * m_hat / (np.sqrt(v_hat) + EPS))
        setattr(params, f"m_{name}", m)
        setattr(params, f"v_{name}", v)


# ─────────────────────────────────────────────────────────────────────────────
# Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prep(suffix: str = "") -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray, np.ndarray]:
    """
    Load processed splits (raw or cleaned), build sliding-window sequences.
    suffix: "" for raw, "_cleaned" for IF-filtered.
    Returns X_train, y_train, X_val, y_val, X_test, y_test
    """
    tag = suffix if suffix else ""
    train_path = PROC / f"train{tag}.parquet"
    val_path   = PROC / f"val{tag}.parquet"
    test_path  = PROC / f"test{tag}.parquet"

    print(f"\n  Loading splits (suffix='{tag}')...")
    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    def make_sequences(df: pd.DataFrame,
                       max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Build (X, y) sliding windows per location then concatenate.
        X shape: (N, SEQ_LEN, INPUT_DIM)
        y shape: (N, N_TARGETS)
        """
        avail_feats  = [f for f in INPUT_FEATURES  if f in df.columns]
        avail_target = [f for f in TARGET_FEATURES if f in df.columns]

        all_X, all_y = [], []
        for _, grp in df.groupby("location_id"):
            grp = grp.sort_values("timestamp").reset_index(drop=True)
            X_vals = grp[avail_feats].values.astype(np.float64)
            y_vals = grp[avail_target].values.astype(np.float64)

            n = len(grp)
            if n <= SEQ_LEN:
                continue
            for i in range(SEQ_LEN, n):
                all_X.append(X_vals[i-SEQ_LEN:i])   # (SEQ_LEN, INPUT_DIM)
                all_y.append(y_vals[i])               # (N_TARGETS,)

        X = np.array(all_X, dtype=DTYPE)
        y = np.array(all_y, dtype=DTYPE)

        if max_rows and len(X) > max_rows:
            idx = RNG.choice(len(X), size=max_rows, replace=False)
            X, y = X[idx], y[idx]

        return X, y

    print(f"  Building sequences (seq_len={SEQ_LEN}) ...")
    X_tr, y_tr = make_sequences(train_df, max_rows=TRAIN_ROWS)
    X_va, y_va = make_sequences(val_df,   max_rows=VAL_ROWS)
    X_te, y_te = make_sequences(test_df,  max_rows=TEST_ROWS)

    print(f"  X_train={X_tr.shape}  X_val={X_va.shape}  X_test={X_te.shape}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_tr: np.ndarray, y_tr: np.ndarray,
                X_va: np.ndarray, y_va: np.ndarray,
                label: str) -> tuple[LSTMParams, dict]:
    """
    Train LSTM with Adam, return best params (by val loss) + history.
    """
    params = LSTMParams(INPUT_DIM, HIDDEN_DIM, N_TARGETS)
    n_batches  = len(X_tr) // BATCH_SIZE
    best_val   = np.inf
    best_state = None
    history    = {"train_loss": [], "val_loss": [], "epoch_time": []}
    step       = 0

    print(f"\n{'='*58}")
    print(f"  Training: {label}")
    print(f"  Batches/epoch={n_batches}  epochs={N_EPOCHS}  hidden={HIDDEN_DIM}")
    print(f"{'='*58}")
    print(f"  {'Epoch':>5}  {'Train MSE':>10}  {'Val MSE':>10}  {'Time':>8}")

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()
        # Shuffle training data
        idx = RNG.permutation(len(X_tr))
        X_sh, y_sh = X_tr[idx], y_tr[idx]

        epoch_loss = 0.0
        for b in range(n_batches):
            Xb = X_sh[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            yb = y_sh[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
            step += 1

            pred, cache, _, _ = forward(params, Xb)
            loss = float(((pred - yb)**2).mean())
            grads = backward(params, Xb, pred, yb, cache)
            adam_step(params, grads, LR, step)
            epoch_loss += loss

        train_loss = epoch_loss / n_batches

        # Validation loss (no grad)
        val_preds, _, _, _ = forward(params, X_va)
        val_loss = float(((val_preds - y_va)**2).mean())

        elapsed = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(elapsed)

        flag = " ✓" if val_loss < best_val else ""
        print(f"  {epoch:>5}  {train_loss:>10.5f}  {val_loss:>10.5f}  {elapsed:>6.1f}s{flag}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {n: getattr(params, n).copy() for n in params.param_names()}

    # Restore best weights
    for n, w in best_state.items():
        setattr(params, n, w)

    print(f"\n  Best val MSE: {best_val:.5f}")
    return params, history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(params: LSTMParams,
                    X_test: np.ndarray,
                    y_test: np.ndarray,
                    target_names: list[str]) -> dict:
    """Compute MAE, RMSE, MAPE per target on test set."""
    pred, _, _, _ = forward(params, X_test)
    metrics = {}
    for i, name in enumerate(target_names):
        y_true = y_test[:, i]
        y_pred = pred[:, i]
        mae    = float(np.mean(np.abs(y_true - y_pred)))
        rmse   = float(np.sqrt(np.mean((y_true - y_pred)**2)))
        # MAPE: only where |y_true| > threshold (avoid division by ~0 at night)
        mask   = np.abs(y_true) > 0.1
        mape   = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) \
                 if mask.sum() > 0 else float("nan")
        metrics[name] = {"MAE": round(mae, 5), "RMSE": round(rmse, 5), "MAPE": round(mape, 3)}
    return metrics, pred


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(hist_raw: dict, hist_clean: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("LSTM Training Curves", fontsize=13, fontweight="bold")

    for ax, hist, label, color in zip(
        axes,
        [hist_raw, hist_clean],
        ["Model A — Raw Data", "Model B — IF-Filtered Data"],
        ["#E53935", "#43A047"]
    ):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(epochs, hist["train_loss"], "o-", color=color,       lw=2, label="Train MSE")
        ax.plot(epochs, hist["val_loss"],   "s--", color=color, lw=2, alpha=0.7, label="Val MSE")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("training_curves.png")


def plot_benchmark(metrics_raw: dict, metrics_clean: dict,
                   target_names: list[str]):
    """Side-by-side MAE and RMSE comparison across targets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Benchmark: Raw vs IF-Filtered Training Data",
                 fontsize=13, fontweight="bold")

    metric_names = ["MAE", "RMSE", "MAPE"]
    for ax, metric in zip(axes, metric_names):
        raw_vals   = [metrics_raw[t][metric]   for t in target_names]
        clean_vals = [metrics_clean[t][metric] for t in target_names]

        x = np.arange(len(target_names))
        w = 0.35
        b1 = ax.bar(x - w/2, raw_vals,   width=w, label="Raw",      color="#E53935", alpha=0.85)
        b2 = ax.bar(x + w/2, clean_vals, width=w, label="Filtered", color="#43A047", alpha=0.85)
        ax.bar_label(b1, [f"{v:.3f}" for v in raw_vals],   padding=2, fontsize=8)
        ax.bar_label(b2, [f"{v:.3f}" for v in clean_vals], padding=2, fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_power_kw","") for t in target_names])
        ax.set_title(f"{metric} ({'scaled units' if metric != 'MAPE' else '%'})")
        ax.legend()

        # Improvement annotation
        avg_improve = 100 * (np.mean(raw_vals) - np.mean(clean_vals)) / (np.mean(raw_vals) + 1e-9)
        ax.set_xlabel(f"Avg improvement: {avg_improve:+.1f}%")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("benchmark_comparison.png")


def plot_forecast_vs_actual(params_raw: LSTMParams,
                             params_clean: LSTMParams,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             target_names: list[str]):
    """Plot 5-day forecast vs actual for total_power_kw."""
    n_show = 120   # 5 days of hours
    idx    = 2     # total_power_kw

    pred_raw,   _, _, _ = forward(params_raw,   X_test[:n_show])
    pred_clean, _, _, _ = forward(params_clean, X_test[:n_show])

    fig, ax = plt.subplots(figsize=(14, 5))
    hours = np.arange(n_show)
    ax.plot(hours, y_test[:n_show, idx],       color="#333",     lw=1.5, label="Actual",        zorder=3)
    ax.plot(hours, pred_raw[:n_show, idx],     color="#E53935",  lw=1.2, label="Raw model",     alpha=0.8, linestyle="--")
    ax.plot(hours, pred_clean[:n_show, idx],   color="#43A047",  lw=1.2, label="Filtered model",alpha=0.8, linestyle="-.")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Total Power (scaled)")
    ax.set_title("120h Forecast vs Actual — Total Power", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_forecast_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("forecast_vs_actual.png")


def plot_error_distribution(pred_raw: np.ndarray,
                             pred_clean: np.ndarray,
                             y_test: np.ndarray,
                             target_names: list[str]):
    """Residual distributions for Raw vs Filtered models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Residual Distributions — Raw vs IF-Filtered Model", fontsize=12, fontweight="bold")

    for ax, i, name in zip(axes, range(3), target_names):
        res_raw   = y_test[:, i] - pred_raw[:, i]
        res_clean = y_test[:, i] - pred_clean[:, i]
        ax.hist(res_raw,   bins=80, density=True, alpha=0.55,
                color="#E53935", label=f"Raw   σ={res_raw.std():.4f}")
        ax.hist(res_clean, bins=80, density=True, alpha=0.55,
                color="#43A047", label=f"Clean σ={res_clean.std():.4f}")
        ax.axvline(0, color="black", lw=1, linestyle=":")
        ax.set_title(name.replace("_power_kw", "").title())
        ax.set_xlabel("Residual (scaled)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("error_distribution.png")


def plot_improvement_summary(metrics_raw: dict, metrics_clean: dict,
                              target_names: list[str]):
    """Single summary chart: % improvement per target and metric."""
    fig, ax = plt.subplots(figsize=(10, 5))

    rows, colors = [], []
    metric_list = ["MAE", "RMSE"]
    bar_labels, bar_vals = [], []

    for t in target_names:
        for m in metric_list:
            raw_v   = metrics_raw[t][m]
            clean_v = metrics_clean[t][m]
            improve = 100 * (raw_v - clean_v) / (raw_v + 1e-9)
            bar_labels.append(f"{t.replace('_power_kw','').title()}\n{m}")
            bar_vals.append(improve)

    colors_bars = ["#43A047" if v >= 0 else "#E53935" for v in bar_vals]
    bars = ax.bar(range(len(bar_vals)), bar_vals, color=colors_bars, alpha=0.85, width=0.6)
    ax.bar_label(bars, [f"{v:+.1f}%" for v in bar_vals], padding=3, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylabel("% Improvement (positive = Filtered better)")
    ax.set_title("Forecasting Improvement from Anomaly Filtering (IF-Filtered vs Raw)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_improvement_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("improvement_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Save / load weights
# ─────────────────────────────────────────────────────────────────────────────

def save_model(params: LSTMParams, path: Path):
    np.savez(path, **params.arrays())
    print(f"Saved model → {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run():
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Load both dataset variants ────────────────────────────────────────────
    print("Loading RAW data...")
    X_tr_r, y_tr_r, X_va_r, y_va_r, X_te_r, y_te_r = load_and_prep(suffix="")
    print("\nLoading IF-FILTERED (cleaned) data...")
    X_tr_c, y_tr_c, X_va_c, y_va_c, X_te_c, y_te_c = load_and_prep(suffix="_cleaned")

    # Both models evaluated on the SAME raw test set for a fair comparison
    X_te_shared, y_te_shared = X_te_r, y_te_r

    # ── Train Model A: raw ────────────────────────────────────────────────────
    params_raw, hist_raw = train_model(
        X_tr_r, y_tr_r, X_va_r, y_va_r, label="Model A — Raw (unfiltered)")

    # ── Train Model B: cleaned ────────────────────────────────────────────────
    params_clean, hist_clean = train_model(
        X_tr_c, y_tr_c, X_va_c, y_va_c, label="Model B — IF-Filtered")

    # ── Evaluate both on shared test set ──────────────────────────────────────
    print("\n── Evaluating on shared test set ──────────────────────")
    metrics_raw,   pred_raw   = compute_metrics(params_raw,   X_te_shared, y_te_shared, TARGET_FEATURES)
    metrics_clean, pred_clean = compute_metrics(params_clean, X_te_shared, y_te_shared, TARGET_FEATURES)

    print("\nModel A (Raw):")
    for t, m in metrics_raw.items():
        print(f"  {t:30s}  MAE={m['MAE']:.5f}  RMSE={m['RMSE']:.5f}  MAPE={m['MAPE']:.2f}%")
    print("\nModel B (IF-Filtered):")
    for t, m in metrics_clean.items():
        print(f"  {t:30s}  MAE={m['MAE']:.5f}  RMSE={m['RMSE']:.5f}  MAPE={m['MAPE']:.2f}%")

    # Compute overall improvement
    mae_improvements = {}
    for t in TARGET_FEATURES:
        raw_mae   = metrics_raw[t]["MAE"]
        clean_mae = metrics_clean[t]["MAE"]
        pct       = 100 * (raw_mae - clean_mae) / (raw_mae + 1e-9)
        mae_improvements[t] = round(pct, 2)
    avg_improve = round(float(np.mean(list(mae_improvements.values()))), 2)

    print(f"\n── MAE Improvements from Anomaly Filtering ────────────")
    for t, pct in mae_improvements.items():
        arrow = "↑" if pct > 0 else "↓"
        print(f"  {t:30s}: {pct:+.2f}% {arrow}")
    print(f"  {'Average':30s}: {avg_improve:+.2f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots:")
    plot_training_curves(hist_raw, hist_clean)
    plot_benchmark(metrics_raw, metrics_clean, TARGET_FEATURES)
    plot_forecast_vs_actual(params_raw, params_clean, X_te_shared, y_te_shared, TARGET_FEATURES)
    plot_error_distribution(pred_raw, pred_clean, y_te_shared, TARGET_FEATURES)
    plot_improvement_summary(metrics_raw, metrics_clean, TARGET_FEATURES)

    # ── Save models ───────────────────────────────────────────────────────────
    save_model(params_raw,   MODEL_DIR / "lstm_raw.npz")
    save_model(params_clean, MODEL_DIR / "lstm_clean.npz")

    # ── Persist results ───────────────────────────────────────────────────────
    results = {
        "model_config": {
            "input_dim": INPUT_DIM, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "n_targets": N_TARGETS,
            "n_epochs": N_EPOCHS, "batch_size": BATCH_SIZE,
            "train_rows": TRAIN_ROWS, "input_features": INPUT_FEATURES,
        },
        "metrics_raw":   metrics_raw,
        "metrics_clean": metrics_clean,
        "mae_improvements_pct": mae_improvements,
        "avg_mae_improvement_pct": avg_improve,
        "training_history_raw":   {k: [round(v,6) for v in hist_raw[k]]
                                    for k in ["train_loss","val_loss"]},
        "training_history_clean": {k: [round(v,6) for v in hist_clean[k]]
                                    for k in ["train_loss","val_loss"]},
    }
    (OUT_DIR / "eval_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results saved → {OUT_DIR / 'eval_results.json'}")

    print(f"\n{'='*58}")
    print(f"  Avg MAE improvement from IF filtering: {avg_improve:+.2f}%")
    print(f"  All outputs → {OUT_DIR}")
    print(f"{'='*58}")

    return results


if __name__ == "__main__":
    run()
