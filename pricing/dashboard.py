"""Pricing model dashboard.

Three views:
  1. Calibration   — model-only diagnostics (calibration curve, LL breakdowns)
  2. Model vs PM   — head-to-head comparison with Polymarket
  3. Single Market  — intra-hour price / model / PM with parameter tuning

Launch:
    streamlit run pricing/dashboard.py
"""

import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pricing.models import get_model, Model, CalibrationResult
from pricing.calibrate import calibrate_vol, calibrate_tail, log_loss
from pricing.dataset import build_dataset, DatasetConfig
from pricing.features.seasonal_vol import compute_seasonal_vol
from pricing.features.realized_vol import compute_rv_ewma
from src.data import (
    load_binance, load_binance_labels,
    load_polymarket_market, get_cache_status,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")
OUTPUT_DIR = project_root / "pricing" / "output"
DATASET_PATH = OUTPUT_DIR / "calibration_dataset.parquet"

COLORS = {
    "model": "#E91E63",
    "pm_mid": "#3b82f6",
    "pm_bid": "#22c55e",
    "pm_ask": "#ef4444",
    "bnc_mid": "#f59e0b",
    "K": "#ff6b00",
    "S_T": "#a855f7",
    "edge_pos": "rgba(34,197,94,0.3)",
    "edge_neg": "rgba(239,68,68,0.3)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _et_hour_to_utc(market_date: date, hour_et: int) -> tuple[datetime, datetime]:
    """Convert ET market hour to UTC start/end datetimes."""
    et_start = datetime(market_date.year, market_date.month, market_date.day,
                        hour_et, 0, 0, tzinfo=ET)
    et_end = et_start + timedelta(hours=1)
    return et_start.astimezone(timezone.utc), et_end.astimezone(timezone.utc)


def _ts_to_dt(ts_ms):
    """Convert epoch ms array/series to UTC datetimes."""
    return pd.to_datetime(ts_ms, unit="ms", utc=True)


def _available_dates(asset: str = "BTC") -> list[date]:
    """Return sorted dates with cached Polymarket data."""
    cache_dir = project_root / "data" / "resampled_data" / "polymarket" / f"asset={asset}" / "interval=1s"
    if not cache_dir.exists():
        return []
    dates = []
    for p in sorted(cache_dir.glob("date=*.parquet")):
        try:
            d = datetime.strptime(p.stem, "date=%Y-%m-%d").date()
            dates.append(d)
        except ValueError:
            continue
    return dates


def equal_mass_reliability(p_pred: np.ndarray, y: np.ndarray, n_bins: int = 20):
    """Reliability curve using equal-mass (quantile) bins."""
    order = np.argsort(p_pred)
    p_sorted = p_pred[order]
    y_sorted = y[order]
    bin_size = len(p_pred) // n_bins

    bin_pred, bin_actual, bin_se, bin_n = [], [], [], []
    for b in range(n_bins):
        lo = b * bin_size
        hi = (b + 1) * bin_size if b < n_bins - 1 else len(p_pred)
        n = hi - lo
        if n == 0:
            continue
        p_mean = np.mean(p_sorted[lo:hi])
        y_mean = np.mean(y_sorted[lo:hi])
        se = np.sqrt(y_mean * (1 - y_mean) / n) if n > 1 else 0
        bin_pred.append(p_mean)
        bin_actual.append(y_mean)
        bin_se.append(se)
        bin_n.append(n)
    return np.array(bin_pred), np.array(bin_actual), np.array(bin_se), np.array(bin_n)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def load_cached_dataset() -> pd.DataFrame:
    if DATASET_PATH.exists():
        return pd.read_parquet(DATASET_PATH)
    return pd.DataFrame()


def load_cached_params(model_name: str) -> dict | None:
    path = OUTPUT_DIR / f"{model_name}_params.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_single_market_data(
    asset: str, market_date: str, hour_et: int,
    ewma_half_life_sec: float = 300.0,
):
    """Load and prepare all data for a single market view.

    Returns dict on success, or error string on failure.
    """
    mdate = date.fromisoformat(market_date)
    utc_start, utc_end = _et_hour_to_utc(mdate, hour_et)
    utc_label = f"{utc_start.strftime('%Y-%m-%d %H:%M')}–{utc_end.strftime('%H:%M')} UTC"

    hour_start_ms = int(utc_start.timestamp() * 1000)
    hour_end_ms = int(utc_end.timestamp() * 1000)

    # Labels
    labels = load_binance_labels(start=utc_start, end=utc_end, asset=asset)
    if labels.empty:
        return f"No Binance labels for {utc_label}. The UTC date may be outside cached data range."

    lbl = labels.iloc[0]
    K = float(lbl["K"])
    S_T = float(lbl["S_T"])
    Y = int(lbl["Y"])

    # Binance 1s with lookback for vol
    lookback_start = utc_start - timedelta(hours=3)
    bnc = load_binance(
        start=lookback_start, end=utc_end, asset=asset, interval="1s",
    )
    if bnc.empty:
        return f"No Binance 1s data for {utc_label}."

    # Seasonal vol
    seasonal = compute_seasonal_vol(bnc)

    # EWMA sigma_rv
    ts_rv, sigma_rv_full = compute_rv_ewma(bnc, seasonal_vol=seasonal, half_life_sec=ewma_half_life_sec)
    if len(ts_rv) == 0:
        return None

    # Filter to market hour
    ts_bnc = bnc["ts_recv"].values
    mid_bnc = bnc["mid_px"].values
    in_market = (ts_bnc >= hour_start_ms) & (ts_bnc < hour_end_ms)
    ts_market = ts_bnc[in_market]
    mid_market = mid_bnc[in_market]

    if len(ts_market) == 0:
        return None

    # Map features
    sigma_tod_market = seasonal.lookup_array(ts_market)

    idx_rv = np.searchsorted(ts_rv, ts_market)
    idx_rv = np.clip(idx_rv, 0, len(ts_rv) - 1)
    idx_prev = np.clip(idx_rv - 1, 0, len(ts_rv) - 1)
    dist_right = np.abs(ts_rv[idx_rv] - ts_market)
    dist_left = np.abs(ts_rv[idx_prev] - ts_market)
    best_idx = np.where(dist_left < dist_right, idx_prev, idx_rv)
    sigma_rv_market = sigma_rv_full[best_idx]
    sigma_rel_market = sigma_rv_market / np.maximum(sigma_tod_market, 1e-12)

    tau_market = (hour_end_ms - ts_market) / 1000.0

    S_arr = mid_market.astype(np.float64)
    K_arr = np.full_like(S_arr, K)

    # Time since last price change
    _valid_mid = ~np.isnan(mid_bnc)
    _ch = np.zeros(len(mid_bnc), dtype=bool)
    _ch[0] = True
    _ch[1:] = (mid_bnc[1:] != mid_bnc[:-1]) & _valid_mid[1:] & _valid_mid[:-1]
    _change_ts = ts_bnc[_ch]
    _si = np.searchsorted(_change_ts, ts_market, side="right") - 1
    _si = np.clip(_si, 0, len(_change_ts) - 1)
    time_since_move_market = np.maximum((ts_market - _change_ts[_si]) / 1000.0, 0.0)

    # ET hour (constant for all rows in this market)
    from zoneinfo import ZoneInfo
    _hour_et_val = utc_start.astimezone(ZoneInfo("America/New_York")).hour
    hour_et_market = np.full(len(ts_market), _hour_et_val, dtype=np.float64)

    features = {
        "sigma_tod": sigma_tod_market.astype(np.float64),
        "sigma_rv": sigma_rv_market.astype(np.float64),
        "sigma_rel": sigma_rel_market.astype(np.float64),
        "time_since_move": time_since_move_market.astype(np.float64),
        "hour_et": hour_et_market,
    }

    # Polymarket
    pm_times_out, pm_mid_out = None, None
    try:
        pm = load_polymarket_market(asset=asset, date=market_date, hour_et=hour_et, interval="1s")
        if not pm.empty:
            pm_in = (pm["ts_recv"].values >= hour_start_ms) & (pm["ts_recv"].values < hour_end_ms)
            if pm_in.sum() > 0:
                pm_times_out = pm["ts_recv"].values[pm_in]
                pm_mid_out = pm["mid"].values[pm_in]
    except Exception:
        pass

    return {
        "ts_market": ts_market,
        "mid_market": mid_market,
        "K": K, "S_T": S_T, "Y": Y,
        "S_arr": S_arr, "K_arr": K_arr,
        "tau_arr": tau_market.astype(np.float64),
        "features": features,
        "pm_ts": pm_times_out,
        "pm_mid": pm_mid_out,
        "hour_start_ms": hour_start_ms,
        "hour_end_ms": hour_end_ms,
    }


# ---------------------------------------------------------------------------
# Dataset view: diagnostic charts (plotly)
# ---------------------------------------------------------------------------

def _bucket_ll(y, p, buckets, values):
    """Compute log loss per bucket. Returns list of (ll, n) tuples."""
    out = []
    for i in range(len(buckets) - 1):
        mask = (values >= buckets[i]) & (values < buckets[i + 1])
        n = mask.sum()
        out.append((log_loss(y[mask], p[mask]), n) if n > 10 else (np.nan, n))
    return out


def _dataset_arrays(model: Model, params: dict, dataset: pd.DataFrame):
    """Extract common arrays from dataset. Returns (S, K, tau, y, feats, p_pred, tau_min, sigma_rel)."""
    S = dataset["S"].values.astype(np.float64)
    K = dataset["K"].values.astype(np.float64)
    tau = dataset["tau"].values.astype(np.float64)
    y = dataset["y"].values.astype(np.float64)
    feats = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}
    p_pred = np.clip(model.predict(params, S, K, tau, feats), 1e-9, 1.0 - 1e-9)
    tau_min = tau / 60.0
    sigma_rel = dataset["sigma_rel"].values.astype(np.float64) if "sigma_rel" in dataset.columns else None
    return S, K, tau, y, feats, p_pred, tau_min, sigma_rel


def render_calibration_view(model: Model, params: dict, dataset: pd.DataFrame):
    """View 1 — model-only calibration diagnostics."""
    _, _, tau, y, _, p_pred, tau_min, sigma_rel = _dataset_arrays(model, params, dataset)

    ll = log_loss(y, p_pred)
    ll_baseline = log_loss(y, np.full_like(y, y.mean()))
    improvement = (ll_baseline - ll) / ll_baseline * 100
    n_markets = dataset["market_id"].nunique() if "market_id" in dataset.columns else 0

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Log Loss", f"{ll:.4f}")
    c2.metric("Baseline", f"{ll_baseline:.4f}")
    c3.metric("Improvement", f"{improvement:+.1f}%")
    c4.metric("Samples", f"{len(y):,} / {n_markets} mkts")

    with st.expander("Fitted parameters"):
        param_df = pd.DataFrame({"Parameter": list(params.keys()), "Value": list(params.values())})
        st.dataframe(param_df, width="stretch", hide_index=True)

    # ===== 2x3 diagnostic grid =====
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Calibration (equal-mass bins)",
            "Log Loss by τ",
            "Log Loss by |score|",
            "Log Loss by σ_rel",
            "Prediction Distribution",
            "Residual by τ",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # 1. Calibration curve
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="black", width=1), showlegend=False,
    ), row=1, col=1)
    bp, ba, bse, _ = equal_mass_reliability(p_pred, y, n_bins=20)
    fig.add_trace(go.Scatter(
        x=bp, y=ba, name="Model",
        error_y=dict(type="data", array=1.96 * bse, visible=True),
        mode="markers", marker=dict(size=7, color=COLORS["model"]),
    ), row=1, col=1)
    fig.update_xaxes(title_text="Predicted", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Actual", range=[-0.05, 1.05], row=1, col=1)

    # 2. LL by tau
    tau_buckets = [0, 5, 15, 30, 60]
    tau_labels = ["0-5", "5-15", "15-30", "30-60"]
    model_tau = _bucket_ll(y, p_pred, tau_buckets, tau_min)
    fig.add_trace(go.Bar(
        x=tau_labels, y=[v[0] for v in model_tau],
        marker_color=COLORS["model"], showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=ll_baseline, line_dash="dot", line_color="gray", row=1, col=2)
    fig.add_hline(y=ll, line_dash="dot", line_color=COLORS["model"], row=1, col=2)
    fig.update_xaxes(title_text="τ (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="Log Loss", row=1, col=2)

    # 3. LL by |score|
    score = norm.ppf(np.clip(p_pred, 1e-6, 1 - 1e-6))
    s_abs = np.abs(score)
    s_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
    s_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
    model_s = _bucket_ll(y, p_pred, s_buckets, s_abs)
    fig.add_trace(go.Bar(
        x=s_labels, y=[v[0] for v in model_s],
        marker_color=COLORS["model"], showlegend=False,
    ), row=1, col=3)
    fig.add_hline(y=ll_baseline, line_dash="dot", line_color="gray", row=1, col=3)
    fig.update_xaxes(title_text="|score|", row=1, col=3)
    fig.update_yaxes(title_text="Log Loss", row=1, col=3)

    # 4. LL by sigma_rel
    if sigma_rel is not None:
        rel_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
        rel_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
        model_rel = _bucket_ll(y, p_pred, rel_buckets, sigma_rel)
        fig.add_trace(go.Bar(
            x=rel_labels, y=[v[0] for v in model_rel],
            marker_color=COLORS["model"], showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=ll_baseline, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_xaxes(title_text="σ_rel", row=2, col=1)
    fig.update_yaxes(title_text="Log Loss", row=2, col=1)

    # 5. Prediction distribution by outcome
    fig.add_trace(go.Histogram(
        x=p_pred[y == 1], nbinsx=30, opacity=0.6,
        marker_color="#2196F3", name="y=1 (Up)", histnorm="probability density",
    ), row=2, col=2)
    fig.add_trace(go.Histogram(
        x=p_pred[y == 0], nbinsx=30, opacity=0.6,
        marker_color="#ff5722", name="y=0 (Down)", histnorm="probability density",
    ), row=2, col=2)
    fig.update_xaxes(title_text="Predicted P(Up)", row=2, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=2)

    # 6. Mean residual (y - p) by tau bucket
    res_means, res_se = [], []
    for i in range(len(tau_buckets) - 1):
        m = (tau_min >= tau_buckets[i]) & (tau_min < tau_buckets[i + 1])
        n = m.sum()
        if n > 10:
            r = y[m] - p_pred[m]
            res_means.append(np.mean(r))
            res_se.append(1.96 * np.std(r) / np.sqrt(n))
        else:
            res_means.append(np.nan)
            res_se.append(0)
    fig.add_trace(go.Bar(
        x=tau_labels, y=res_means,
        error_y=dict(type="data", array=res_se, visible=True),
        marker_color=["#22c55e" if r > 0 else "#ef4444" for r in res_means],
        showlegend=False,
    ), row=2, col=3)
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=3)
    fig.update_xaxes(title_text="τ (minutes)", row=2, col=3)
    fig.update_yaxes(title_text="Mean(y − p)", row=2, col=3)

    fig.update_layout(
        height=700, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=40, b=60),
    )
    st.plotly_chart(fig, width="stretch")

    # Conditional reliability
    if sigma_rel is not None:
        with st.expander("Conditional Reliability (τ × σ_rel)"):
            _render_conditional_reliability(p_pred, y, tau_min, sigma_rel)


# ---------------------------------------------------------------------------
# View 2: Model vs Polymarket
# ---------------------------------------------------------------------------

def render_model_vs_pm_view(model: Model, params: dict, dataset: pd.DataFrame):
    """View 2 — head-to-head Model vs Polymarket comparison."""
    _, _, tau, y, _, p_pred, tau_min, sigma_rel = _dataset_arrays(model, params, dataset)

    ll_baseline = log_loss(y, np.full_like(y, y.mean()))

    has_pm = "pm_mid" in dataset.columns and dataset["pm_mid"].notna().sum() > 100
    if not has_pm:
        st.warning("No Polymarket data in dataset. Click **Recalibrate** to rebuild with PM data.")
        return

    pm_valid = dataset["pm_mid"].notna().values
    p_pm = np.clip(dataset["pm_mid"].values[pm_valid].astype(np.float64), 1e-9, 1.0 - 1e-9)
    y_v = y[pm_valid]
    p_m = p_pred[pm_valid]
    tau_v = tau_min[pm_valid]
    sigma_v = sigma_rel[pm_valid] if sigma_rel is not None else None

    ll_model = log_loss(y_v, p_m)
    ll_pm = log_loss(y_v, p_pm)
    edge = ll_pm - ll_model  # positive = model wins

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model LL", f"{ll_model:.4f}",
              delta=f"{(ll_baseline - ll_model) / ll_baseline * 100:+.1f}% vs baseline")
    c2.metric("Polymarket LL", f"{ll_pm:.4f}",
              delta=f"{(ll_baseline - ll_pm) / ll_baseline * 100:+.1f}% vs baseline")
    c3.metric("Advantage", f"{edge:+.4f} LL",
              delta="Model wins" if edge > 0 else "PM wins",
              delta_color="normal" if edge > 0 else "inverse")
    c4.metric("PM coverage", f"{pm_valid.sum():,} / {len(y):,}")

    # ===== 2x3 comparison grid =====
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Calibration: Model",
            "Calibration: Polymarket",
            "Model P vs PM P",
            "LL by τ (grouped)",
            "LL by σ_rel (grouped)",
            "LL Advantage by τ",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )

    # 1. Calibration curve — Model only
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="black", width=1), showlegend=False,
    ), row=1, col=1)
    bp_m, ba_m, bse_m, _ = equal_mass_reliability(p_m, y_v, n_bins=20)
    fig.add_trace(go.Scatter(
        x=bp_m, y=ba_m, name="Model",
        error_y=dict(type="data", array=1.96 * bse_m, visible=True),
        mode="markers", marker=dict(size=7, color=COLORS["model"]),
    ), row=1, col=1)
    fig.update_xaxes(title_text="Predicted", range=[-0.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Actual", range=[-0.05, 1.05], row=1, col=1)

    # 2. Calibration curve — Polymarket only
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="black", width=1), showlegend=False,
    ), row=1, col=2)
    bp_pm, ba_pm, bse_pm, _ = equal_mass_reliability(p_pm, y_v, n_bins=20)
    fig.add_trace(go.Scatter(
        x=bp_pm, y=ba_pm, name="Polymarket",
        error_y=dict(type="data", array=1.96 * bse_pm, visible=True),
        mode="markers", marker=dict(size=7, color=COLORS["pm_mid"]),
    ), row=1, col=2)
    fig.update_xaxes(title_text="Predicted", range=[-0.05, 1.05], row=1, col=2)
    fig.update_yaxes(title_text="Actual", range=[-0.05, 1.05], row=1, col=2)

    # 3. Scatter: Model P vs PM P (colored by outcome)
    sample_idx = np.random.default_rng(42).choice(len(p_m), size=min(2000, len(p_m)), replace=False)
    fig.add_trace(go.Scatter(
        x=p_pm[sample_idx][y_v[sample_idx] == 1],
        y=p_m[sample_idx][y_v[sample_idx] == 1],
        mode="markers", marker=dict(size=3, color="#2196F3", opacity=0.4),
        name="y=1 (Up)",
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=p_pm[sample_idx][y_v[sample_idx] == 0],
        y=p_m[sample_idx][y_v[sample_idx] == 0],
        mode="markers", marker=dict(size=3, color="#ff5722", opacity=0.4),
        name="y=0 (Down)",
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="black", width=1), showlegend=False,
    ), row=1, col=3)
    fig.update_xaxes(title_text="Polymarket P", range=[-0.05, 1.05], row=1, col=3)
    fig.update_yaxes(title_text="Model P", range=[-0.05, 1.05], row=1, col=3)

    # 4. Grouped LL by tau
    tau_buckets = [0, 5, 15, 30, 60]
    tau_labels = ["0-5", "5-15", "15-30", "30-60"]
    m_tau = _bucket_ll(y_v, p_m, tau_buckets, tau_v)
    pm_tau = _bucket_ll(y_v, p_pm, tau_buckets, tau_v)
    fig.add_trace(go.Bar(
        x=tau_labels, y=[v[0] for v in m_tau], name="Model",
        marker_color=COLORS["model"], opacity=0.85,
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=tau_labels, y=[v[0] for v in pm_tau], name="Polymarket",
        marker_color=COLORS["pm_mid"], opacity=0.85,
    ), row=2, col=1)
    fig.add_hline(y=ll_baseline, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_xaxes(title_text="τ (minutes)", row=2, col=1)
    fig.update_yaxes(title_text="Log Loss", row=2, col=1)

    # 5. Grouped LL by sigma_rel
    if sigma_v is not None:
        rel_buckets = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
        rel_labels = ["0-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3+"]
        m_rel = _bucket_ll(y_v, p_m, rel_buckets, sigma_v)
        pm_rel = _bucket_ll(y_v, p_pm, rel_buckets, sigma_v)
        fig.add_trace(go.Bar(
            x=rel_labels, y=[v[0] for v in m_rel], name="Model",
            marker_color=COLORS["model"], opacity=0.85, showlegend=False,
        ), row=2, col=2)
        fig.add_trace(go.Bar(
            x=rel_labels, y=[v[0] for v in pm_rel], name="Polymarket",
            marker_color=COLORS["pm_mid"], opacity=0.85, showlegend=False,
        ), row=2, col=2)
        fig.add_hline(y=ll_baseline, line_dash="dot", line_color="gray", row=2, col=2)
    fig.update_xaxes(title_text="σ_rel", row=2, col=2)
    fig.update_yaxes(title_text="Log Loss", row=2, col=2)

    # 6. LL advantage by tau (positive green = model wins)
    diff = [(pm_tau[i][0] - m_tau[i][0]) if not (np.isnan(pm_tau[i][0]) or np.isnan(m_tau[i][0])) else 0
            for i in range(len(tau_labels))]
    colors = ["#22c55e" if d > 0 else "#ef4444" for d in diff]
    fig.add_trace(go.Bar(
        x=tau_labels, y=diff, marker_color=colors, showlegend=False,
    ), row=2, col=3)
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=3)
    fig.update_xaxes(title_text="τ (minutes)", row=2, col=3)
    fig.update_yaxes(title_text="LL(PM) − LL(Model)", row=2, col=3)

    fig.update_layout(
        height=700, showlegend=True, barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=40, b=60),
    )
    st.plotly_chart(fig, width="stretch")

    # ===== Hour-of-day breakdown =====
    st.subheader("Log Loss by Hour of Day (ET)")

    # Extract ET hour from market_id (format: ASSET_YYYYMMDD_HH, HH is UTC)
    def _market_id_to_et_hour(mid):
        try:
            parts = mid.split("_")
            utc_h = int(parts[2])
            d = datetime.strptime(parts[1], "%Y%m%d").replace(tzinfo=timezone.utc)
            return d.replace(hour=utc_h).astimezone(ET).hour
        except Exception:
            return -1

    et_hours_all = np.array([_market_id_to_et_hour(m) for m in dataset["market_id"].values])
    et_hours_v = et_hours_all[pm_valid]

    # Compute per-hour LL for both
    hour_labels = [f"{h:02d}" for h in range(24)]
    hour_model_ll, hour_pm_ll, hour_diff, hour_n = [], [], [], []
    for h in range(24):
        hm = et_hours_v == h
        n_h = hm.sum()
        if n_h > 10:
            ml = log_loss(y_v[hm], p_m[hm])
            pl = log_loss(y_v[hm], p_pm[hm])
            hour_model_ll.append(ml)
            hour_pm_ll.append(pl)
            hour_diff.append(pl - ml)  # positive = model wins
            hour_n.append(n_h)
        else:
            hour_model_ll.append(np.nan)
            hour_pm_ll.append(np.nan)
            hour_diff.append(0)
            hour_n.append(n_h)

    fig_tod = make_subplots(
        rows=2, cols=1, row_heights=[0.6, 0.4],
        shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=["LL by Hour (ET)", "Model Advantage (PM LL − Model LL)"],
    )

    # Top: Model vs PM log loss lines
    fig_tod.add_trace(go.Scatter(
        x=hour_labels, y=hour_model_ll, name="Model",
        mode="lines+markers", line=dict(color=COLORS["model"], width=2),
        marker=dict(size=6),
    ), row=1, col=1)
    fig_tod.add_trace(go.Scatter(
        x=hour_labels, y=hour_pm_ll, name="Polymarket",
        mode="lines+markers", line=dict(color=COLORS["pm_mid"], width=2),
        marker=dict(size=6),
    ), row=1, col=1)
    fig_tod.add_hline(y=ll_baseline, line_dash="dot", line_color="gray", row=1, col=1)
    fig_tod.update_yaxes(title_text="Log Loss", row=1, col=1)

    # Bottom: advantage bars (green = model wins, red = PM wins)
    bar_colors = ["#22c55e" if d > 0 else "#ef4444" for d in hour_diff]
    fig_tod.add_trace(go.Bar(
        x=hour_labels, y=hour_diff, marker_color=bar_colors,
        showlegend=False,
        hovertemplate="Hour %{x} ET<br>Advantage: %{y:+.4f}<extra></extra>",
    ), row=2, col=1)
    fig_tod.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig_tod.update_xaxes(title_text="Hour (ET)", row=2, col=1)
    fig_tod.update_yaxes(title_text="LL(PM) − LL(Model)", row=2, col=1)

    fig_tod.update_layout(
        height=450, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig_tod, width="stretch")

    # Per-market breakdown in expander
    with st.expander("Per-market log loss"):
        _render_per_market_ll(dataset, p_pred, pm_valid, p_pm)


def _render_per_market_ll(dataset, p_pred, pm_valid, p_pm):
    """Table of per-market log loss for model vs PM."""
    eps = 1e-9
    y = dataset["y"].values.astype(np.float64)
    mids = dataset["market_id"].values

    # Compute per-row log loss for model
    ll_rows_model = -(y * np.log(np.clip(p_pred, eps, 1 - eps))
                      + (1 - y) * np.log(np.clip(1 - p_pred, eps, 1 - eps)))

    # Per-row PM log loss (NaN where no PM data)
    ll_rows_pm = np.full(len(y), np.nan)
    p_pm_full = np.full(len(y), np.nan)
    p_pm_full[pm_valid] = p_pm
    valid = pm_valid
    ll_rows_pm[valid] = -(y[valid] * np.log(np.clip(p_pm_full[valid], eps, 1 - eps))
                          + (1 - y[valid]) * np.log(np.clip(1 - p_pm_full[valid], eps, 1 - eps)))

    df = pd.DataFrame({
        "market_id": mids,
        "y": y,
        "ll_model": ll_rows_model,
        "ll_pm": ll_rows_pm,
    })
    agg = df.groupby("market_id").agg(
        y=("y", "first"),
        model_ll=("ll_model", "mean"),
        pm_ll=("ll_pm", "mean"),
        n=("ll_model", "count"),
    ).dropna()
    agg["advantage"] = agg["pm_ll"] - agg["model_ll"]

    # Parse UTC hour from market_id (format: ASSET_YYYYMMDD_HH) and convert to ET
    def _utc_to_et_label(mid):
        try:
            parts = mid.split("_")
            utc_h = int(parts[2])
            d = datetime.strptime(parts[1], "%Y%m%d").replace(tzinfo=timezone.utc)
            d = d.replace(hour=utc_h)
            et = d.astimezone(ET)
            return f"{et.strftime('%m/%d %H')}ET"
        except Exception:
            return ""

    agg = agg.reset_index()
    agg["ET"] = agg["market_id"].apply(_utc_to_et_label)
    agg["Outcome"] = agg["y"].map({1: "Up", 0: "Down"})
    agg = agg.sort_values("advantage", ascending=False)
    agg = agg[["market_id", "ET", "Outcome", "model_ll", "pm_ll", "n", "advantage"]]
    agg.columns = ["Market (UTC)", "ET", "Outcome", "Model LL", "PM LL", "N", "Advantage"]
    st.dataframe(agg.style.format({
        "Model LL": "{:.4f}", "PM LL": "{:.4f}", "Advantage": "{:+.4f}",
    }), width="stretch", hide_index=True)


def _render_conditional_reliability(p_pred, y, tau_min, sigma_rel):
    """4x4 conditional reliability grid."""
    tau_edges = [0, 5, 15, 30, 60]
    tau_labels = ["τ: 0-5m", "τ: 5-15m", "τ: 15-30m", "τ: 30-60m"]
    vol_edges = list(np.quantile(sigma_rel, [0.0, 0.25, 0.5, 0.75, 1.0]))
    vol_labels = [f"Q{i+1}" for i in range(4)]

    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=[f"{tl} / {vl}" for tl in tau_labels for vl in vol_labels],
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
    )

    for ti in range(4):
        tau_mask = (tau_min >= tau_edges[ti]) & (tau_min < tau_edges[ti + 1])
        for qi in range(4):
            v_lo, v_hi = vol_edges[qi], vol_edges[qi + 1]
            vol_mask = (sigma_rel >= v_lo) if qi == 3 else (sigma_rel >= v_lo) & (sigma_rel < v_hi)
            cell = tau_mask & vol_mask
            n = cell.sum()

            row, col = ti + 1, qi + 1

            # Diagonal
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(dash="dash", color="gray", width=1),
                showlegend=False,
            ), row=row, col=col)

            if n >= 30:
                n_bins_cell = max(8, min(15, n // 50))
                bp_c, ba_c, bse_c, _ = equal_mass_reliability(p_pred[cell], y[cell], n_bins=n_bins_cell)
                cell_ll = log_loss(y[cell], p_pred[cell])
                cell_e = np.mean(y[cell] - p_pred[cell])
                fig.add_trace(go.Scatter(
                    x=bp_c, y=ba_c,
                    error_y=dict(type="data", array=1.96 * bse_c, visible=True),
                    mode="markers", marker=dict(size=4, color="#2196F3"),
                    showlegend=False,
                    hovertemplate=f"LL={cell_ll:.3f} e={cell_e:+.3f} n={n}<extra></extra>",
                ), row=row, col=col)

            fig.update_xaxes(range=[-0.05, 1.05], row=row, col=col, showticklabels=(ti == 3))
            fig.update_yaxes(range=[-0.05, 1.05], row=row, col=col, showticklabels=(qi == 0))

    fig.update_layout(height=900, margin=dict(l=40, r=20, t=30, b=30))
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Single market view
# ---------------------------------------------------------------------------

def render_single_market_view(model: Model, params: dict, data: dict):
    """Render 3-panel single market chart with model vs polymarket."""
    K = data["K"]
    S_T = data["S_T"]
    Y = data["Y"]
    outcome = "Up" if Y == 1 else "Down"
    ret = (S_T - K) / K * 100

    # Metrics
    cols = st.columns(5)
    cols[0].metric("Strike (K)", f"${K:,.2f}")
    cols[1].metric("Close (S_T)", f"${S_T:,.2f}")
    cols[2].metric("Return", f"{ret:+.4f}%")
    cols[3].metric("Outcome", outcome)
    cols[4].metric("Data points", f"{len(data['S_arr']):,}")

    # Model predictions with current params
    model_features = {f: data["features"][f] for f in model.required_features() if f in data["features"]}
    p_model = model.predict(params, data["S_arr"], data["K_arr"], data["tau_arr"], model_features)

    times = _ts_to_dt(data["ts_market"])
    mid = data["mid_market"]

    has_pm = data["pm_ts"] is not None and data["pm_mid"] is not None

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.4, 0.35, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=["Binance Price", "P(Up): Model vs Polymarket", "Edge (Model − Polymarket)"],
    )

    # Panel 1: Price
    fig.add_trace(go.Scatter(
        x=times, y=mid, name="Binance Mid",
        line=dict(color=COLORS["bnc_mid"], width=1.5),
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=K, line_dash="dash", line_color=COLORS["K"], line_width=1.5,
                  annotation_text=f"K = ${K:,.2f}", annotation_position="top left",
                  row=1, col=1)
    fig.add_hline(y=S_T, line_dash="dot", line_color=COLORS["S_T"], line_width=1.5,
                  annotation_text=f"S_T = ${S_T:,.2f}", annotation_position="bottom left",
                  row=1, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)

    # Panel 2: P(Up)
    fig.add_trace(go.Scatter(
        x=times, y=p_model, name=f"Model ({model.name})",
        line=dict(color=COLORS["model"], width=2),
        hovertemplate="Model: %{y:.3f}<extra></extra>",
    ), row=2, col=1)

    if has_pm:
        pm_times = _ts_to_dt(data["pm_ts"])
        pm_mid = data["pm_mid"]
        fig.add_trace(go.Scatter(
            x=pm_times, y=pm_mid, name="Polymarket",
            line=dict(color=COLORS["pm_mid"], width=1.5),
            hovertemplate="PM: %{y:.3f}<extra></extra>",
        ), row=2, col=1)

    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", line_width=0.5, row=2, col=1)
    fig.update_yaxes(title_text="P(Up)", range=[-0.05, 1.05], row=2, col=1)

    # Panel 3: Edge
    if has_pm:
        pm_ts = data["pm_ts"]
        ts_market = data["ts_market"]
        idx_align = np.searchsorted(ts_market, pm_ts)
        idx_align = np.clip(idx_align, 0, len(p_model) - 1)
        edge = p_model[idx_align] - data["pm_mid"]
        pm_dt = _ts_to_dt(pm_ts)

        # Positive edge (green fill)
        edge_pos = np.where(edge > 0, edge, 0)
        edge_neg = np.where(edge < 0, edge, 0)

        fig.add_trace(go.Scatter(
            x=pm_dt, y=edge_pos, name="Model > PM",
            fill="tozeroy", fillcolor=COLORS["edge_pos"],
            line=dict(color="rgba(34,197,94,0.5)", width=0),
            hovertemplate="Edge: %{y:+.3f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=pm_dt, y=edge_neg, name="Model < PM",
            fill="tozeroy", fillcolor=COLORS["edge_neg"],
            line=dict(color="rgba(239,68,68,0.5)", width=0),
            hovertemplate="Edge: %{y:+.3f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=pm_dt, y=edge, showlegend=False,
            line=dict(color="black", width=0.8),
        ), row=3, col=1)
    else:
        fig.add_annotation(
            text="No Polymarket data", xref="x3 domain", yref="y3 domain",
            x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="gray"),
        )

    fig.add_hline(y=0, line_color="black", line_width=0.5, row=3, col=1)
    fig.update_yaxes(title_text="Edge", row=3, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)

    fig.update_layout(
        height=800,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# STREAMLIT APP
# ===========================================================================

st.set_page_config(page_title="Pricing Dashboard", layout="wide")
st.title("Binary Option Pricing Dashboard")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Adaptive-t Model")
    model_name = "gaussian_t"
    model = get_model(model_name)

    # Load cached params
    cached_params = load_cached_params(model_name)
    has_params = cached_params is not None

    if has_params:
        neg_t_ll = cached_params.get("neg_t_ll", 0)
        nu_med = cached_params.get("nu_median", 0)
        st.success(f"Calibrated: neg_t_LL={neg_t_ll:.3f}, nu_med={nu_med:.1f}")
    else:
        st.warning("Not calibrated yet")

    if st.button("Calibrate" if not has_params else "Recalibrate"):
        dataset = load_cached_dataset()
        if dataset.empty:
            with st.spinner("Building calibration dataset..."):
                status = get_cache_status("binance", "BTC", "1s")
                available = status.get("available_dates", [])
                if available:
                    cfg = DatasetConfig(
                        start_date=date.fromisoformat(available[0]),
                        end_date=date.fromisoformat(available[-1]),
                    )
                    dataset = build_dataset(cfg)
                    load_cached_dataset.clear()
                else:
                    st.error("No cached 1s data. Cannot build dataset.")
                    st.stop()
        with st.spinner("Stage 1: Calibrating volatility (QLIKE)..."):
            model_gauss = get_model("gaussian")
            calibrate_vol(model_gauss, dataset, output_dir=str(OUTPUT_DIR), verbose=False)
        with st.spinner("Stage 2: Calibrating tails (Student-t LL)..."):
            model_t = get_model("gaussian_t")
            calibrate_tail(model_t, dataset, output_dir=str(OUTPUT_DIR), verbose=False)
            st.rerun()

    st.divider()
    view_mode = st.radio("View", ["Calibration", "Model vs PM", "Single Market"], horizontal=True)

    # Single market controls
    asset = "BTC"
    selected_date = None
    hour_et = 10
    if view_mode == "Single Market":
        st.divider()
        st.subheader("Market")
        asset = st.selectbox("Asset", ["BTC", "ETH"])
        available_dates = _available_dates(asset)
        if available_dates:
            selected_date = st.date_input(
                "Date", value=available_dates[-1],
                min_value=available_dates[0], max_value=available_dates[-1],
            )
            hour_et = st.slider("Hour (ET)", 0, 23, 10)
        else:
            st.warning(f"No Polymarket cache for {asset}")

    # Parameter sliders
    slider_params = None
    if has_params:
        st.divider()
        st.subheader("Parameters")

        calibrated = {k: cached_params[k] for k in model.param_names() if k in cached_params}
        bounds = model.param_bounds()

        # Initialize session state on first load
        if not st.session_state.get("_params_initialized"):
            st.session_state["_params_initialized"] = True
            for pname in model.param_names():
                st.session_state[f"p_{pname}"] = calibrated.get(pname, model.initial_params()[pname])

        # Reset must happen BEFORE sliders are created (can't modify widget keys after)
        if st.session_state.get("_reset_params"):
            st.session_state["_reset_params"] = False
            for pname in model.param_names():
                st.session_state[f"p_{pname}"] = calibrated.get(pname, model.initial_params()[pname])

        slider_params = {}
        for pname in model.param_names():
            lo, hi = bounds[pname]
            step = (hi - lo) / 500
            slider_params[pname] = st.slider(
                pname,
                min_value=float(lo),
                max_value=float(hi),
                step=step,
                format="%.4f",
                key=f"p_{pname}",
            )

        params_changed = any(
            abs(slider_params[p] - calibrated.get(p, 0)) > 1e-8
            for p in model.param_names()
        )

        if params_changed:
            st.warning("Modified from calibrated")

        if st.button("Reset to calibrated"):
            st.session_state["_reset_params"] = True
            st.rerun()

        # Live LL on dataset when params differ
        if params_changed:
            dataset = load_cached_dataset()
            if not dataset.empty:
                S_ds = dataset["S"].values.astype(np.float64)
                K_ds = dataset["K"].values.astype(np.float64)
                tau_ds = dataset["tau"].values.astype(np.float64)
                y_ds = dataset["y"].values.astype(np.float64)
                feats_ds = {f: dataset[f].values.astype(np.float64) for f in model.required_features()}
                p_ds = model.predict(slider_params, S_ds, K_ds, tau_ds, feats_ds)
                ll_slider = log_loss(y_ds, p_ds)
                # Compare against calibrated params
                p_cal = model.predict(calibrated, S_ds, K_ds, tau_ds, feats_ds)
                ll_cal = log_loss(y_ds, p_cal)
                st.metric("Dataset LL", f"{ll_slider:.4f}", delta=f"{ll_slider - ll_cal:+.4f}", delta_color="inverse")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
active_params = slider_params if slider_params else (
    {k: cached_params[k] for k in model.param_names()} if has_params else None
)

if view_mode == "Calibration":
    if not has_params:
        st.info("No calibrated parameters. Click **Calibrate** in the sidebar.")
    else:
        dataset = load_cached_dataset()
        if dataset.empty:
            st.warning("No cached dataset. Click **Calibrate** to build one.")
        else:
            render_calibration_view(model, active_params, dataset)

elif view_mode == "Model vs PM":
    if not has_params:
        st.info("No calibrated parameters. Click **Calibrate** in the sidebar.")
    else:
        dataset = load_cached_dataset()
        if dataset.empty:
            st.warning("No cached dataset. Click **Calibrate** to build one.")
        else:
            render_model_vs_pm_view(model, active_params, dataset)

elif view_mode == "Single Market":
    if active_params is None:
        st.info("No parameters available. Click **Calibrate** first.")
    elif selected_date is None:
        st.warning("No dates available for the selected asset.")
    else:
        utc_start, utc_end = _et_hour_to_utc(selected_date, hour_et)
        st.caption(f"{hour_et:02d}:00 ET → {utc_start.strftime('%Y-%m-%d %H:%M')}–{utc_end.strftime('%H:%M')} UTC")
        with st.spinner("Loading market data..."):
            data = load_single_market_data(asset, str(selected_date), hour_et)
        if isinstance(data, str):
            st.error(data)
        elif data is None:
            st.error(f"No data for {asset} {selected_date} hour={hour_et} ET")
        else:
            render_single_market_view(model, active_params, data)
