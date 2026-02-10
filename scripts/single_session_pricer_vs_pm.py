#!/usr/bin/env python3
"""
Plot pricer probabilities vs Polymarket mid for a single 1-hour session.

Prices every 10 seconds using the calibrated model and overlays on
the Polymarket midprice timeline.
"""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm

from src.data import load_session

# =============================================================================
# CONFIG
# =============================================================================

ASSET = "BTC"
PRICER_OUTPUT = Path("pricer_calibration/output")
SCRIPT_OUTPUT = Path("outputs")

# Pick a session — change these to explore different hours
SESSION_DATE = date(2026, 1, 28)   # market date (ET calendar date)
SESSION_HOUR_ET = 11               # hour in Eastern Time

ET = ZoneInfo("America/New_York")
UTC = timezone.utc

SAMPLE_INTERVAL_SEC = 1  # price every 1s


# =============================================================================
# Model (same as polymarket_vs_pricer.py)
# =============================================================================

def apply_model(S, K, tau, sigma_tod, sigma_rv, params, use_probit=True):
    """Apply fitted model to get P(Up)."""
    a0, a1, beta = params["a0"], params["a1"], params["beta"]

    sigma_rel = sigma_rv / np.maximum(sigma_tod, 1e-12)
    tau_min = tau / 60.0
    a_tau = a0 + a1 * np.sqrt(np.maximum(tau_min, 0))
    sigma_eff = a_tau * sigma_tod * np.power(np.maximum(sigma_rel, 1e-12), beta)

    sqrt_tau = np.sqrt(np.maximum(tau, 1e-6))
    z = (np.log(K / S) + 0.5 * sigma_eff**2 * tau) / (sigma_eff * sqrt_tau)

    if use_probit:
        probit = params["probit_layer"]
        score = -z
        log_tau = np.log(np.maximum(tau, 1.0))
        log_sr = np.log(np.maximum(sigma_rel, 1e-6))
        zp = (probit["b0"] + probit["b1"] * score + probit["b2"] * log_tau
              + probit["b3"] * log_sr + probit["b4"] * log_sr**2)
        return np.clip(norm.cdf(zp), 1e-9, 1 - 1e-9)
    else:
        return np.clip(1 - norm.cdf(z), 1e-9, 1 - 1e-9)


# =============================================================================
# Infrastructure (reused from polymarket_vs_pricer.py)
# =============================================================================

def load_seasonal():
    with open(PRICER_OUTPUT / "seasonal_curve.json") as f:
        data = json.load(f)
    return np.array(data["sigma_tod"]), data["bucket_minutes"]


def load_ewma_rv():
    rv_df = pd.read_parquet(PRICER_OUTPUT / "sigma_rv_cache.parquet")
    return rv_df["ts"].values, rv_df["sigma_rv"].values


def sigma_tod_at_ms(ts_ms, sigma_tod_curve, bucket_minutes=5):
    total_sec = (ts_ms // 1000) % 86400
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    buckets = ((hours * 60 + minutes) // bucket_minutes).astype(int)
    buckets = np.clip(buckets, 0, len(sigma_tod_curve) - 1)
    return sigma_tod_curve[buckets]


def lookup_sigma_rv(ts_ms, ts_rv, sigma_rv_full):
    idx = np.searchsorted(ts_rv, ts_ms)
    idx = np.clip(idx, 0, len(ts_rv) - 1)
    idx_prev = np.clip(idx - 1, 0, len(ts_rv) - 1)
    dist_r = np.abs(ts_rv[idx] - ts_ms)
    dist_l = np.abs(ts_rv[idx_prev] - ts_ms)
    best = np.where(dist_l < dist_r, idx_prev, idx)
    return sigma_rv_full[best]


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print(f"Single Session: {ASSET} | {SESSION_DATE} {SESSION_HOUR_ET}:00 ET")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load session
    # ------------------------------------------------------------------
    session = load_session(ASSET, SESSION_DATE, hour_et=SESSION_HOUR_ET,
                           lookback_hours=0)

    if session.outcome is None:
        print("No outcome for this session — market may not have resolved.")
        return

    utc_start = session.utc_start
    utc_end = session.utc_end
    start_ms = int(utc_start.timestamp() * 1000)
    end_ms = int(utc_end.timestamp() * 1000)

    print(f"  UTC window : {utc_start} → {utc_end}")
    print(f"  Outcome    : {'Up' if session.outcome.is_up else 'Down'} "
          f"({session.outcome.return_pct:+.2f}%)")
    print(f"  Strike (K) : {session.outcome.open_price:.2f}")
    K = session.outcome.open_price

    # ------------------------------------------------------------------
    # Polymarket mid
    # ------------------------------------------------------------------
    pm_bbo = session.polymarket_bbo
    if pm_bbo is None or pm_bbo.empty:
        print("No Polymarket BBO data for this session.")
        return

    pm_bbo = pm_bbo.copy()
    pm_bbo["mid"] = (pm_bbo["bid_px"] + pm_bbo["ask_px"]) / 2
    if session.token_is_up is False:
        pm_bbo["mid"] = 1.0 - pm_bbo["mid"]

    # Filter to market window
    pm_bbo = pm_bbo[(pm_bbo["ts_recv"] >= start_ms) &
                    (pm_bbo["ts_recv"] <= end_ms)].copy()
    pm_bbo["dt_utc"] = pd.to_datetime(pm_bbo["ts_recv"], unit="ms", utc=True)
    print(f"  PM BBO rows in window: {len(pm_bbo):,}")

    # ------------------------------------------------------------------
    # Build 10-second pricing grid
    # ------------------------------------------------------------------
    grid_ms = np.arange(start_ms, end_ms, SAMPLE_INTERVAL_SEC * 1000)
    print(f"  Pricing grid: {len(grid_ms)} points ({SAMPLE_INTERVAL_SEC}s)")

    # Get Binance mid at each grid point
    binance_bbo = session.binance_bbo
    if binance_bbo is None or binance_bbo.empty:
        print("No Binance BBO data.")
        return

    binance_bbo = binance_bbo.copy()
    binance_bbo["mid"] = (binance_bbo["bid_px"] + binance_bbo["ask_px"]) / 2

    # Forward-fill Binance mid to grid
    b_ts = binance_bbo["ts_recv"].values
    b_mid = binance_bbo["mid"].values
    idx = np.searchsorted(b_ts, grid_ms, side="right") - 1
    idx = np.clip(idx, 0, len(b_ts) - 1)
    S_grid = b_mid[idx]

    # tau at each grid point (seconds to expiry)
    tau_grid = (end_ms - grid_ms) / 1000.0

    # ------------------------------------------------------------------
    # Load vol infrastructure
    # ------------------------------------------------------------------
    sigma_tod_curve, tod_bucket_min = load_seasonal()
    ts_rv, sigma_rv_full = load_ewma_rv()

    sigma_tod_grid = sigma_tod_at_ms(grid_ms, sigma_tod_curve, tod_bucket_min)
    sigma_rv_grid = lookup_sigma_rv(grid_ms, ts_rv, sigma_rv_full)

    # ------------------------------------------------------------------
    # Load model params
    # ------------------------------------------------------------------
    with open(PRICER_OUTPUT / "params_final.json") as f:
        params = json.load(f)
    print(f"  Model: {params['model']}")

    # ------------------------------------------------------------------
    # Compute pricer probabilities
    # ------------------------------------------------------------------
    model_p = apply_model(S_grid, K, tau_grid, sigma_tod_grid,
                          sigma_rv_grid, params, use_probit=True)
    raw_p = apply_model(S_grid, K, tau_grid, sigma_tod_grid,
                        sigma_rv_grid, params, use_probit=False)

    grid_dt = pd.to_datetime(grid_ms, unit="ms", utc=True)

    # ------------------------------------------------------------------
    # Also get PM mid at each grid point (for direct comparison)
    # ------------------------------------------------------------------
    pm_ts = pm_bbo["ts_recv"].values
    pm_mids = pm_bbo["mid"].values
    pm_idx = np.searchsorted(pm_ts, grid_ms, side="right") - 1
    pm_idx = np.clip(pm_idx, 0, len(pm_ts) - 1)
    pm_at_grid = pm_mids[pm_idx]
    # Mark points where PM data is stale (>30s gap)
    pm_dist = np.abs(pm_ts[pm_idx] - grid_ms)
    pm_at_grid[pm_dist > 30_000] = np.nan

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    print("\nCreating chart...")
    SCRIPT_OUTPUT.mkdir(exist_ok=True)

    fig = go.Figure()

    # Polymarket raw ticks
    fig.add_trace(go.Scatter(
        x=pm_bbo["dt_utc"], y=pm_bbo["mid"],
        mode="lines", name="Polymarket Mid",
        line=dict(color="#2196F3", width=1.5),
        opacity=0.6,
    ))

    # Pricer with probit
    fig.add_trace(go.Scatter(
        x=grid_dt, y=model_p,
        mode="lines", name="Pricer (with probit)",
        line=dict(color="#FF5722", width=2),
    ))

    # Pricer raw (no probit)
    fig.add_trace(go.Scatter(
        x=grid_dt, y=raw_p,
        mode="lines", name="Pricer (raw, no probit)",
        line=dict(color="#4CAF50", width=2, dash="dot"),
    ))

    # Outcome line
    outcome_y = 1.0 if session.outcome.is_up else 0.0
    fig.add_hline(y=outcome_y, line_dash="dot", line_color="green",
                  annotation_text=f"Outcome: {'Up' if session.outcome.is_up else 'Down'}")

    # Strike reference
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.3)

    # ET labels for x-axis
    et_start = utc_start.astimezone(ET)
    title_str = (
        f"{ASSET} | {et_start.strftime('%Y-%m-%d %H:%M ET')} "
        f"| K={K:,.2f} | Outcome: {'Up' if session.outcome.is_up else 'Down'} "
        f"({session.outcome.return_pct:+.2f}%)"
    )

    fig.update_layout(
        title=title_str,
        xaxis_title="Time (UTC)",
        yaxis_title="P(Up)",
        yaxis=dict(range=[-0.02, 1.02]),
        height=500, width=1100,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        hovermode="x unified",
    )

    html_path = SCRIPT_OUTPUT / f"session_{ASSET}_{SESSION_DATE}_{SESSION_HOUR_ET:02d}ET.html"
    fig.write_html(str(html_path))
    print(f"  Saved: {html_path}")

    # ------------------------------------------------------------------
    # Quick stats
    # ------------------------------------------------------------------
    valid = ~np.isnan(pm_at_grid)
    if valid.sum() > 0:
        diff = model_p[valid] - pm_at_grid[valid]
        print(f"\n  Model vs PM (at grid points):")
        print(f"    Mean diff : {np.mean(diff):+.4f}")
        print(f"    Std diff  : {np.std(diff):.4f}")
        print(f"    Max |diff|: {np.max(np.abs(diff)):.4f}")
        corr = np.corrcoef(model_p[valid], pm_at_grid[valid])[0, 1]
        print(f"    Correlation: {corr:.4f}")

    try:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
    except Exception:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
