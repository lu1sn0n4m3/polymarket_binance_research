"""Single-market visualization: model predictions vs Polymarket on one hourly market.

Three-panel chart:
    Top:    Binance price with K (strike) and S_T (close) marked
    Middle: P(Up) -- model prediction vs Polymarket mid
    Bottom: Edge (model - polymarket)
"""

from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pricing.models.base import Model
from pricing.features.seasonal_vol import SeasonalVolCurve
from pricing.features.realized_vol import compute_rv_ewma


def plot_market(
    model: Model,
    params: dict[str, float],
    asset: str,
    market_date: date | str,
    hour_utc: int,
    seasonal: SeasonalVolCurve | None = None,
    ewma_half_life_sec: float = 300.0,
    output_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    """Visualize model predictions on a single hourly market.

    Args:
        model: Model instance.
        params: Fitted parameters dict.
        asset: "BTC" or "ETH".
        market_date: Date of the market.
        hour_utc: UTC hour (0-23).
        seasonal: Precomputed SeasonalVolCurve. If None, computes from lookback data.
        ewma_half_life_sec: EWMA half-life for sigma_rv.
        output_path: Save figure to this path (None = don't save).
        show: Call plt.show() (set False for batch usage).

    Returns:
        matplotlib Figure.
    """
    from src.data import load_binance, load_binance_labels, load_polymarket_market
    from pricing.features.seasonal_vol import compute_seasonal_vol

    if isinstance(market_date, str):
        market_date = date.fromisoformat(market_date)

    # Market boundaries (UTC)
    hour_start = datetime(market_date.year, market_date.month, market_date.day,
                          hour_utc, 0, 0, tzinfo=timezone.utc)
    hour_end = datetime(market_date.year, market_date.month, market_date.day,
                        hour_utc + 1 if hour_utc < 23 else 0, 0, 0, tzinfo=timezone.utc)
    if hour_utc == 23:
        from datetime import timedelta
        hour_end = hour_start + timedelta(hours=1)

    hour_start_ms = int(hour_start.timestamp() * 1000)
    hour_end_ms = int(hour_end.timestamp() * 1000)

    # Load labels for this hour
    labels = load_binance_labels(
        start=hour_start.strftime("%Y-%m-%d %H:%M:%S"),
        end=hour_end.strftime("%Y-%m-%d %H:%M:%S"),
        asset=asset,
    )
    if labels.empty:
        raise ValueError(f"No labels found for {asset} {market_date} hour={hour_utc}")

    lbl = labels.iloc[0]
    K = float(lbl["K"])
    S_T = float(lbl["S_T"])
    Y = int(lbl["Y"])
    outcome = "Up" if Y == 1 else "Down"

    # Load Binance 1s data (with lookback for vol estimation)
    lookback_hours = 3
    from datetime import timedelta
    lookback_start = hour_start - timedelta(hours=lookback_hours)

    bnc = load_binance(
        start=lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
        end=hour_end.strftime("%Y-%m-%d %H:%M:%S"),
        asset=asset,
        interval="1s",
    )

    if bnc.empty:
        raise ValueError(f"No Binance data for {asset} {market_date} hour={hour_utc}")

    # Compute seasonal vol if not provided
    if seasonal is None:
        seasonal = compute_seasonal_vol(bnc)

    # Compute sigma_rv on the full (lookback + market) data
    ts_rv, sigma_rv_full = compute_rv_ewma(bnc, half_life_sec=ewma_half_life_sec)

    # Filter to market hour only
    ts_bnc = bnc["ts_recv"].values
    mid_bnc = bnc["mid_px"].values
    in_market = (ts_bnc >= hour_start_ms) & (ts_bnc < hour_end_ms)
    ts_market = ts_bnc[in_market]
    mid_market = mid_bnc[in_market]

    if len(ts_market) == 0:
        raise ValueError("No data in market hour")

    # Map features to market timestamps
    sigma_tod_market = seasonal.lookup_array(ts_market)

    idx_rv = np.searchsorted(ts_rv, ts_market)
    idx_rv = np.clip(idx_rv, 0, len(ts_rv) - 1)
    idx_prev = np.clip(idx_rv - 1, 0, len(ts_rv) - 1)
    dist_right = np.abs(ts_rv[idx_rv] - ts_market)
    dist_left = np.abs(ts_rv[idx_prev] - ts_market)
    best_idx = np.where(dist_left < dist_right, idx_prev, idx_rv)
    sigma_rv_market = sigma_rv_full[best_idx]
    sigma_rel_market = sigma_rv_market / np.maximum(sigma_tod_market, 1e-12)

    tau_market = (hour_end_ms - ts_market) / 1000.0  # seconds to expiry

    # Run model predictions
    S_arr = mid_market.astype(np.float64)
    K_arr = np.full_like(S_arr, K)
    features = {
        "sigma_tod": sigma_tod_market,
        "sigma_rv": sigma_rv_market,
        "sigma_rel": sigma_rel_market,
    }
    # Only pass features the model needs
    model_features = {f: features[f] for f in model.required_features() if f in features}
    p_model = model.predict(params, S_arr, K_arr, tau_market, model_features)

    # Convert timestamps to datetime for plotting
    times = pd.to_datetime(ts_market, unit="ms", utc=True)

    # Try loading Polymarket data
    pm_times, pm_mid = None, None
    try:
        # Convert UTC hour to ET hour for load_polymarket_market
        # ET is UTC-5 in winter, UTC-4 in summer
        # For simplicity, try both offsets
        pm = None
        for offset in [5, 4]:
            hour_et = (hour_utc - offset) % 24
            try:
                pm = load_polymarket_market(asset, str(market_date), hour_et, "1s")
                if not pm.empty:
                    break
            except Exception:
                continue

        if pm is not None and not pm.empty:
            pm_in_market = (pm["ts_recv"].values >= hour_start_ms) & (pm["ts_recv"].values < hour_end_ms)
            if pm_in_market.sum() > 0:
                pm_times = pd.to_datetime(pm["ts_recv"].values[pm_in_market], unit="ms", utc=True)
                pm_mid = pm["mid"].values[pm_in_market]
    except Exception:
        pass  # Polymarket data optional

    # ========== Plot ==========
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                                         gridspec_kw={"height_ratios": [2, 2, 1]})
    fig.suptitle(
        f"{asset} {market_date} {hour_utc:02d}:00 UTC  |  Outcome: {outcome}  "
        f"|  K={K:.2f}  S_T={S_T:.2f}",
        fontsize=13,
    )

    # Panel 1: Binance price
    ax1.plot(times, mid_market, color="#2196F3", lw=0.8, label="Binance mid")
    ax1.axhline(K, color="#ff6b00", linestyle="--", lw=1.5, label=f"K = {K:.2f}")
    ax1.axhline(S_T, color="#a855f7", linestyle="--", lw=1.5, label=f"S_T = {S_T:.2f}")
    ax1.set_ylabel("Price")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: P(Up) comparison
    ax2.plot(times, p_model, color="#E91E63", lw=1.0, label=f"Model ({model.name})")
    if pm_times is not None and pm_mid is not None:
        ax2.plot(pm_times, pm_mid, color="#4CAF50", lw=0.8, alpha=0.7, label="Polymarket")
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.set_ylabel("P(Up)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Edge (if PM data available)
    if pm_times is not None and pm_mid is not None:
        # Align model predictions to PM timestamps via nearest neighbor
        pm_ts = pm["ts_recv"].values[pm_in_market]
        idx_align = np.searchsorted(ts_market, pm_ts)
        idx_align = np.clip(idx_align, 0, len(p_model) - 1)
        edge = p_model[idx_align] - pm_mid
        ax3.fill_between(pm_times, edge, 0, where=edge > 0, color="#4CAF50", alpha=0.3, label="Model > PM")
        ax3.fill_between(pm_times, edge, 0, where=edge < 0, color="#E91E63", alpha=0.3, label="Model < PM")
        ax3.plot(pm_times, edge, color="black", lw=0.5, alpha=0.5)
        ax3.axhline(0, color="black", linestyle="-", lw=0.5)
        ax3.set_ylabel("Edge")
        ax3.legend(loc="upper left", fontsize=8)
    else:
        ax3.text(0.5, 0.5, "No Polymarket data", ha="center", va="center",
                 transform=ax3.transAxes)
        ax3.set_ylabel("Edge")
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax3.set_xlabel("Time (UTC)")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved market view to {output_path}")

    if show:
        plt.show()

    return fig
