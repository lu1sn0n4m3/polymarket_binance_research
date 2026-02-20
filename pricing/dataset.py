"""Build calibration dataset from 1s Binance data + hourly labels.

Pipeline:
    1. load_binance(interval="1s") -> BBO DataFrame
    2. load_binance_labels() -> hourly labels (K, S_T, Y)
    3. compute_seasonal_vol(bbo) -> SeasonalVolCurve
    4. compute_rv_ewma(bbo) -> sigma_rv time series
    5. For each hour: sample at sample_interval_sec, attach features + label
    6. Output: flat DataFrame ready for calibration
"""

from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pricing.features.seasonal_vol import SeasonalVolCurve, compute_seasonal_vol, compute_seasonal_vol_split
from pricing.features.realized_vol import compute_rv_ewma

_ET = ZoneInfo("America/New_York")


@dataclass
class DatasetConfig:
    """Configuration for building a calibration dataset."""

    start_date: date
    end_date: date
    asset: str = "BTC"
    sample_interval_sec: float = 60.0
    tod_bucket_minutes: int = 5
    tod_smoothing_window: int = 3
    sigma_tod_floor: float = 1e-10
    ewma_half_life_sec: float = 600.0
    start_hour: int = 0
    end_hour: int = 23
    output_dir: str = "pricing/output"


def build_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    """Build calibration dataset from 1s Binance data.

    Returns DataFrame with columns:
        market_id, t, S, K, tau, y, sigma_tod, sigma_rv, sigma_rel,
        time_since_move, hour_et

    The dataset is cached as parquet in cfg.output_dir.
    """
    from marketdata.data import load_binance, load_binance_labels

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.combine(cfg.start_date, dt_time(cfg.start_hour), tzinfo=timezone.utc)
    end_dt = datetime.combine(cfg.end_date, dt_time(cfg.end_hour), tzinfo=timezone.utc)

    # Step 1: Load 1s BBO data
    print(f"Loading Binance BBO {cfg.start_date} to {cfg.end_date} ({cfg.asset}, 1s) ...")
    bbo = load_binance(start=start_dt, end=end_dt, asset=cfg.asset, interval="1s")
    print(f"  {len(bbo):,} BBO rows")

    if bbo.empty:
        print("No BBO data loaded.")
        return pd.DataFrame()

    # Step 2: Load hourly labels
    print("Loading hourly labels ...")
    labels = load_binance_labels(start=start_dt, end=end_dt, asset=cfg.asset)
    print(f"  {len(labels)} market hours labeled")

    if labels.empty:
        print("No labels found.")
        return pd.DataFrame()

    # Step 3: Compute seasonal volatility (weekday/weekend split)
    print("Computing seasonal volatility (weekday/weekend split) ...")
    seasonal = compute_seasonal_vol_split(
        bbo,
        bucket_minutes=cfg.tod_bucket_minutes,
        smoothing_window=cfg.tod_smoothing_window,
        floor=cfg.sigma_tod_floor,
    )
    print(f"  {seasonal.n_buckets} TOD buckets")

    # Save seasonal vol curves for downstream use (pricer)
    for label, curve in [("weekday", seasonal.weekday), ("weekend", seasonal.weekend)]:
        df_sv = pd.DataFrame({
            "bucket": np.arange(curve.n_buckets),
            "sigma_tod": curve.sigma_tod,
        })
        df_sv.to_parquet(output_dir / f"seasonal_vol_{label}.parquet", index=False)

    # Step 4: Compute EWMA realized vol
    print(f"Computing EWMA sigma_rv (H={cfg.ewma_half_life_sec}s) ...")
    ts_rv, sigma_rv_full = compute_rv_ewma(
        bbo, half_life_sec=cfg.ewma_half_life_sec,
    )
    print(f"  {len(ts_rv):,} tick-time sigma_rv values")

    if len(ts_rv) == 0:
        print("No sigma_rv computed.")
        return pd.DataFrame()

    # Step 5: Sample calibration rows
    print(f"Sampling calibration rows (every {cfg.sample_interval_sec}s) ...")
    ts_bbo = bbo["ts_recv"].values
    mid_bbo = bbo["mid_px"].values

    # Precompute sigma_tod for all BBO timestamps
    sigma_tod_all = seasonal.lookup_array(ts_bbo)

    # Precompute sigma_rv for all BBO timestamps (nearest-neighbor lookup)
    idx_rv = np.searchsorted(ts_rv, ts_bbo)
    idx_rv = np.clip(idx_rv, 0, len(ts_rv) - 1)
    idx_prev = np.clip(idx_rv - 1, 0, len(ts_rv) - 1)
    dist_right = np.abs(ts_rv[idx_rv] - ts_bbo)
    dist_left = np.abs(ts_rv[idx_prev] - ts_bbo)
    best_idx = np.where(dist_left < dist_right, idx_prev, idx_rv)
    sigma_rv_all = sigma_rv_full[best_idx]

    # sigma_rel
    sigma_rel_all = sigma_rv_all / np.maximum(sigma_tod_all, 1e-12)

    # Precompute time-since-last-move for all BBO rows
    _valid_mid = ~np.isnan(mid_bbo)
    _changed = np.zeros(len(mid_bbo), dtype=bool)
    _changed[0] = True
    _changed[1:] = (mid_bbo[1:] != mid_bbo[:-1]) & _valid_mid[1:] & _valid_mid[:-1]
    _change_times_ms = ts_bbo[_changed]

    _si = np.searchsorted(_change_times_ms, ts_bbo, side="right") - 1
    _si = np.clip(_si, 0, len(_change_times_ms) - 1)
    time_since_move_all = np.maximum((ts_bbo - _change_times_ms[_si]) / 1000.0, 0.0)

    # Build market_id from labels
    sample_step_ms = int(cfg.sample_interval_sec * 1000)

    # Step 4b: Pre-load Polymarket data (single load for full date range)
    pm_lookup = {}  # hour_start_ms -> (ts_array, mid_array)
    try:
        from marketdata.data.resampled_polymarket import load_resampled_polymarket
        _has_pm = True
    except ImportError:
        _has_pm = False

    if _has_pm:
        print("Pre-loading Polymarket data ...")
        try:
            pm_all = load_resampled_polymarket(
                start_dt=start_dt, end_dt=end_dt,
                interval_ms=1000, asset=cfg.asset,
            )
        except Exception:
            pm_all = pd.DataFrame()

        if not pm_all.empty:
            pm_ts = pm_all["ts_recv"].values
            pm_bid = pm_all["bid"].values
            pm_ask = pm_all["ask"].values
            pm_mid = pm_all["mid"].values

            for _, lbl in labels.iterrows():
                hs = int(lbl["hour_start_ms"])
                he = int(lbl["hour_end_ms"])

                hour_mask = (pm_ts >= hs) & (pm_ts < he)
                if not hour_mask.any():
                    continue

                h_mid = pm_mid[hour_mask]
                h_bid = pm_bid[hour_mask]
                h_ask = pm_ask[hour_mask]

                # Normalize to Up probability (replicates easy_api._normalize_to_up)
                terminal_mid = (h_bid[-1] + h_ask[-1]) / 2
                Y_lbl = int(lbl["Y"])
                token_is_up = (terminal_mid > 0.5) if Y_lbl == 1 else (terminal_mid <= 0.5)

                if token_is_up:
                    pm_lookup[hs] = (pm_ts[hour_mask], h_mid.astype(np.float64))
                else:
                    pm_lookup[hs] = (pm_ts[hour_mask], (1.0 - h_mid).astype(np.float64))

            del pm_all

        print(f"  {len(pm_lookup)} markets pre-loaded")

    rows = []
    pm_hit, pm_miss = 0, 0
    for _, lbl in labels.iterrows():
        hour_start_ms = int(lbl["hour_start_ms"])
        hour_end_ms = int(lbl["hour_end_ms"])
        K = float(lbl["K"])
        S_T = float(lbl["S_T"])
        Y = int(lbl["Y"])

        # Market ID: ASSET_YYYYMMDD_HH
        dt_start = datetime.fromtimestamp(hour_start_ms / 1000, tz=timezone.utc)
        market_id = f"{cfg.asset}_{dt_start.strftime('%Y%m%d_%H')}"
        et_hour = dt_start.astimezone(_ET).hour

        # Find BBO rows within this hour
        mask = (ts_bbo >= hour_start_ms) & (ts_bbo < hour_end_ms)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        # Look up pre-loaded Polymarket data
        pm_ts_arr, pm_mid_arr = None, None
        if hour_start_ms in pm_lookup:
            pm_ts_arr, pm_mid_arr = pm_lookup[hour_start_ms]
            pm_hit += 1
        elif _has_pm:
            pm_miss += 1

        # Subsample at sample_interval_sec spacing
        first_ts = ts_bbo[indices[0]]
        sample_times = np.arange(first_ts, hour_end_ms, sample_step_ms)

        for st in sample_times:
            # Find nearest BBO row at or after sample time
            idx = np.searchsorted(ts_bbo[indices], st)
            if idx >= len(indices):
                continue
            i = indices[idx]

            tau = (hour_end_ms - ts_bbo[i]) / 1000.0
            if tau <= 0:
                continue

            # Nearest-neighbor PM mid price
            pm_val = np.nan
            if pm_ts_arr is not None:
                pm_idx = np.searchsorted(pm_ts_arr, ts_bbo[i])
                pm_idx = np.clip(pm_idx, 0, len(pm_ts_arr) - 1)
                if abs(pm_ts_arr[pm_idx] - ts_bbo[i]) < 5000:  # within 5s
                    pm_val = float(pm_mid_arr[pm_idx])

            rows.append({
                "market_id": market_id,
                "t": int(ts_bbo[i]),
                "S": float(mid_bbo[i]),
                "K": K,
                "tau": tau,
                "y": Y,
                "sigma_tod": float(sigma_tod_all[i]),
                "sigma_rv": float(sigma_rv_all[i]),
                "sigma_rel": float(sigma_rel_all[i]),
                "S_T": S_T,
                "time_since_move": float(time_since_move_all[i]),
                "hour_et": et_hour,
                "pm_mid": pm_val,
            })

    if _has_pm:
        print(f"  Polymarket: {pm_hit} markets matched, {pm_miss} missing")

    dataset = pd.DataFrame(rows)

    # Recompute sigma_tod as integrated remaining seasonal vol (not point estimate).
    # This captures the volatility profile over [t, T] — e.g. pricing in an
    # upcoming NY-open spike even when the current bucket is still quiet.
    if not dataset.empty:
        avg_var = seasonal.integrated_variance_array(
            dataset["t"].values, dataset["tau"].values,
        )
        dataset["sigma_tod"] = np.sqrt(avg_var)
        dataset["sigma_rel"] = (
            dataset["sigma_rv"].values
            / np.maximum(dataset["sigma_tod"].values, 1e-12)
        )


    # Drop rows with NaN or inf in any feature column
    feature_cols = ["S", "K", "S_T", "sigma_tod", "sigma_rv", "sigma_rel"]
    n_before = len(dataset)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    n_dropped = n_before - len(dataset)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} rows with NaN/inf features")
    print(f"  {len(dataset):,} calibration rows from {labels.shape[0]} markets")

    # Save
    dataset_path = output_dir / "calibration_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    print(f"  Saved to {dataset_path}")

    return dataset
