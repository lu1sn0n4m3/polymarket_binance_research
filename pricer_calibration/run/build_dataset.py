"""Build calibration dataset: raw BBO → grid → features → sampled rows with labels.

Usage:
    python -m pricer_calibration.run.build_dataset [--config path/to/config.yaml]
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pricer_calibration.config import PipelineConfig, load_config
from pricer_calibration.data.ingest import load_binance_bbo_clean
from pricer_calibration.data.grid import build_grid
from pricer_calibration.data.labels import build_hourly_labels
from pricer_calibration.features.seasonal_vol import compute_seasonal_vol_ticktime
from pricer_calibration.features.ewma_shock import compute_ewma_shock


def build_dataset(cfg: PipelineConfig) -> pd.DataFrame:
    """Run the full pipeline: ingest → grid → features → calibration rows.

    Uses incremental daily caching for BBO data (only fetches missing days).
    """
    from datetime import time as dt_time
    start_dt = datetime.combine(cfg.start_date, dt_time(cfg.start_hour), tzinfo=timezone.utc)
    end_dt = datetime.combine(cfg.end_date, dt_time(cfg.end_hour), tzinfo=timezone.utc)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load BBO with incremental daily caching
    print(f"Loading Binance BBO {cfg.start_date} to {cfg.end_date} ({cfg.asset}) ...")
    bbo = load_binance_bbo_clean(start_dt, end_dt, asset=cfg.asset)
    print(f"  Total: {len(bbo):,} BBO rows")

    if bbo.empty:
        print("No BBO data loaded. Exiting.")
        return pd.DataFrame()

    # Save combined BBO for downstream use (e.g., σ_rv computation)
    bbo_cache = output_dir / "bbo_cache.parquet"
    bbo.to_parquet(bbo_cache, index=False)
    print(f"  Saved combined BBO to {bbo_cache}")

    # Step 2: Build 100ms grid
    grid_cache = output_dir / "grid_cache.parquet"
    print("Building 100ms grid ...")
    grid = build_grid(bbo, delta_ms=cfg.delta_ms)
    grid.to_parquet(grid_cache, index=False)
    print(f"  {len(grid):,} grid rows")

    # Step 3: Seasonal volatility using TICK-TIME method
    print("Computing seasonal volatility (MAD on tick-time returns) ...")
    seasonal = compute_seasonal_vol_ticktime(
        bbo,
        bucket_minutes=cfg.tod_bucket_minutes,
        smoothing_window=cfg.tod_smoothing_window,
        floor=cfg.sigma_tod_floor,
        target_interval_sec=1.0,
    )
    print(f"  {seasonal.n_buckets} TOD buckets, median σ_tod = {np.median(seasonal.sigma_tod):.2e}")

    # Save seasonal vol
    seasonal_df = pd.DataFrame({
        "bucket": np.arange(seasonal.n_buckets),
        "sigma_tod": seasonal.sigma_tod,
    })
    seasonal_df.to_parquet(output_dir / "seasonal_vol.parquet", index=False)

    # Free BBO memory after seasonal vol is computed
    del bbo

    # Step 4: EWMA + shock
    print("Computing EWMA + shock ...")
    features = compute_ewma_shock(
        grid,
        seasonal,
        ewma_half_life_sec=cfg.ewma_half_life_seconds,
        delta_ms=cfg.delta_ms,
        u_sq_cap=cfg.ewma_u_sq_cap,
        shock_M=cfg.shock_M,
    )
    print(f"  Done. σ_rel median = {np.median(features['sigma_rel'].values):.3f}")

    # Step 5: Contract labels
    print("Building hourly labels from trades ...")
    labels = build_hourly_labels(start_dt, end_dt, asset=cfg.asset)
    print(f"  {len(labels)} market hours labeled")

    if labels.empty:
        print("No labels found. Exiting.")
        return pd.DataFrame()

    # Step 6: Sample calibration rows
    print(f"Sampling calibration rows (every {cfg.sample_interval_sec}s) ...")
    sample_step = int(cfg.sample_interval_sec / cfg.delta_sec)

    rows = []
    for _, lbl in labels.iterrows():
        hour_start_ms = lbl["hour_start_ms"]
        hour_end_ms = lbl["hour_end_ms"]
        K = lbl["K"]
        Y = lbl["Y"]
        market_id = lbl["market_id"]

        # Select feature rows within this hour, subsampled
        mask = (features["t"] >= hour_start_ms) & (features["t"] < hour_end_ms)
        hour_features = features[mask]

        if len(hour_features) < sample_step:
            continue

        sampled = hour_features.iloc[::sample_step]

        for _, row in sampled.iterrows():
            tau = (hour_end_ms - row["t"]) / 1000.0  # seconds to expiry
            if tau <= 0:
                continue
            rows.append({
                "market_id": market_id,
                "t": row["t"],
                "S": row["S"],
                "tau": tau,
                "K": K,
                "sigma_base": row["sigma_base"],
                "z": row["z"],
                "y": Y,
            })

    dataset = pd.DataFrame(rows)
    print(f"  {len(dataset):,} calibration rows")

    # Save
    dataset_path = output_dir / "calibration_dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    print(f"  Saved to {dataset_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build calibration dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    build_dataset(cfg)


if __name__ == "__main__":
    main()
