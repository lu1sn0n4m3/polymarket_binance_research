#!/usr/bin/env python3
"""
Polymarket Calibration Analysis

Analyzes how well Polymarket's implied probabilities match actual outcomes.
For each market, records the midprice at minutes 5, 15, 25, 35, 45, 55
and compares to the actual resolution.

If PM is well-calibrated, when it says 70% P(Up), ~70% of markets should resolve Up.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.data import load_session

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date range (UTC)
START_DATE = datetime(2026, 1, 18, 13, tzinfo=timezone.utc)
END_DATE = datetime(2026, 1, 23, 19, tzinfo=timezone.utc)

# Minutes to sample
SAMPLE_MINUTES = [5, 15, 25, 35, 45, 55]

# Probability buckets for calibration
PROB_BUCKETS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Asset
ASSET = "BTC"


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print("=" * 70)
    print("Polymarket Calibration Analysis")
    print("=" * 70)
    print()
    print(f"Date range: {START_DATE} to {END_DATE} (UTC)")
    print(f"Asset: {ASSET}")
    print(f"Sample minutes: {SAMPLE_MINUTES}")
    print()
    
    # -------------------------------------------------------------------------
    # 1. Load all sessions in the date range
    # -------------------------------------------------------------------------
    print("Loading sessions...")
    
    sessions_data = []
    current = START_DATE
    
    while current <= END_DATE:
        print(f"  {current.date()}")
        market_date = current.date()
        hour_utc = current.hour
        
        # Convert UTC hour to ET hour (ET = UTC - 5 in winter)
        hour_et = (hour_utc - 5) % 24
        if hour_utc < 5:
            # Previous day in ET
            market_date = (current - timedelta(days=1)).date()
        
        try:
            session = load_session(ASSET, market_date, hour_et=hour_et, lookback_hours=0)
            
            if session.outcome is None:
                print(f"  {current}: No outcome data, skipping")
                current += timedelta(hours=1)
                continue
            
            # Get PM midprice at each sample minute
            pm_bbo = session.polymarket_bbo
            if pm_bbo is None or pm_bbo.empty:
                print(f"  {current}: No PM data, skipping")
                current += timedelta(hours=1)
                continue
            
            # Compute mid price
            pm_bbo = pm_bbo.copy()
            pm_bbo["mid"] = (pm_bbo["bid_px"] + pm_bbo["ask_px"]) / 2
            
            # Normalize to P(Up) using session's token info
            if session.token_is_up is False:
                pm_bbo["mid"] = 1.0 - pm_bbo["mid"]
            
            # Get start time
            start_ms = int(session.utc_start.timestamp() * 1000)
            pm_bbo["elapsed_ms"] = pm_bbo["ts_recv"] - start_ms
            pm_bbo["elapsed_min"] = pm_bbo["elapsed_ms"] / 60000
            
            # Sample at each minute
            outcome_is_up = session.outcome.outcome.lower() == "up"

            # Create unique session identifier
            session_id = f"{market_date}_{hour_et:02d}"

            for minute in SAMPLE_MINUTES:
                # Find nearest quote to exactly this minute (not "last before t+30s")
                pm_bbo["dist_to_target"] = (pm_bbo["elapsed_min"] - minute).abs()
                nearest_idx = pm_bbo["dist_to_target"].idxmin()
                nearest_dist = pm_bbo.loc[nearest_idx, "dist_to_target"]

                # Skip if nearest quote is more than 30 seconds from target
                if nearest_dist > 0.5:
                    continue

                pm_mid = pm_bbo.loc[nearest_idx, "mid"]

                sessions_data.append({
                    "session_id": session_id,
                    "utc_time": current,
                    "market_date": market_date,
                    "hour_et": hour_et,
                    "minute": minute,
                    "pm_mid": pm_mid,
                    "outcome_up": outcome_is_up,
                    "return_pct": session.outcome.return_pct,
                })
            
            print(f"  {current}: {session.outcome.outcome.upper()} ({session.outcome.return_pct:+.4f}%)")
            
        except Exception as e:
            print(f"  {current}: Error - {e}")
        
        current += timedelta(hours=1)
    
    print()
    print(f"Loaded {len(sessions_data)} data points from {len(set(d['session_id'] for d in sessions_data))} sessions")
    print()
    
    if not sessions_data:
        print("No data loaded!")
        return
    
    # -------------------------------------------------------------------------
    # 2. Create DataFrame and bucket analysis
    # -------------------------------------------------------------------------
    df = pd.DataFrame(sessions_data)
    
    # Bucket probabilities
    df["prob_bucket"] = pd.cut(df["pm_mid"], bins=PROB_BUCKETS, labels=False, include_lowest=True)
    df["prob_bucket_label"] = pd.cut(df["pm_mid"], bins=PROB_BUCKETS, include_lowest=True)
    
    # -------------------------------------------------------------------------
    # 3. Calibration analysis by bucket
    # -------------------------------------------------------------------------
    print("Calibration Analysis by Probability Bucket:")
    print("-" * 70)
    
    calibration_data = []
    
    for bucket_idx in range(len(PROB_BUCKETS) - 1):
        bucket_low = PROB_BUCKETS[bucket_idx]
        bucket_high = PROB_BUCKETS[bucket_idx + 1]

        bucket_df = df[df["prob_bucket"] == bucket_idx]
        n = len(bucket_df)

        if n == 0:
            continue

        # Use mean pm_mid as predicted probability (not bucket midpoint)
        predicted_prob = bucket_df["pm_mid"].mean()

        # Actual hit rate
        hits = bucket_df["outcome_up"].sum()
        hit_rate = hits / n

        calibration_data.append({
            "prob_bucket": bucket_idx,
            "bucket_low": bucket_low,
            "bucket_high": bucket_high,
            "predicted_prob": predicted_prob,
            "bucket_label": f"{bucket_low:.0%}-{bucket_high:.0%}",
            "n": n,
            "hits": hits,
            "hit_rate": hit_rate,
            "calibration_error": hit_rate - predicted_prob,
        })

        print(f"  {bucket_low:.0%}-{bucket_high:.0%}: {hit_rate:.1%} actual ({hits}/{n}) "
              f"| Pred: {predicted_prob:.1%} | Error: {(hit_rate - predicted_prob)*100:+.1f}pp")
    
    print()

    calib_df = pd.DataFrame(calibration_data)

    # -------------------------------------------------------------------------
    # 3b. Cluster bootstrap for confidence intervals
    # -------------------------------------------------------------------------
    print("Running cluster bootstrap (resampling sessions)...")
    bootstrap_df = cluster_bootstrap_calibration(df, n_bootstrap=1000, alpha=0.05)

    # Merge bootstrap CIs into calibration data
    calib_df = calib_df.merge(
        bootstrap_df,
        left_on="prob_bucket",
        right_on="bucket_idx",
        how="left",
    )
    calib_df = calib_df.drop(columns=["bucket_idx"], errors="ignore")

    # Print calibration with bootstrap CIs
    print("\nCalibration with Bootstrap CIs:")
    print("-" * 70)
    for _, row in calib_df.iterrows():
        ci_str = f"[{row['hit_rate_ci_low']:.1%}-{row['hit_rate_ci_high']:.1%}]"
        print(f"  {row['bucket_label']}: {row['hit_rate']:.1%} actual "
              f"({row['hits']:.0f}/{row['n']:.0f}) {ci_str}")
    print()

    # Store overall bootstrap CIs
    brier_ci = (bootstrap_df.attrs.get("brier_ci_low"), bootstrap_df.attrs.get("brier_ci_high"))
    mae_ci = (bootstrap_df.attrs.get("mae_ci_low"), bootstrap_df.attrs.get("mae_ci_high"))

    # -------------------------------------------------------------------------
    # 4. Calibration by minute
    # -------------------------------------------------------------------------
    print("Calibration by Sample Minute:")
    print("-" * 70)
    
    minute_data = []
    
    for minute in SAMPLE_MINUTES:
        minute_df = df[df["minute"] == minute]
        n = len(minute_df)
        
        if n == 0:
            continue
        
        # Brier score (lower is better)
        brier = ((minute_df["pm_mid"] - minute_df["outcome_up"].astype(float)) ** 2).mean()
        
        # Mean absolute error
        mae = (minute_df["pm_mid"] - minute_df["outcome_up"].astype(float)).abs().mean()
        
        minute_data.append({
            "minute": minute,
            "n": n,
            "brier_score": brier,
            "mae": mae,
        })
        
        print(f"  Minute {minute:2d}: Brier={brier:.4f}, MAE={mae:.4f} (n={n})")

    print()

    # -------------------------------------------------------------------------
    # 4b. Two-dimensional calibration (minute × probability)
    # -------------------------------------------------------------------------
    print("Computing 2D calibration (time-to-expiry × probability)...")
    print("-" * 70)

    # Define minute bins (0-10, 10-20, ..., 50-60)
    minute_bins = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]

    calib_2d_df = compute_2d_calibration(
        df,
        minute_bins=minute_bins,
        prob_buckets=PROB_BUCKETS,
        n_bootstrap=1000,
        alpha=0.05,
        min_cell_n=3,
    )

    # Print 2D calibration summary
    print("\n2D Calibration Grid (Calib Gap = Hit Rate - Predicted):")
    print("-" * 70)
    for _, row in calib_2d_df.sort_values(["minute_bin", "prob_bucket"]).iterrows():
        ci_str = ""
        if not np.isnan(row.get("calib_gap_ci_low", np.nan)):
            ci_str = f" [{row['calib_gap_ci_low']*100:+.1f}, {row['calib_gap_ci_high']*100:+.1f}]"
        print(f"  {row['minute_bin']}min × {row['bucket_low']:.0%}-{row['bucket_high']:.0%}: "
              f"gap={row['calib_gap']*100:+.1f}pp{ci_str} (n={row['n']})")

    print()

    # -------------------------------------------------------------------------
    # 5. Overall statistics
    # -------------------------------------------------------------------------
    print("Overall Statistics:")
    print("-" * 70)

    overall_brier = ((df["pm_mid"] - df["outcome_up"].astype(float)) ** 2).mean()
    overall_mae = (df["pm_mid"] - df["outcome_up"].astype(float)).abs().mean()

    # Calibration error (average absolute bucket error)
    avg_calib_error = calib_df["calibration_error"].abs().mean()

    print(f"  Total observations: {len(df)}")
    print(f"  Unique sessions: {df['session_id'].nunique()}")
    print(f"  Overall Brier Score: {overall_brier:.4f} [95% CI: {brier_ci[0]:.4f}-{brier_ci[1]:.4f}]")
    print(f"  Overall MAE: {overall_mae:.4f} [95% CI: {mae_ci[0]:.4f}-{mae_ci[1]:.4f}]")
    print(f"  Avg Calibration Error: {avg_calib_error*100:.2f}pp")
    print()
    
    # -------------------------------------------------------------------------
    # 6. Visualization
    # -------------------------------------------------------------------------
    print("Creating visualizations...")

    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)

    # 1D calibration chart
    fig = create_calibration_chart(calib_df, df, minute_data)
    html_path = output_path / f"polymarket_calibration_{ASSET}.html"
    fig.write_html(str(html_path))
    print(f"  Saved 1D calibration to: {html_path}")

    # 2D calibration heatmap
    fig_2d = create_2d_calibration_heatmap(
        calib_2d_df,
        minute_bins=minute_bins,
        prob_buckets=PROB_BUCKETS,
        min_n_threshold=5,
    )
    html_path_2d = output_path / f"polymarket_calibration_2d_{ASSET}.html"
    fig_2d.write_html(str(html_path_2d))
    print(f"  Saved 2D heatmap to: {html_path_2d}")

    # Open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{html_path_2d.absolute()}")
        print("  Opened 2D heatmap in browser")
    except Exception:
        pass

    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)

    return df, calib_df, calib_2d_df


def cluster_bootstrap_calibration(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute calibration metrics with cluster bootstrap over sessions.

    Resamples sessions (not individual observations) to account for
    dependence from multiple time slices per session.

    Returns DataFrame with columns: bucket_idx, hit_rate, hit_rate_ci_low,
    hit_rate_ci_high, brier, brier_ci_low, brier_ci_high, mae, mae_ci_low, mae_ci_high
    """
    rng = np.random.default_rng(rng_seed)
    session_ids = df["session_id"].unique()
    n_sessions = len(session_ids)

    # Storage for bootstrap samples
    bucket_indices = sorted(df["prob_bucket"].dropna().unique().astype(int))
    bootstrap_hit_rates = {b: [] for b in bucket_indices}
    bootstrap_brier = []
    bootstrap_mae = []

    for _ in range(n_bootstrap):
        # Resample sessions with replacement
        sampled_sessions = rng.choice(session_ids, size=n_sessions, replace=True)

        # Build resampled dataframe (sessions can appear multiple times)
        resampled_dfs = []
        for i, sid in enumerate(sampled_sessions):
            session_df = df[df["session_id"] == sid].copy()
            session_df["bootstrap_session_idx"] = i
            resampled_dfs.append(session_df)
        resampled = pd.concat(resampled_dfs, ignore_index=True)

        # Compute Brier and MAE for this bootstrap sample
        outcomes = resampled["outcome_up"].astype(float)
        preds = resampled["pm_mid"]
        bootstrap_brier.append(((preds - outcomes) ** 2).mean())
        bootstrap_mae.append((preds - outcomes).abs().mean())

        # Compute hit rates per bucket
        for bucket_idx in bucket_indices:
            bucket_df = resampled[resampled["prob_bucket"] == bucket_idx]
            if len(bucket_df) > 0:
                bootstrap_hit_rates[bucket_idx].append(
                    bucket_df["outcome_up"].mean()
                )

    # Compute percentile CIs
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    results = []
    for bucket_idx in bucket_indices:
        rates = bootstrap_hit_rates[bucket_idx]
        if len(rates) > 0:
            results.append({
                "bucket_idx": bucket_idx,
                "hit_rate_ci_low": np.percentile(rates, lo_pct),
                "hit_rate_ci_high": np.percentile(rates, hi_pct),
            })

    result_df = pd.DataFrame(results)

    # Add overall metrics
    result_df.attrs["brier_ci_low"] = np.percentile(bootstrap_brier, lo_pct)
    result_df.attrs["brier_ci_high"] = np.percentile(bootstrap_brier, hi_pct)
    result_df.attrs["mae_ci_low"] = np.percentile(bootstrap_mae, lo_pct)
    result_df.attrs["mae_ci_high"] = np.percentile(bootstrap_mae, hi_pct)

    return result_df


def compute_2d_calibration(
    df: pd.DataFrame,
    minute_bins: list[tuple[int, int]],
    prob_buckets: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    min_cell_n: int = 5,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """
    Compute 2D calibration grid (minute_bin x prob_bin) with cluster bootstrap.

    For each cell, computes:
    - mean_pred: mean pm_mid in the cell
    - hit_rate: realized outcome rate
    - calib_gap: hit_rate - mean_pred
    - n: cell count
    - Bootstrap CIs for hit_rate and calib_gap

    Returns DataFrame with one row per cell.
    """
    rng = np.random.default_rng(rng_seed)

    # Create minute bin labels
    df = df.copy()
    minute_bin_labels = []
    for low, high in minute_bins:
        mask = (df["minute"] >= low) & (df["minute"] < high)
        df.loc[mask, "minute_bin"] = f"{low}-{high}"
        minute_bin_labels.append(f"{low}-{high}")

    # Get unique cells
    cells = []
    for mb in minute_bin_labels:
        for pb_idx in range(len(prob_buckets) - 1):
            cells.append((mb, pb_idx))

    # Compute point estimates for each cell
    cell_stats = {}
    for mb, pb_idx in cells:
        cell_df = df[(df["minute_bin"] == mb) & (df["prob_bucket"] == pb_idx)]
        n = len(cell_df)
        if n >= min_cell_n:
            mean_pred = cell_df["pm_mid"].mean()
            hit_rate = cell_df["outcome_up"].mean()
            cell_stats[(mb, pb_idx)] = {
                "minute_bin": mb,
                "prob_bucket": pb_idx,
                "bucket_low": prob_buckets[pb_idx],
                "bucket_high": prob_buckets[pb_idx + 1],
                "n": n,
                "mean_pred": mean_pred,
                "hit_rate": hit_rate,
                "calib_gap": hit_rate - mean_pred,
            }

    # Bootstrap for CIs
    session_ids = df["session_id"].unique()
    n_sessions = len(session_ids)

    bootstrap_hit_rates = {cell: [] for cell in cell_stats.keys()}
    bootstrap_calib_gaps = {cell: [] for cell in cell_stats.keys()}

    for _ in range(n_bootstrap):
        # Resample sessions with replacement
        sampled_sessions = rng.choice(session_ids, size=n_sessions, replace=True)

        # Build resampled dataframe
        resampled_dfs = []
        for i, sid in enumerate(sampled_sessions):
            session_df = df[df["session_id"] == sid].copy()
            session_df["bootstrap_idx"] = i
            resampled_dfs.append(session_df)
        resampled = pd.concat(resampled_dfs, ignore_index=True)

        # Compute stats for each cell
        for cell in cell_stats.keys():
            mb, pb_idx = cell
            cell_df = resampled[
                (resampled["minute_bin"] == mb) & (resampled["prob_bucket"] == pb_idx)
            ]
            if len(cell_df) >= min_cell_n:
                hr = cell_df["outcome_up"].mean()
                mp = cell_df["pm_mid"].mean()
                bootstrap_hit_rates[cell].append(hr)
                bootstrap_calib_gaps[cell].append(hr - mp)

    # Compute percentile CIs
    lo_pct = 100 * alpha / 2
    hi_pct = 100 * (1 - alpha / 2)

    for cell, stats in cell_stats.items():
        hr_samples = bootstrap_hit_rates[cell]
        cg_samples = bootstrap_calib_gaps[cell]
        if len(hr_samples) > 0:
            stats["hit_rate_ci_low"] = np.percentile(hr_samples, lo_pct)
            stats["hit_rate_ci_high"] = np.percentile(hr_samples, hi_pct)
            stats["calib_gap_ci_low"] = np.percentile(cg_samples, lo_pct)
            stats["calib_gap_ci_high"] = np.percentile(cg_samples, hi_pct)
            stats["bootstrap_n"] = len(hr_samples)
        else:
            stats["hit_rate_ci_low"] = np.nan
            stats["hit_rate_ci_high"] = np.nan
            stats["calib_gap_ci_low"] = np.nan
            stats["calib_gap_ci_high"] = np.nan
            stats["bootstrap_n"] = 0

    return pd.DataFrame(list(cell_stats.values()))


def create_2d_calibration_heatmap(
    calib_2d_df: pd.DataFrame,
    minute_bins: list[tuple[int, int]],
    prob_buckets: list[float],
    min_n_threshold: int = 10,
) -> go.Figure:
    """
    Create heatmap visualization for 2D calibration.

    Shows:
    - Left: Calibration gap heatmap (minute x prob)
    - Right: Sample count heatmap
    """
    minute_labels = [f"{low}-{high}" for low, high in minute_bins]
    prob_labels = [f"{prob_buckets[i]:.0%}-{prob_buckets[i+1]:.0%}"
                   for i in range(len(prob_buckets) - 1)]

    # Create matrices for heatmaps
    n_minutes = len(minute_labels)
    n_probs = len(prob_labels)

    gap_matrix = np.full((n_probs, n_minutes), np.nan)
    count_matrix = np.full((n_probs, n_minutes), np.nan)
    annotation_matrix = [['' for _ in range(n_minutes)] for _ in range(n_probs)]

    for _, row in calib_2d_df.iterrows():
        mb_idx = minute_labels.index(row["minute_bin"])
        pb_idx = int(row["prob_bucket"])

        n = row["n"]
        gap = row["calib_gap"] * 100  # Convert to percentage points

        count_matrix[pb_idx, mb_idx] = n

        # Only show gap if sufficient sample size
        if n >= min_n_threshold:
            gap_matrix[pb_idx, mb_idx] = gap
            annotation_matrix[pb_idx][mb_idx] = f"n={n}"
        else:
            annotation_matrix[pb_idx][mb_idx] = f"n={n}"

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "Calibration Gap (Hit Rate − Predicted) [pp]",
            "Sample Count per Cell",
        ],
        horizontal_spacing=0.12,
    )

    # Calibration gap heatmap
    fig.add_trace(
        go.Heatmap(
            z=gap_matrix,
            x=minute_labels,
            y=prob_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-30,
            zmax=30,
            text=annotation_matrix,
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="Minute: %{x}<br>Prob: %{y}<br>Gap: %{z:.1f}pp<extra></extra>",
            colorbar=dict(title="Gap (pp)", x=0.45),
            showscale=True,
        ),
        row=1, col=1,
    )

    # Count heatmap
    fig.add_trace(
        go.Heatmap(
            z=count_matrix,
            x=minute_labels,
            y=prob_labels,
            colorscale="Purples",
            hovertemplate="Minute: %{x}<br>Prob: %{y}<br>Count: %{z:.0f}<extra></extra>",
            colorbar=dict(title="Count", x=1.0),
            showscale=True,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Time into Hour (minutes)", row=1, col=1)
    fig.update_xaxes(title_text="Time into Hour (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="Probability Bucket", row=1, col=1)
    fig.update_yaxes(title_text="Probability Bucket", row=1, col=2)

    fig.update_layout(
        title=dict(
            text="2D Calibration: Time-to-Expiry × Probability",
            font=dict(size=16),
        ),
        height=500,
        width=1000,
    )

    return fig


def create_calibration_chart(calib_df: pd.DataFrame, raw_df: pd.DataFrame, minute_data: list) -> go.Figure:
    """Create calibration visualization."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Calibration Plot (Predicted vs Actual)",
            "Calibration Error by Bucket",
            "Sample Size by Bucket",
            "Brier Score by Minute",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )
    
    # -------------------------------------------------------------------------
    # Plot 1: Calibration curve
    # -------------------------------------------------------------------------
    # Perfect calibration line
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="Perfect Calibration",
            showlegend=True,
        ),
        row=1, col=1,
    )
    
    # Actual calibration with error bars (using bootstrap CIs)
    fig.add_trace(
        go.Scatter(
            x=calib_df["predicted_prob"],
            y=calib_df["hit_rate"],
            mode="markers+lines",
            marker=dict(size=10, color="#2196F3"),
            line=dict(color="#2196F3", width=2),
            error_y=dict(
                type="data",
                symmetric=False,
                array=calib_df["hit_rate_ci_high"] - calib_df["hit_rate"],
                arrayminus=calib_df["hit_rate"] - calib_df["hit_rate_ci_low"],
                color="#2196F3",
                thickness=1.5,
            ),
            name="Polymarket",
            showlegend=True,
        ),
        row=1, col=1,
    )
    
    fig.update_xaxes(title_text="Predicted P(Up)", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="Actual P(Up)", range=[0, 1], row=1, col=1)
    
    # -------------------------------------------------------------------------
    # Plot 2: Calibration error bars
    # -------------------------------------------------------------------------
    colors = ["#4CAF50" if e >= 0 else "#F44336" for e in calib_df["calibration_error"]]
    
    fig.add_trace(
        go.Bar(
            x=calib_df["bucket_label"],
            y=calib_df["calibration_error"] * 100,
            marker_color=colors,
            name="Calibration Error",
            showlegend=False,
        ),
        row=1, col=2,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="Probability Bucket", row=1, col=2)
    fig.update_yaxes(title_text="Error (pp)", row=1, col=2)
    
    # -------------------------------------------------------------------------
    # Plot 3: Sample size histogram
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Bar(
            x=calib_df["bucket_label"],
            y=calib_df["n"],
            marker_color="#9C27B0",
            name="Sample Size",
            showlegend=False,
        ),
        row=2, col=1,
    )
    
    fig.update_xaxes(title_text="Probability Bucket", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    # -------------------------------------------------------------------------
    # Plot 4: Brier score by minute
    # -------------------------------------------------------------------------
    minute_df = pd.DataFrame(minute_data)
    
    fig.add_trace(
        go.Scatter(
            x=minute_df["minute"],
            y=minute_df["brier_score"],
            mode="markers+lines",
            marker=dict(size=10, color="#FF9800"),
            line=dict(color="#FF9800", width=2),
            name="Brier Score",
            showlegend=False,
        ),
        row=2, col=2,
    )
    
    fig.update_xaxes(title_text="Minutes into Hour", row=2, col=2)
    fig.update_yaxes(title_text="Brier Score (lower=better)", row=2, col=2)
    
    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    n_sessions = raw_df["session_id"].nunique()
    overall_brier = ((raw_df["pm_mid"] - raw_df["outcome_up"].astype(float)) ** 2).mean()
    
    fig.update_layout(
        title=dict(
            text=f"Polymarket Calibration Analysis | {n_sessions} sessions | Brier={overall_brier:.4f}",
            font=dict(size=16),
        ),
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    
    return fig


if __name__ == "__main__":
    main()
