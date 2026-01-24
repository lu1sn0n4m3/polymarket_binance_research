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
            
            for minute in SAMPLE_MINUTES:
                # Find closest observation to this minute
                pm_at_minute = pm_bbo[pm_bbo["elapsed_min"] <= minute + 0.5]
                if pm_at_minute.empty:
                    continue
                
                pm_mid = pm_at_minute.iloc[-1]["mid"]
                
                sessions_data.append({
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
    print(f"Loaded {len(sessions_data)} data points from {len(set(d['utc_time'] for d in sessions_data))} sessions")
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
        bucket_mid = (bucket_low + bucket_high) / 2
        
        bucket_df = df[df["prob_bucket"] == bucket_idx]
        n = len(bucket_df)
        
        if n == 0:
            continue
        
        # Actual hit rate
        hits = bucket_df["outcome_up"].sum()
        hit_rate = hits / n
        
        # Wilson confidence interval (better for proportions)
        ci_low, ci_high = wilson_ci(hits, n, alpha=0.05)
        
        calibration_data.append({
            "bucket_low": bucket_low,
            "bucket_high": bucket_high,
            "bucket_mid": bucket_mid,
            "bucket_label": f"{bucket_low:.0%}-{bucket_high:.0%}",
            "n": n,
            "hits": hits,
            "hit_rate": hit_rate,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "calibration_error": hit_rate - bucket_mid,
        })
        
        print(f"  {bucket_low:.0%}-{bucket_high:.0%}: {hit_rate:.1%} actual ({hits}/{n}) "
              f"[95% CI: {ci_low:.1%}-{ci_high:.1%}] | Error: {(hit_rate - bucket_mid)*100:+.1f}pp")
    
    print()
    
    calib_df = pd.DataFrame(calibration_data)
    
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
    # 5. Overall statistics
    # -------------------------------------------------------------------------
    print("Overall Statistics:")
    print("-" * 70)
    
    overall_brier = ((df["pm_mid"] - df["outcome_up"].astype(float)) ** 2).mean()
    overall_mae = (df["pm_mid"] - df["outcome_up"].astype(float)).abs().mean()
    
    # Calibration error (average absolute bucket error)
    avg_calib_error = calib_df["calibration_error"].abs().mean()
    
    print(f"  Total observations: {len(df)}")
    print(f"  Unique sessions: {df['utc_time'].nunique()}")
    print(f"  Overall Brier Score: {overall_brier:.4f}")
    print(f"  Overall MAE: {overall_mae:.4f}")
    print(f"  Avg Calibration Error: {avg_calib_error*100:.2f}pp")
    print()
    
    # -------------------------------------------------------------------------
    # 6. Visualization
    # -------------------------------------------------------------------------
    print("Creating visualization...")
    
    fig = create_calibration_chart(calib_df, df, minute_data)
    
    # Save
    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)
    
    html_path = output_path / f"polymarket_calibration_{ASSET}.html"
    fig.write_html(str(html_path))
    print(f"  Saved to: {html_path}")
    
    # Open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
        print("  Opened in browser")
    except Exception:
        pass
    
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    
    return df, calib_df


def wilson_ci(hits: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if n == 0:
        return 0.0, 1.0
    
    p = hits / n
    z = stats.norm.ppf(1 - alpha / 2)
    
    denominator = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    
    ci_low = (centre - spread) / denominator
    ci_high = (centre + spread) / denominator
    
    return max(0, ci_low), min(1, ci_high)


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
    
    # Actual calibration with error bars
    fig.add_trace(
        go.Scatter(
            x=calib_df["bucket_mid"],
            y=calib_df["hit_rate"],
            mode="markers+lines",
            marker=dict(size=10, color="#2196F3"),
            line=dict(color="#2196F3", width=2),
            error_y=dict(
                type="data",
                symmetric=False,
                array=calib_df["ci_high"] - calib_df["hit_rate"],
                arrayminus=calib_df["hit_rate"] - calib_df["ci_low"],
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
    n_sessions = raw_df["utc_time"].nunique()
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
