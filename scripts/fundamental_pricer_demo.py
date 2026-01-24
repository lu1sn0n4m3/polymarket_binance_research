#!/usr/bin/env python3
"""
Fundamental Pricer Demo
=======================

This script demonstrates how to use the FundamentalPricer to compute
theoretical probabilities for Polymarket hourly Up/Down markets and
compare them against actual Polymarket prices.

The model is based on the whitepaper:
"A Fundamental Pricing Framework for Hourly Up/Down Prediction Markets"

Key idea:
- At time t, we've observed return r_{0→t} = log(S_t / S_0)
- The remaining return r_{t→T} follows a Student-t distribution
- P(Up) = P(r_{t→T} ≥ -r_{0→t})

Run this script:
    python scripts/fundamental_pricer_demo.py

Or import and use interactively in Jupyter.
"""

import sys
from datetime import date
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import load_session
from src.pricing import (
    FundamentalPricer,
    FundamentalPricerConfig,
    IncrementalPricer,
    LinearBlender,
    analyze_edge,
)


# =============================================================================
# CONFIGURATION - Play with these parameters!
# =============================================================================

# Session to analyze
ASSET = "BTC"                           # "BTC" or "ETH"
MARKET_DATE = date(2026, 1, 20)         # Date of the market
HOUR_ET = 7                            # Hour in Eastern Time (0-23)
LOOKBACK_HOURS = 3                      # Hours of Binance data before market for vol estimation

# Pricer parameters
NU = 5                                # Degrees of freedom for Student-t
                                        #   - Lower ν (3-5): Heavier tails, more conservative
                                        #   - Higher ν (10-30): Closer to Gaussian
                                        #   - ν → ∞: Gaussian

ESTIMATE_NU_FROM_DATA = False            # If True, estimate ν from lookback data (3hr before market)
                                        # If False, use the fixed NU value above

# =============================================================================
# VOLATILITY MODE: Choose how to estimate volatility
# =============================================================================
USE_LOOKBACK_VOL_ONLY = True            # <<< KEY PARAMETER >>>
                                        # True:  Use ONLY pre-market (lookback) volatility
                                        #        - Stable, no in-market vol updates
                                        #        - Vol is "locked in" at market open
                                        #        - Good for avoiding vol spikes during market
                                        # False: Use in-market volatility estimation
                                        #        - More reactive to current conditions
                                        #        - Can be noisy after price jumps

LOOKBACK_SAMPLE_MS = 1000                # Sample rate for lookback vol estimation (ms)
                                        # 100ms captures microstructure well

# In-market vol parameters (only used if USE_LOOKBACK_VOL_ONLY = False)
SAMPLE_MS = 1000                        # Sample interval for in-market RV (ms)

RECENT_WINDOW_MS = 300_000              # Window for recent RV (ms)
                                        #   - 300000 (5 min): More stable
                                        #   - 60000 (60 sec): More reactive but noisy

W_RECENT = 0.3                          # Weight on recent window variance
                                        #   - Higher: More reactive to current regime
                                        #   - Lower: More stable

DRIFT_MU = 0.0                          # Drift term (0 for baseline model)

# Output parameters
OUTPUT_SAMPLE_MS = 1000                 # Grid for output DataFrame (1 second)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 60)
    print("Fundamental Pricer Demo")
    print("=" * 60)
    print()
    
    # -------------------------------------------------------------------------
    # 1. Load the session
    # -------------------------------------------------------------------------
    print(f"Loading session: {ASSET} {MARKET_DATE} {HOUR_ET}:00 ET")
    print(f"  Lookback: {LOOKBACK_HOURS} hours")
    
    session = load_session(
        asset=ASSET,
        market_date=MARKET_DATE,
        hour_et=HOUR_ET,
        lookback_hours=LOOKBACK_HOURS,
    )
    
    print(f"  UTC window: {session.utc_start} to {session.utc_end}")
    
    if session.outcome:
        print(f"  Outcome: {session.outcome.outcome.upper()} ({session.outcome.return_pct:+.4f}%)")
        print(f"  Open: ${session.outcome.open_price:,.2f}")
        print(f"  Close: ${session.outcome.close_price:,.2f}")
        print(f"  Token normalized to Up: {session.token_is_up is False}")
    else:
        print("  Outcome: Not available")
    
    print()
    
    # -------------------------------------------------------------------------
    # 2. Create the pricer
    # -------------------------------------------------------------------------
    print("Creating FundamentalPricer...")
    print(f"  Mode: {'LOOKBACK-ONLY (pre-market vol)' if USE_LOOKBACK_VOL_ONLY else 'IN-MARKET vol estimation'}")
    print(f"  ν (degrees of freedom): {NU}" + (" (will estimate from lookback)" if ESTIMATE_NU_FROM_DATA else ""))
    if USE_LOOKBACK_VOL_ONLY:
        print(f"  Lookback sample rate: {LOOKBACK_SAMPLE_MS} ms")
    else:
        print(f"  In-market sample interval: {SAMPLE_MS} ms")
        print(f"  Recent window: {RECENT_WINDOW_MS / 1000:.0f} seconds")
        print(f"  Recent weight: {W_RECENT}")
    print(f"  Drift (μ): {DRIFT_MU}")
    
    config = FundamentalPricerConfig(
        nu=NU,
        sample_ms=SAMPLE_MS,
        recent_window_ms=RECENT_WINDOW_MS,
        estimate_nu_from_data=ESTIMATE_NU_FROM_DATA,
        mu=DRIFT_MU,
        use_lookback_vol_only=USE_LOOKBACK_VOL_ONLY,
        lookback_sample_ms=LOOKBACK_SAMPLE_MS,
    )
    
    blender = LinearBlender(
        w_recent=W_RECENT,
        use_historical=True,  # Only used if USE_LOOKBACK_VOL_ONLY=False
    )
    
    pricer = FundamentalPricer(
        config=config,
        blender=blender,
    )
    
    # Show estimated parameters from lookback
    if ESTIMATE_NU_FROM_DATA:
        nu_estimated = pricer._get_nu(session)
        print(f"  Estimated ν (MLE from lookback): {nu_estimated:.2f}")
    
    if USE_LOOKBACK_VOL_ONLY:
        sigma_hourly = pricer._get_lookback_hourly_vol(session)
        print(f"  Estimated σ_hourly (from lookback): {sigma_hourly*100:.4f}%")
    
    print()
    
    # -------------------------------------------------------------------------
    # 3. Price the session
    # -------------------------------------------------------------------------
    print(f"Pricing session on {OUTPUT_SAMPLE_MS}ms grid...")
    
    df = pricer.price_session(session, sample_ms=OUTPUT_SAMPLE_MS)
    
    print(f"  Generated {len(df)} price points")
    print()
    
    # -------------------------------------------------------------------------
    # 4. Analyze edge
    # -------------------------------------------------------------------------
    print("Edge Analysis (Polymarket - Fundamental):")
    print("-" * 40)
    
    stats = analyze_edge(df)
    
    print(f"  Mean edge:          {stats['mean_edge']:+.4f}")
    print(f"  Std edge:           {stats['std_edge']:.4f}")
    print(f"  Mean |edge|:        {stats['mean_abs_edge']:.4f}")
    print(f"  Max edge:           {stats['max_edge']:+.4f}")
    print(f"  Min edge:           {stats['min_edge']:+.4f}")
    print(f"  Median edge:        {stats['median_edge']:+.4f}")
    print()
    print(f"  PM overpriced:      {stats['pct_pm_overpriced']:.1%} of time")
    print(f"  PM underpriced:     {stats['pct_pm_underpriced']:.1%} of time")
    print()
    
    # -------------------------------------------------------------------------
    # 5. Show sample data
    # -------------------------------------------------------------------------
    print("Sample data (every 5 minutes):")
    print("-" * 60)
    
    # Sample every 5 minutes (300 rows at 1-second intervals)
    sample_df = df.iloc[::300][["elapsed_sec", "tau_sec", "bnc_mid", "r_0_to_t", "pm_mid", "fundamental_prob", "edge"]].copy()
    sample_df["elapsed_min"] = sample_df["elapsed_sec"] / 60
    sample_df = sample_df[["elapsed_min", "bnc_mid", "r_0_to_t", "pm_mid", "fundamental_prob", "edge"]]
    sample_df.columns = ["min", "BNC_mid", "log_ret", "PM_prob", "Fund_prob", "edge"]
    
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    print(sample_df.to_string(index=False))
    print()
    
    # -------------------------------------------------------------------------
    # 6. Create visualization
    # -------------------------------------------------------------------------
    print("Creating visualization...")
    
    fig = create_comparison_chart(df, session, stats)
    
    # Save to HTML
    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)
    
    html_path = output_path / f"fundamental_pricer_{ASSET}_{MARKET_DATE}_{HOUR_ET}.html"
    fig.write_html(str(html_path))
    print(f"  Saved chart to: {html_path}")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{html_path.absolute()}")
        print("  Opened in browser")
    except Exception:
        pass
    
    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
    
    return df, stats, fig


def create_comparison_chart(df: pd.DataFrame, session, stats: dict) -> go.Figure:
    """Create a comparison chart with prices and edge."""
    
    # Convert elapsed seconds to minutes for x-axis
    x = df["elapsed_sec"] / 60
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.25, 0.30],
        vertical_spacing=0.05,
        subplot_titles=[
            "Probability Comparison",
            "Edge (Polymarket - Fundamental)",
            "Binance Price & Log Return",
        ],
    )
    
    # Row 1: Probability comparison
    # Show Polymarket bid-ask spread as a filled area
    fig.add_trace(
        go.Scatter(
            x=x, y=df["pm_ask"],
            name="PM Ask",
            line=dict(color="#3b82f6", width=1),
            mode="lines",
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=x, y=df["pm_bid"],
            name="PM Bid",
            line=dict(color="#3b82f6", width=1),
            fill="tonexty",  # Fill between bid and ask
            fillcolor="rgba(59, 130, 246, 0.3)",
            mode="lines",
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=x, y=df["fundamental_prob"],
            name="Fundamental",
            line=dict(color="#f59e0b", width=2),
        ),
        row=1, col=1,
    )
    
    # Add 50% line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Row 2: Edge
    fig.add_trace(
        go.Scatter(
            x=x, y=df["edge"],
            name="Edge",
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.3)",
            line=dict(color="#8b5cf6", width=1),
        ),
        row=2, col=1,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Add edge stats annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="x2 domain", yref="y2 domain",
        text=f"μ={stats['mean_edge']:+.3f}  |μ|={stats['mean_abs_edge']:.3f}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        bordercolor="gray",
    )
    
    # Row 3: Price and return
    fig.add_trace(
        go.Scatter(
            x=x, y=df["bnc_mid"],
            name="Binance Mid",
            line=dict(color="#10b981", width=1.5),
        ),
        row=3, col=1,
    )
    
    # Add outcome annotation
    if session.outcome:
        outcome = session.outcome
        outcome_text = f"{outcome.outcome.upper()} ({outcome.return_pct:+.3f}%)"
        
        fig.add_annotation(
            x=60, y=1.02,
            xref="x", yref="paper",
            text=f"Outcome: {outcome_text}",
            showarrow=False,
            font=dict(size=12, color="gray"),
        )
    
    # Update layout
    title = f"Fundamental Pricer Analysis: {session.asset} {session.market_date} {session.hour_et}:00 ET"
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )
    
    # Update axes
    fig.update_xaxes(title_text="Minutes since open", row=3, col=1)
    fig.update_yaxes(title_text="Probability", tickformat=".0%", row=1, col=1)
    fig.update_yaxes(title_text="Edge", tickformat=".2f", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    
    return fig


# =============================================================================
# Parameter Sensitivity Analysis (Optional)
# =============================================================================

def sensitivity_analysis(session, param_name: str, param_values: list):
    """
    Analyze how edge statistics change with different parameter values.
    
    Example:
        sensitivity_analysis(session, "nu", [3, 5, 7, 10, 15])
        sensitivity_analysis(session, "recent_window_ms", [30000, 60000, 120000])
    """
    results = []
    
    for value in param_values:
        # Build config with this parameter value
        config_kwargs = {
            "nu": NU,
            "sample_ms": SAMPLE_MS,
            "recent_window_ms": RECENT_WINDOW_MS,
            "estimate_nu_from_data": False,  # Use fixed value for comparison
            "mu": DRIFT_MU,
        }
        config_kwargs[param_name] = value
        
        config = FundamentalPricerConfig(**config_kwargs)
        pricer = FundamentalPricer(config=config, blender=LinearBlender(w_recent=W_RECENT))
        
        df = pricer.price_session(session, sample_ms=OUTPUT_SAMPLE_MS)
        stats = analyze_edge(df)
        
        results.append({
            param_name: value,
            "mean_edge": stats["mean_edge"],
            "mean_abs_edge": stats["mean_abs_edge"],
            "std_edge": stats["std_edge"],
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Live Trading Example
# =============================================================================

def live_trading_example():
    """
    Demonstrates how to use IncrementalPricer for live trading.
    
    The IncrementalPricer maintains rolling state and can price in ~20μs.
    """
    print("=" * 60)
    print("Live Trading Example (IncrementalPricer)")
    print("=" * 60)
    print()
    
    # In production, you'd get these from your trading system
    session = load_session(ASSET, MARKET_DATE, HOUR_ET, LOOKBACK_HOURS)
    
    # Create incremental pricer
    pricer = IncrementalPricer.from_session(
        session,
        nu=NU,
        sample_ms=SAMPLE_MS,
        recent_window_ms=RECENT_WINDOW_MS,
        w_recent=W_RECENT,
    )
    
    print("IncrementalPricer created.")
    print("  Feed prices with: pricer.update(price, ts_ms)")
    print("  Get probability:  pricer.get_probability()")
    print()
    
    # Simulate feeding prices
    mid_series = session.get_binance_mid_series(sample_ms=1000)
    
    print("Simulating price feed...")
    for i, row in mid_series.iterrows():
        pricer.update(row["mid"], row["ts_ms"])
        
        # Print every 10 minutes
        if i % 600 == 0:
            prob = pricer.get_probability()
            state = pricer.get_state()
            print(f"  t={state['tau_sec']/60:.0f}min remaining, "
                  f"r={state['r_0_to_t']*100:+.3f}%, "
                  f"P(Up)={prob:.3f}")
    
    print()
    print("Final state:")
    state = pricer.get_state()
    for k, v in state.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    df, stats, fig = main()
    
    # Uncomment to run live trading example:
    # live_trading_example()
    
    # Uncomment to run sensitivity analysis:
    # from src.data import load_session
    # session = load_session(ASSET, MARKET_DATE, HOUR_ET, LOOKBACK_HOURS)
    # 
    # print("\nSensitivity to degrees of freedom (ν):")
    # sens_nu = sensitivity_analysis(session, "nu", [3, 4, 5, 7, 10, 15, 20])
    # print(sens_nu.to_string(index=False))
    # 
    # print("\nSensitivity to recent window:")
    # sens_window = sensitivity_analysis(session, "recent_window_ms", [30000, 60000, 90000, 120000])
    # print(sens_window.to_string(index=False))
