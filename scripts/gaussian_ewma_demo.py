#!/usr/bin/env python3
"""
Gaussian EWMA Pricer Demo

Demonstrates the Gaussian EWMA pricer with:
- Dual EWMA (fast/slow) volatility estimation
- Capped squared returns to prevent jump-induced vol explosion
- Time-of-day prior for early-hour stability
- Comparison of Gaussian vs Student-t distributions

Based on the specification in docs/pricer_guide.md.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data import load_session
from src.pricing import (
    GaussianEWMAPricer,
    GaussianEWMAConfig,
    TODProfile,
    analyze_edge,
)

# =============================================================================
# CONFIGURATION - Adjust these parameters to experiment
# =============================================================================

# Session to analyze
ASSET = "BTC"                           # "BTC" or "ETH"
MARKET_DATE = date(2026, 1, 21)         # Date of the market
HOUR_ET = 8                             # Hour in Eastern Time (0-23)

# EWMA parameters
FAST_HALFLIFE_SEC = 300.0                # Fast EWMA half-life (~1.5 min)
SLOW_HALFLIFE_SEC = 900.0               # Slow EWMA half-life (~15 min)
ALPHA = 0.5                             # Weight on fast vs slow

# Jump capping
CAP_MULTIPLIER = 8.0                    # Cap = c² * v_slow * dt
ENABLE_CAPPING = False                   # Set False to see effect without capping

# TOD prior
TOD_RAMP_SEC = 60.0                     # Ramp to trust live estimate (~1 min)
USE_TOD_PRIOR = True                    # Set False to disable TOD prior
TOD_LOOKBACK_DAYS = 2                   # How many previous days to use for TOD (same hour)
TOD_BLEND_BETA = 0.0                    # Permanent blend: beta*TOD + (1-beta)*EWMA
                                        # 0 = pure EWMA, 1 = pure TOD, 0.3 = 30% TOD anchor

# Volatility multiplier
VOL_MULTIPLIER = 4                   # Scale variance (>1 = more conservative, closer to 50%)

# Student-t degrees of freedom
# scale = sigma * sqrt((nu-2)/nu) to match variance with Gaussian
# Lower nu = fatter tails, but variance-matched → probabilities FURTHER from 50%
STUDENT_T_NU = 8                      # Degrees of freedom (must be > 2)

# Output
OUTPUT_SAMPLE_MS = 1000                 # Grid for output DataFrame (1 second)


# =============================================================================
# MAIN SCRIPT
# =============================================================================

def main():
    print("=" * 70)
    print("Gaussian EWMA Pricer Demo - Gaussian vs Student-t")
    print("=" * 70)
    print()
    
    # -------------------------------------------------------------------------
    # 1. Build TOD Profile from same hour on previous days
    # -------------------------------------------------------------------------
    print(f"Building TOD profile for hour {HOUR_ET} ET from previous {TOD_LOOKBACK_DAYS} days...")
    
    tod_sessions = []
    for days_back in range(1, TOD_LOOKBACK_DAYS + 1):
        past_date = MARKET_DATE - timedelta(days=days_back)
        try:
            sess = load_session(ASSET, past_date, hour_et=HOUR_ET, lookback_hours=0)
            tod_sessions.append(sess)
            print(f"  Loaded {past_date} hour {HOUR_ET} ET")
        except Exception as e:
            print(f"  No data for {past_date} hour {HOUR_ET} ET: {e}")
    
    if not tod_sessions:
        print("  WARNING: No historical data found, using default variance")
        default_var_rate = (0.003 ** 2) / 3600
        tod_profile = TODProfile({HOUR_ET: default_var_rate})
    else:
        tod_profile = TODProfile.from_sessions(tod_sessions, sample_ms=1000)
        v = tod_profile.get(HOUR_ET)
        sigma_hourly = np.sqrt(v * 3600) * 100
        print(f"  TOD variance rate for hour {HOUR_ET} ET: {sigma_hourly:.4f}% hourly vol")
    print()
    
    # -------------------------------------------------------------------------
    # 2. Load the session to analyze
    # -------------------------------------------------------------------------
    print(f"Loading session: {ASSET} {MARKET_DATE} {HOUR_ET}:00 ET")
    
    session = load_session(
        asset=ASSET,
        market_date=MARKET_DATE,
        hour_et=HOUR_ET,
        lookback_hours=3,
    )
    
    print(f"  UTC window: {session.utc_start} to {session.utc_end}")
    
    if session.outcome:
        print(f"  Outcome: {session.outcome.outcome.upper()} ({session.outcome.return_pct:+.4f}%)")
        print(f"  Open: ${session.outcome.open_price:,.2f}")
        print(f"  Close: ${session.outcome.close_price:,.2f}")
    else:
        print("  Outcome: Not available")
    print()
    
    # -------------------------------------------------------------------------
    # 3. Create pricer and price session
    # -------------------------------------------------------------------------
    print("Creating GaussianEWMAPricer...")
    print(f"  Fast half-life: {FAST_HALFLIFE_SEC:.0f}s")
    print(f"  Slow half-life: {SLOW_HALFLIFE_SEC:.0f}s")
    print(f"  Alpha (fast weight): {ALPHA}")
    print(f"  Cap multiplier: {CAP_MULTIPLIER}" + (" (DISABLED)" if not ENABLE_CAPPING else ""))
    print(f"  TOD ramp: {TOD_RAMP_SEC:.0f}s" + (" (DISABLED)" if not USE_TOD_PRIOR else ""))
    print(f"  TOD blend β: {TOD_BLEND_BETA} ({TOD_BLEND_BETA*100:.0f}% TOD anchor)")
    print(f"  Vol multiplier: {VOL_MULTIPLIER}x")
    print(f"  Student-t ν: {STUDENT_T_NU} (scale={np.sqrt((STUDENT_T_NU-2)/STUDENT_T_NU):.3f} for var-match)")
    
    config = GaussianEWMAConfig(
        fast_halflife_sec=FAST_HALFLIFE_SEC,
        slow_halflife_sec=SLOW_HALFLIFE_SEC,
        alpha=ALPHA,
        cap_multiplier=CAP_MULTIPLIER,
        enable_capping=ENABLE_CAPPING,
        tod_ramp_sec=TOD_RAMP_SEC,
        use_tod_prior=USE_TOD_PRIOR,
        tod_blend_beta=TOD_BLEND_BETA,
        vol_multiplier=VOL_MULTIPLIER,
        student_t_nu=STUDENT_T_NU,
    )
    
    pricer = GaussianEWMAPricer(config=config, tod_profile=tod_profile)
    print()
    
    # -------------------------------------------------------------------------
    # 4. Price the session
    # -------------------------------------------------------------------------
    print(f"Pricing session on {OUTPUT_SAMPLE_MS}ms grid...")
    
    df, diagnostics = pricer.price_session(session, output_sample_ms=OUTPUT_SAMPLE_MS)
    
    print(f"  Generated {len(df)} price points")
    print()
    print("  Volatility:")
    print(f"    TOD (from prev days):    {diagnostics.get('sigma_hourly_tod', 0):.4f}% hourly vol")
    print(f"    Lookback (3h pre-mkt):   {diagnostics.get('sigma_hourly_lookback', 0):.4f}% hourly vol")
    print(f"    EWMA init:               {diagnostics.get('sigma_hourly_init', 0):.4f}% hourly vol")
    print()
    print(f"  Capping applied: {diagnostics.get('total_capped', 0)} times ({diagnostics.get('capping_pct', 0):.1f}%)")
    print()
    
    # -------------------------------------------------------------------------
    # 5. Analyze edge for both distributions
    # -------------------------------------------------------------------------
    stats_gaussian = analyze_edge(df, prob_col="prob_gaussian")
    stats_student_t = analyze_edge(df, prob_col="prob_student_t")
    
    print("=" * 70)
    print("COMPARISON: Gaussian vs Student-t(ν={})".format(STUDENT_T_NU))
    print("=" * 70)
    print()
    print(f"{'Distribution':<20} {'Mean Edge':>12} {'|Edge|':>10} {'Std Edge':>10}")
    print("-" * 55)
    print(f"{'Gaussian':<20} {stats_gaussian.get('mean_edge', 0):>+12.4f} {stats_gaussian.get('mean_abs_edge', 0):>10.4f} {stats_gaussian.get('std_edge', 0):>10.4f}")
    print(f"{'Student-t (ν={})'.format(STUDENT_T_NU):<20} {stats_student_t.get('mean_edge', 0):>+12.4f} {stats_student_t.get('mean_abs_edge', 0):>10.4f} {stats_student_t.get('std_edge', 0):>10.4f}")
    print()
    
    # Determine which is better
    if stats_student_t.get('mean_abs_edge', float('inf')) < stats_gaussian.get('mean_abs_edge', float('inf')):
        print(f"✓ Student-t fits better (lower |edge|)")
    else:
        print(f"✓ Gaussian fits better (lower |edge|)")
    print()
    
    # -------------------------------------------------------------------------
    # 6. Create visualization comparing Gaussian vs Student-t
    # -------------------------------------------------------------------------
    print("Creating visualization...")
    
    fig = create_comparison_chart(df, session, stats_gaussian, stats_student_t, STUDENT_T_NU)
    
    # Save to HTML
    output_path = Path(__file__).parent.parent / "outputs"
    output_path.mkdir(exist_ok=True)
    
    html_path = output_path / f"gaussian_ewma_{ASSET}_{MARKET_DATE}_{HOUR_ET}.html"
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
    print("=" * 70)
    print("Done!")
    print("=" * 70)
    
    return df, fig


def create_comparison_chart(df: pd.DataFrame, session, stats_gaussian: dict, stats_student_t: dict, nu: float) -> go.Figure:
    """Create visualization comparing Gaussian vs Student-t distributions."""
    
    x = df["elapsed_sec"] / 60  # Minutes
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.30, 0.25],
        vertical_spacing=0.05,
        subplot_titles=[
            f"Probability: PM bid-ask vs Gaussian & Student-t(ν={nu})",
            "Edge (PM - Model)",
            "Binance Price & Capping Events",
        ],
    )
    
    # -------------------------------------------------------------------------
    # Row 1: Probabilities - PM spread + both distribution lines
    # -------------------------------------------------------------------------
    # PM bid-ask spread (gray background)
    fig.add_trace(
        go.Scatter(
            x=x, y=df["pm_ask"],
            name="PM Ask",
            line=dict(color="#64748b", width=1),
            mode="lines",
            legendgroup="pm",
        ),
        row=1, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=x, y=df["pm_bid"],
            name="PM Bid",
            line=dict(color="#64748b", width=1),
            fill="tonexty",
            fillcolor="rgba(100, 116, 139, 0.2)",
            mode="lines",
            legendgroup="pm",
        ),
        row=1, col=1,
    )
    
    # Gaussian probability
    fig.add_trace(
        go.Scatter(
            x=x, y=df["prob_gaussian"],
            name="Gaussian",
            line=dict(color="#2196F3", width=2),
        ),
        row=1, col=1,
    )
    
    # Student-t probability
    fig.add_trace(
        go.Scatter(
            x=x, y=df["prob_student_t"],
            name=f"Student-t (ν={nu})",
            line=dict(color="#FF9800", width=2),
        ),
        row=1, col=1,
    )
    
    # 50% line
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    
    # -------------------------------------------------------------------------
    # Row 2: Edge for both distributions
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=x, y=df["edge_gaussian"],
            name="Edge (Gaussian)",
            line=dict(color="#2196F3", width=1.5),
            showlegend=False,
        ),
        row=2, col=1,
    )
    
    fig.add_trace(
        go.Scatter(
            x=x, y=df["edge_student_t"],
            name="Edge (Student-t)",
            line=dict(color="#FF9800", width=1.5),
            showlegend=False,
        ),
        row=2, col=1,
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    
    # Stats annotation
    fig.add_annotation(
        x=0.02, y=0.95,
        xref="x2 domain", yref="y2 domain",
        text=f"Mean Edge: Gauss={stats_gaussian.get('mean_edge', 0):+.3f} | t(ν={nu})={stats_student_t.get('mean_edge', 0):+.3f}",
        showarrow=False,
        font=dict(size=10),
        bgcolor="white",
        bordercolor="gray",
    )
    
    # -------------------------------------------------------------------------
    # Row 3: Price and capping
    # -------------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=x, y=df["bnc_mid"],
            name="Binance Mid",
            line=dict(color="#10b981", width=1.5),
        ),
        row=3, col=1,
    )
    
    # Mark capping events
    cap_mask = df["cap_applied"]
    if cap_mask.any():
        fig.add_trace(
            go.Scatter(
                x=x[cap_mask],
                y=df.loc[cap_mask, "bnc_mid"],
                name="Cap Applied",
                mode="markers",
                marker=dict(color="red", size=6, symbol="x"),
            ),
            row=3, col=1,
        )
    
    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    outcome_str = ""
    if session.outcome:
        outcome_str = f" | Outcome: {session.outcome.outcome.upper()} ({session.outcome.return_pct:+.3f}%)"
    
    fig.update_layout(
        title=dict(
            text=f"Gaussian vs Student-t: {session.asset} {session.market_date} {session.hour_et}:00 ET{outcome_str}",
            font=dict(size=16),
        ),
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
    
    # Axis labels
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_yaxes(title_text="Edge", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=3, col=1)
    fig.update_xaxes(title_text="Minutes since open", row=3, col=1)
    
    return fig


if __name__ == "__main__":
    main()
