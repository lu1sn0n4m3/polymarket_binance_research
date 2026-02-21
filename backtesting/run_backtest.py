"""Run a backtest of market-taker strategies against Polymarket.

Usage:
    venv/bin/python3 backtesting/run_backtest.py
"""

import sys
sys.path.insert(0, ".")

from datetime import date

from pricing.pricer import Pricer
from backtesting.simulator import run_backtest
from backtesting.strategies.threshold import EdgeThresholdStrategy
from backtesting.analytics import print_summary, plot_backtest


# =====================================================================
# Configuration
# =====================================================================
ASSET = "BTC"
START_DATE = date(2026, 1, 19)
END_DATE = date(2026, 2, 18)
EWMA_HALF_LIFE = 600.0

# Strategy sweep: vary min_edge to see trade-off
EDGE_THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08]

# =====================================================================
# Run
# =====================================================================
pricer = Pricer.from_calibration("pricing/output")

for min_edge in EDGE_THRESHOLDS:
    strategy = EdgeThresholdStrategy(
        min_edge=min_edge,
        min_tau=60.0,
        max_entries=1,
        sizing="fixed",
        fixed_size=1.0,
    )

    result = run_backtest(
        asset=ASSET,
        start_date=START_DATE,
        end_date=END_DATE,
        strategy=strategy,
        pricer=pricer,
        ewma_half_life=EWMA_HALF_LIFE,
    )

    print_summary(result)

# Detailed plot for the middle threshold
strategy = EdgeThresholdStrategy(min_edge=0.03, min_tau=60.0, sizing="fixed")
result = run_backtest(ASSET, START_DATE, END_DATE, strategy, pricer, EWMA_HALF_LIFE)
plot_backtest(result)
