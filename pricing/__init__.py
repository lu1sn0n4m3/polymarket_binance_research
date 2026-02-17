"""Fixed-t binary option pricer for Polymarket hourly BTC markets.

Two-stage calibration:
    1. Volatility:  calibrate_vol() fits variance params via QLIKE
    2. Tails:       calibrate_tail_fixed() fits scalar nu via Student-t LL

Run calibration:    python pricing/run_calibration.py
Explore:            streamlit run pricing/dashboard.py
"""
