"""Adaptive-t binary option pricer for Polymarket hourly BTC markets.

Two-stage calibration:
    1. Volatility:  calibrate_vol() fits sigma_eff via QLIKE
    2. Tails:       calibrate_tail() fits nu(state) via Student-t LL

Run calibration:    python pricing/run_calibration.py
Explore:            streamlit run pricing/dashboard.py
"""
