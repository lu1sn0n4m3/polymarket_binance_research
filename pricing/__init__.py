"""Lightweight pricing framework for binary option model research.

Workflow:
    1. Define a model (subclass Model in pricing/models/)
    2. Build a calibration dataset from 1s Binance data
    3. Calibrate via MLE (minimize log loss)
    4. Explore via: streamlit run pricing/dashboard.py
"""
