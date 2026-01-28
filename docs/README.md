# Estimator + Pricer for BTC-Linked Hourly Digitals

This repo implements the approach described in `estimation.tex` and decomposed in `task.md`.

## What it does
- Builds a 100ms mid-price grid from Binance best bid/ask (previous-tick sampling).
- Estimates an intraday seasonal volatility curve σ_tod via robust MAD (MAD × 1.4826).
- Computes live volatility as an EWMA on normalized 100ms returns u_k = r_k / σ_tod.
- Computes a fast shock score z_k as a rolling max of |u_k| over a short window (e.g., 500ms).
- Prices hourly digitals `1{S_T > K}` using a location–scale model with an abstract CDF, specialized to Student‑t.
- Calibrates a small parameter set (typically 1–3 params) by Bernoulli log-likelihood with optional L2 penalty.
- Outputs fair probability p* and sensitivities (dp/dS, dp/dlnS, Δp±(h)).

## Quickstart (suggested)
1. Put raw Binance best bid/ask updates in `data/raw/` (parquet/csv).
2. Configure `config/config.yaml`.
3. Run:
   - `python -m src.run.build_dataset`
   - `python -m src.run.fit_params`
   - `python -m src.run.backtest_pricer`

## Key design choices
- Single grid (100ms) for all features to avoid unit mistakes.
- Seasonal volatility uses full-sample computation (intentional look-ahead) for simplicity in v1.
- Live volatility is estimated relative to time-of-day baseline to make thresholds comparable across sessions.
- Probability is computed as p = P(S_T > K) = 1 - F(x), not F(x), with x = ln(K/S)/(σ_eff√τ).

## Files
- `estimation.tex`: whitepaper.
- `task.md`: engineering plan and acceptance criteria.
