# pricing/

Lightweight framework for researching and calibrating binary option pricing models on hourly Polymarket BTC/ETH markets.

## Quick start

```bash
streamlit run pricing/dashboard.py
```

The dashboard has three views: **Calibration** (model diagnostics), **Model vs PM** (head-to-head comparison with Polymarket), and **Single Market** (intra-hour visualization with live parameter tuning).

## How it works

Each hour, Polymarket runs a binary option: "Will BTC close this hour above the opening price?" The opening price K is the Binance mid at the top of the hour, and S_T is the mid at expiry. The outcome is Y=1 if S_T > K, else Y=0.

We model P(Up) = P(S_T > K) using 1-second Binance BBO data and calibrate against realized outcomes via maximum likelihood (minimizing log loss).

## The Gaussian model

The main model (`pricing/models/gaussian.py`) prices the binary option under geometric Brownian motion with zero drift and a time/regime-dependent effective volatility.

### Derivation

Under GBM with drift mu and volatility sigma:

    ln(S_T / S) ~ N((mu - sigma^2/2) * tau, sigma^2 * tau)

For hourly horizons (tau < 3600s), expected drift is negligible relative to volatility, so we set mu = 0:

    P(S_T > K) = P(ln(S_T/S) > ln(K/S))
               = P(Z > (ln(K/S) + 0.5 * sigma^2 * tau) / (sigma * sqrt(tau)))
               = Phi(-z)

where:

    z = (ln(K/S) + 0.5 * sigma_eff^2 * tau) / (sigma_eff * sqrt(tau))

The key modeling choice is how to construct sigma_eff.

### Effective volatility

Rather than using a single global sigma, sigma_eff adapts to both the time-of-day pattern and the current volatility regime:

    sigma_eff = a(tau) * sigma_tod * sigma_rel^beta

where:

- **sigma_tod** — seasonal (time-of-day) volatility. Computed per 5-minute bucket using tick-time realized variance: `sqrt(sum(dx^2) / sum(dt))`, median across days, then circular-smoothed. Units: per-sqrt(sec). This captures the well-known intraday volatility pattern (higher at US open, lower overnight).

- **sigma_rv** — EWMA realized volatility. An exponentially-weighted moving average (half-life = 300s) of the same tick-time estimator, tracking the current vol regime in real time. Units: per-sqrt(sec).

- **sigma_rel = sigma_rv / sigma_tod** — relative volatility. How elevated (or depressed) current realized vol is compared to what's typical for this time of day. sigma_rel > 1 means a volatile regime; < 1 means calm.

- **a(tau) = a0 + a1 * sqrt(tau_min)** — a time-to-expiry adjustment. a0 is the base scale at expiry; a1 * sqrt(tau) lets the model widen or narrow the vol estimate depending on how far out we are.

- **beta** — vol-of-vol exponent. Controls how strongly the model responds to deviations from seasonal vol. beta < 1 dampens extremes; beta > 1 amplifies them.

### Parameters

| Param | Calibrated | Bounds | Role |
|-------|-----------|--------|------|
| a0 | 0.831 | [0.1, 3.0] | Base volatility scale |
| a1 | 0.045 | [-0.5, 0.5] | Tau-dependent vol adjustment |
| beta | 0.739 | [0.0, 2.0] | Relative vol exponent |

Calibrated on BTC Jan 19-30 2026: **LL = 0.451** (+34.7% vs constant-rate baseline 0.691).

### Interpretation of calibrated values

- **a0 = 0.83**: The effective vol is about 83% of the raw seasonal vol at expiry. The raw sigma_tod slightly overstates realized moves for this purpose.
- **a1 = 0.045**: Very mild widening with time — at tau = 60 min, the multiplier is 0.83 + 0.045 * sqrt(60) = 1.18. The model is more uncertain further from expiry.
- **beta = 0.74**: Sub-linear response to vol regimes. When sigma_rel = 2 (vol is 2x seasonal), the model uses 2^0.74 = 1.67x, not the full 2x. This dampens overreaction to vol spikes.

## The simple_gaussian model

A minimal 1-parameter baseline (`pricing/models/simple_gaussian.py`):

    p = Phi(a * ln(S/K) / (sigma_rv * sqrt(tau)))

Uses only sigma_rv (no seasonal vol, no regime adjustment). LL = 0.475 (+31.1% vs baseline). Useful as a sanity check — any new model should beat this.

## Adding a new model

1. Create `pricing/models/your_model.py`, subclass `Model` from `pricing/models/base.py`
2. Implement `predict()`, `param_names()`, `initial_params()`, `param_bounds()`
3. Register in `pricing/models/__init__.py` → `MODEL_REGISTRY`
4. It will appear in the dashboard dropdown automatically

## File structure

```
pricing/
├── __init__.py             # Package docstring
├── dashboard.py            # Streamlit app (3 views)
├── calibrate.py            # MLE calibration (L-BFGS-B) + cluster-robust SE
├── dataset.py              # Builds calibration dataset from 1s Binance + labels
├── models/
│   ├── __init__.py         # MODEL_REGISTRY + get_model()
│   ├── base.py             # Model ABC + CalibrationResult
│   ├── gaussian.py         # 3-param Gaussian model (main)
│   └── simple_gaussian.py  # 1-param baseline
├── features/
│   ├── seasonal_vol.py     # Time-of-day volatility curve
│   └── realized_vol.py     # EWMA realized vol
└── output/                 # Cached dataset + params (git-ignored)
```

## Data dependencies

All data loading goes through `src/data/` (not modified by this package):

- `load_binance(start, end, asset, interval)` — 1s BBO data with `ts_recv`, `mid_px`
- `load_binance_labels(start, end, asset)` — hourly labels with `K`, `S_T`, `Y`
- `load_polymarket_market(asset, date, hour_et, interval)` — PM bid/ask/mid at 1s
