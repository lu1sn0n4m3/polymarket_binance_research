# Fixed-t Binary Option Pricer

A binary option pricing model for Polymarket hourly BTC markets. Given the current BTC price *S*, a strike *K*, and time-to-expiry *tau*, the model outputs P(S_T > K) -- the probability that BTC finishes the hour above the strike.

## The Model

Five parameters total: four for variance (`c`, `alpha`, `k0`, `k1`) and one for tail shape (`nu`).

### Variance model

```
v = c^2 * sigma_tod^2 * tau^alpha * [m + (1-m) * sigma_rel^2]
m = sigmoid(k0 + k1 * log(tau))
```

where:
- `sigma_tod` is the seasonal (time-of-day) volatility curve (median of 5-min tick-time RV across days)
- `sigma_rel = sigma_rv / sigma_tod` is the real-time regime level from an EWMA estimator
- `tau` is time to expiry in seconds
- `m` controls shrinkage: at short horizons (m->0), variance tracks the live regime; at long horizons (m->1), it reverts to seasonal

### Pricing equation

```
scale = sqrt((nu - 2) / nu)        # variance-preserving scale
p = T_nu(-k / (scale * sqrt(v)))   # k = log(K/S)
```

The variance-preserving scale ensures that changing `nu` adjusts tail shape without distorting the variance forecast. This is the key property that allows two-stage calibration.

## Two-Stage Calibration

**Stage 1 -- Volatility (QLIKE)**

Fits `(c, alpha, k0, k1)` by minimizing the QLIKE scoring rule:

```
QLIKE = mean( log(var_pred) + var_realized / var_pred )
```

QLIKE is a proper scoring rule for the conditional mean of squared returns. It produces honest variance forecasts without gaming any downstream objective.

File: `calibrate.py` -> `calibrate_vol()`
Model: `models/gaussian.py`

**Stage 2 -- Tails (Fixed-nu MLE)**

With variance params frozen from Stage 1, computes standardized residuals `z = r / sqrt(v)` and fits a single scalar `nu` by maximizing the Student-t log-likelihood. Uses Brent's method for the 1D bounded optimization.

File: `calibrate.py` -> `calibrate_tail_fixed()`
Model: `models/fixed_t.py`

Two tail-targeted alternatives are also available for comparison:
- `calibrate_tail_truncated()`: fits nu only on residuals with |z| > z_cutoff (truncated MLE)
- `calibrate_tail_exceedance()`: fits nu to minimize binary log-loss on the indicator 1{|z| > q}

### Why two stages?

The decoupling guarantee: changing `nu` does not affect the variance forecast, and changing `(c, alpha, k0, k1)` does not affect the tail parameter. Each stage targets a different aspect of the distribution with its own proper scoring rule, preventing the adversarial dynamics that arise when optimizing binary log-loss directly over all parameters.

### Design decisions

- **beta=1 fixed**: The exponent on `sigma_rel` was tested as a free parameter and found to be "free" -- fixing beta=1 (i.e., using `sigma_rel^2` instead of `sigma_rel^{2*beta}`) degrades QLIKE by only +0.003 while actually improving binary LL by 0.4 bps.
- **alpha != 1**: Unlike beta, alpha is doing real work. Fixing alpha=1 causes E[u|tau] to develop a clear slope (0.82 at short horizons, 1.11 at long), confirming the horizon-dependent variance scaling.
- **Tail-targeted nu**: Truncated MLE (|z|>z0) and exceedance matching were tested as alternatives to full-sample MLE. Neither salvages Stage 2 — the fundamental issue is that a single symmetric t-distribution cannot simultaneously match center kurtosis and tail pricing. Full MLE (nu~4.7) is the best compromise.

## File Structure

```
pricing/
├── __init__.py              # Package docstring
├── README.md                # This file
├── calibrate.py             # Two-stage calibration: calibrate_vol() + calibrate_tail_fixed()
├── dataset.py               # Build calibration dataset from 1s Binance data
├── diagnostics.py           # Variance ratio and tail diagnostics
├── dashboard.py             # Streamlit dashboard
├── pricer.py                # Standalone pricer (loads params from JSON)
├── run_calibration.py       # End-to-end calibration script with diagnostics
├── cross_validate.py        # Walk-forward cross-validation
├── analyze_vs_polymarket.py # Head-to-head comparison with Polymarket prices
├── features/
│   ├── __init__.py
│   ├── seasonal_vol.py      # Time-of-day seasonal volatility (sigma_tod)
│   └── realized_vol.py      # EWMA realized volatility (sigma_rv)
├── models/
│   ├── __init__.py           # Model registry: get_model("gaussian") / get_model("fixed_t")
│   ├── base.py               # Model ABC + CalibrationResult dataclass
│   ├── gaussian.py           # Stage 1 model (QLIKE vol calibration)
│   └── fixed_t.py            # Stage 2 model (fixed-nu Student-t tails)
├── paper/
│   ├── model_spec.tex        # Full mathematical specification
│   └── generate_figures.py   # Generate paper figures from calibration output
└── output/
    ├── gaussian_vol_params.json       # QLIKE vol params (c, alpha, k0, k1)
    ├── fixed_t_params.json            # Tail param (nu)
    ├── calibration_dataset.parquet    # Cached calibration dataset
    └── seasonal_vol.parquet           # Cached seasonal vol curve
```

## How to Run

### Full calibration

```bash
python pricing/run_calibration.py
```

Runs both stages, prints diagnostics, and saves a 6-panel diagnostic plot to `pricing/output/calibration_diagnostics.png`.

### Cross-validation

```bash
python pricing/cross_validate.py
```

Walk-forward CV with expanding window (7-day minimum training). Confirms no overfitting.

### Interactive dashboard

```bash
streamlit run pricing/dashboard.py
```

### Data dependencies

All data loading goes through `src/data/` (not modified by this package):

- `load_binance(start, end, asset, interval)` -- 1s BBO data with `ts_recv`, `mid_px`
- `load_binance_labels(start, end, asset)` -- hourly labels with `K`, `S_T`, `Y`
- `load_polymarket_market(asset, date, hour_et, interval)` -- PM bid/ask/mid at 1s

## Calibration Results (BTC, Jan 19 -- Feb 18 2026)

720 markets, 42,934 observations.

### Full-sample

| Model | Binary LL | vs Baseline | Params |
|-------|-----------|-------------|--------|
| Baseline (constant) | 0.6905 | -- | 0 |
| Gaussian (QLIKE vol) | 0.4483 | +35.1% | 4 |
| **Fixed-t (nu=4.7)** | **0.4451** | **+35.5%** | **5** |

Calibrated parameters:
- `c = 0.685`, `alpha = 1.141`, `k0 = -2.576`, `k1 = 0.258`
- `nu = 4.69`

### Walk-forward cross-validation (24 folds, 7-day warm-up)

| Model | OOS LL (mean +/- SE) | OOS vs Baseline |
|-------|----------------------|-----------------|
| Gaussian | 0.4315 +/- 0.0131 | +37.9% |
| Fixed-t | 0.4297 +/- 0.0148 | +38.2% |

The negative overfit gap (-6.8%) means OOS performance exceeds in-sample -- the model generalizes well. Parameter stability: `c` CV=20.8%, `alpha` CV=5.5%, `nu` CV=9.6%.
