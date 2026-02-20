# pricing

Gaussian binary option pricer for Polymarket hourly BTC markets. Given current price *S*, strike *K*, and time-to-expiry *tau*, outputs P(S_T > K).

## The Model

Four parameters: `c`, `alpha`, `k0`, `k1`.

### Variance model

```
v = c^2 * sigma_tod^2 * tau^alpha * [m + (1-m) * sigma_rel^2]
m = sigmoid(k0 + k1 * log(tau))
```

- `sigma_tod` -- seasonal (time-of-day) volatility, integrated over [t, T] to capture upcoming regime shifts (e.g. NY open)
- `sigma_rel = sigma_rv / sigma_tod` -- real-time regime level from an EWMA estimator
- `tau` -- time to expiry in seconds
- `m` -- shrinkage: at short horizons (m->0), variance tracks the live regime; at long horizons (m->1), it reverts to seasonal

### Pricing equation

```
p = Phi(-k / sqrt(v))      k = log(K/S)
```

where `Phi` is the standard normal CDF.

## Calibration

Fits `(c, alpha, k0, k1)` by minimizing the QLIKE scoring rule:

```
QLIKE = mean( log(v_hat) + r^2 / v_hat )
```

QLIKE is a proper scoring rule for the conditional mean of squared returns. It produces honest variance forecasts without gaming any downstream objective.

### Design decisions

- **beta=1 fixed**: The exponent on `sigma_rel` was tested as a free parameter and found to be "free" -- fixing beta=1 degrades QLIKE by only +0.003 while actually improving binary LL by 0.4 bps.
- **alpha != 1**: Unlike beta, alpha is doing real work. Fixing alpha=1 causes E[u|tau] to develop a clear slope (0.82 at short horizons, 1.11 at long), confirming horizon-dependent variance scaling.
- **Student-t tail**: A second stage fitting degrees-of-freedom via MLE on z-residuals was tested and removed -- negligible OOS improvement, single symmetric t can't match both center kurtosis and tail pricing.

## Results (BTC, Jan 19 -- Feb 18 2026)

720 markets, 42,934 observations.

### Full-sample

| Model | Binary LL | vs Baseline | Params |
|---|---|---|---|
| Baseline (constant) | 0.6905 | -- | 0 |
| **Gaussian (QLIKE vol)** | **0.4483** | **+35.1%** | **4** |

Parameters: `c=0.685`, `alpha=1.141`, `k0=-2.576`, `k1=0.258`.

### Walk-forward cross-validation (24 folds, 7-day warm-up)

| Model | OOS LL (mean +/- SE) | OOS vs Baseline |
|---|---|---|
| Gaussian | 0.4315 +/- 0.0131 | +37.9% |

Negative overfit gap (-6.8%) -- OOS exceeds in-sample. Parameter stability: `c` CV=20.8%, `alpha` CV=5.5%.

## How to Run

### Full calibration

```bash
python pricing/run_calibration.py
```

Runs QLIKE calibration, prints variance ratio diagnostics, and saves a diagnostic plot to `pricing/output/calibration_diagnostics.png`.

### Cross-validation

```bash
python pricing/cross_validate.py
```

Walk-forward CV with expanding window (7-day minimum training).

### Interactive dashboard

```bash
streamlit run pricing/dashboard.py
```

### Standalone pricer

```python
from pricing.pricer import Pricer

pricer = Pricer.from_calibration("pricing/output")
p = pricer.price(S=104000, K=103500, tau=1800, sigma_rv=3.5e-5, t_ms=1706745600000)
```

No pandas dependency at runtime -- loads frozen parameters from JSON and seasonal vol curves from Parquet.

## Data Dependencies

All data loading goes through `marketdata/` (see [marketdata/README.md](../marketdata/README.md)):

- `load_binance(start, end, asset, interval)` -- 1s BBO with `ts_recv`, `mid_px`
- `load_binance_labels(start, end, asset)` -- hourly labels with `K`, `S_T`, `Y`
- `load_resampled_polymarket(start_dt, end_dt, interval_ms, asset)` -- Polymarket bid/ask/mid

## File Structure

```
pricing/
├── __init__.py              # Package docstring
├── calibrate.py             # QLIKE calibration: calibrate_vol()
├── dataset.py               # Build calibration dataset from 1s Binance data
├── diagnostics.py           # Variance ratio and tail diagnostics
├── pricer.py                # Standalone pricer (loads frozen params from JSON)
├── run_calibration.py       # End-to-end calibration script
├── cross_validate.py        # Walk-forward cross-validation
├── dashboard.py             # Streamlit dashboard
├── analyze_vs_polymarket.py # Head-to-head comparison with Polymarket prices
├── features/
│   ├── seasonal_vol.py      # Time-of-day seasonal volatility (sigma_tod)
│   └── realized_vol.py      # EWMA realized volatility (sigma_rv)
├── models/
│   ├── __init__.py           # Model registry: get_model("gaussian")
│   ├── base.py               # Model ABC + CalibrationResult dataclass
│   └── gaussian.py           # Gaussian model implementation
├── paper/
│   ├── model_spec.tex        # Full mathematical specification (LaTeX)
│   └── generate_figures.py   # Generate paper figures from calibration output
└── output/
    ├── gaussian_vol_params.json       # Calibrated params (c, alpha, k0, k1)
    ├── calibration_dataset.parquet    # Cached calibration dataset
    ├── seasonal_vol_weekday.parquet   # Weekday seasonal vol curve
    └── seasonal_vol_weekend.parquet   # Weekend seasonal vol curve
```
