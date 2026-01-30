# Binary Option Pricer Calibration

Calibrates a model for pricing binary options of the form "Will BTC be above $K at time T?". Estimates P(S_T > K) using volatility-based pricing with a two-stage calibrated parameter set.

## Model

Two-stage pipeline: a volatility model (Stage 1) fed through a probit calibration layer (Stage 2).

### Stage 1: Effective Volatility

$$\sigma_{\text{eff}} = a(\tau) \cdot \sigma_{\text{tod}}^{1-\beta} \cdot \sigma_{\text{rv}}^{\beta}$$

where $a(\tau) = a_0 + a_1 \cdot \sqrt{\tau_{\min}}$ and $\tau_{\min} = \tau / 60$.

Raw z-score from log-normal diffusion with convexity correction:

$$z = \frac{\ln(K/S) + \tfrac{1}{2}\sigma_{\text{eff}}^2 \tau}{\sigma_{\text{eff}} \sqrt{\tau}}$$

**Components:**

- **$\sigma_{\text{tod}}$** (seasonal vol): MAD estimator on tick-time returns, 5-minute buckets (288/day), captures intraday seasonality.
- **$\sigma_{\text{rv}}$** (realized vol): EWMA of sum(dx^2)/sum(dt) with 5-minute half-life, captures current volatility regime.
- **$\beta$** (regime blend): Equal weighting ($\beta \approx 0.50$) of seasonal and realized vol.

### Stage 2: Probit Calibration

$$z' = b_0 + b_1 \cdot \text{score} + b_2 \cdot \ln\tau + b_3 \cdot \ln\sigma_{\text{rel}} + b_4 \cdot (\ln\sigma_{\text{rel}})^2$$

$$p = \Phi(z')$$

where $\text{score} = -z$ and $\sigma_{\text{rel}} = \sigma_{\text{rv}} / \sigma_{\text{tod}}$.

This corrects for residual miscalibration in the raw model across time horizons and volatility regimes.

### Fitted Parameters

8 total parameters (3 + 5), trained on 15,840 samples (264 hourly candles, Jan 19-30 2026).

| Parameter | Value | Role |
|-----------|-------|------|
| $a_0$ | 1.604 | Base vol scale |
| $a_1$ | 0.071 | $\sqrt{\tau}$ dependence |
| $\beta$ | 0.502 | Regime blend (equal weight) |
| $b_0$ | +0.028 | Probit intercept |
| $b_1$ | +1.003 | Score scaling |
| $b_2$ | -0.015 | $\ln\tau$ adjustment |
| $b_3$ | +0.026 | $\ln\sigma_{\text{rel}}$ adjustment |
| $b_4$ | -0.019 | $(\ln\sigma_{\text{rel}})^2$ curvature |

**Performance:** Log loss = 0.4706, +31.9% improvement vs constant rate baseline.

Diagnostics use **cluster-robust standard errors** (blocking by hourly candle) since all 60 samples within each hour share the same outcome. No statistically significant bias remains in any tau-bucket or volatility quartile.

## Directory Structure

```
pricer_calibration/
├── config.yaml            # Pipeline configuration
├── config.py              # Config loader
├── data/
│   ├── ingest.py          # Load Binance BBO data from S3
│   ├── grid.py            # Build 100ms calendar-time grid
│   └── labels.py          # Generate binary labels (did S_T > K?)
├── features/
│   ├── seasonal_vol.py    # Compute sigma_tod (time-of-day vol)
│   └── ewma_shock.py      # EWMA + shock features for grid pipeline
├── model/
│   ├── pricer.py          # Core pricing function
│   └── sensitivities.py   # dp/dS and related derivatives
├── run/
│   ├── build_dataset.py   # Build calibration dataset
│   └── run_calibration.py # One-click calibration script
├── tests/                 # Unit tests
└── output/                # Cached data and results
    ├── params_final.json
    ├── calibration_diagnostics.png
    ├── calibration_conditional.png
    ├── pure_model_diagnostics.png
    └── calibration_dataset.parquet
```

## Quick Start

### 1. Configure

Edit `config.yaml`:

```yaml
start_date: "2026-01-19"
end_date: "2026-01-30"
end_hour: 12
asset: "BTC"
```

### 2. Run Calibration

```bash
# Use existing dataset (fast recalibration)
python -m pricer_calibration.run.run_calibration

# Force rebuild dataset from scratch (after config change or new data)
python -m pricer_calibration.run.run_calibration --rebuild
```

Output:
- `output/params_final.json` -- fitted parameters
- `output/calibration_diagnostics.png` -- main diagnostic plots (2x3)
- `output/calibration_conditional.png` -- conditional reliability (4x4 grid: tau x vol)
- `output/pure_model_diagnostics.png` -- diagnostics without probit layer

### Diagnostic Plots

**Main diagnostics** (2x3):
1. Calibration curve (equal-mass bins)
2. Log loss by time-to-expiry
3. Log loss by |z'| (calibrated score)
4. Log loss by sigma_rel (volatility regime)
5. Prediction distribution by outcome
6. Mean residual vs z'

**Conditional reliability** (4x4):
- Rows: tau buckets [0-5, 5-15, 15-30, 30-60] minutes
- Columns: sigma_rel quartiles [Q1, Q2, Q3, Q4]

## Using the Model

```python
import json
import numpy as np
from scipy.stats import norm

# Load parameters
with open("pricer_calibration/output/params_final.json") as f:
    params = json.load(f)

a0, a1, beta = params["a0"], params["a1"], params["beta"]
probit = params["probit_layer"]

# Given: S (current price), K (strike), tau (seconds to expiry),
#        sigma_tod (seasonal vol), sigma_rv (realized vol)
sigma_rel = sigma_rv / sigma_tod
tau_min = tau / 60.0
a_tau = a0 + a1 * np.sqrt(tau_min)
sigma_eff = a_tau * sigma_tod * sigma_rel**beta

# Stage 1: raw z-score
sqrt_tau = np.sqrt(tau)
z = (np.log(K / S) + 0.5 * sigma_eff**2 * tau) / (sigma_eff * sqrt_tau)

# Stage 2: probit calibration
score = -z
log_tau = np.log(tau)
log_sr = np.log(sigma_rel)
zp = (probit["b0"] + probit["b1"] * score + probit["b2"] * log_tau
      + probit["b3"] * log_sr + probit["b4"] * log_sr**2)
p = norm.cdf(zp)
```

## Tests

```bash
pytest pricer_calibration/tests/
```
