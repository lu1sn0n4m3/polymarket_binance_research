# Binary Option Pricer Calibration

This module calibrates a model for pricing binary options of the form "Will BTC be above $K at time T?". The model estimates the probability P(S_T > K) using volatility-based pricing with calibrated parameters.

## Model

The core pricing formula is:

$$p = 1 - \Phi\left(\frac{\ln(K/S)}{\sigma_{\text{eff}} \cdot \sqrt{\tau}}\right)$$

where $\Phi$ is the standard normal CDF, $S$ is current price, $K$ is strike, and $\tau$ is time to expiry in seconds.

### Effective Volatility

The key innovation is the effective volatility model:

$$\sigma_{\text{eff}} = a(\tau) \cdot \sigma_{\text{tod}} \cdot \left(\frac{\sigma_{\text{rv}}}{\sigma_{\text{tod}}}\right)^\beta$$

This can be rewritten as:

$$\sigma_{\text{eff}} = a(\tau) \cdot \sigma_{\text{tod}}^{1-\beta} \cdot \sigma_{\text{rv}}^\beta$$

**Components:**

1. **τ-dependent scaling**: $a(\tau) = a_0 + a_1 \cdot \sqrt{\tau_{\min}}$
   - Accounts for the fact that short-horizon volatility behaves differently than long-horizon
   - $\tau_{\min} = \tau / 60$ (time to expiry in minutes)

2. **Time-of-day volatility** ($\sigma_{\text{tod}}$):
   - Captures intraday seasonality (e.g., higher vol during US market hours)
   - Computed using MAD estimator on tick-time returns
   - Stored per 5-minute bucket (288 buckets per day)

3. **Realized volatility** ($\sigma_{\text{rv}}$):
   - EWMA of squared tick-time returns with 5-minute half-life
   - Captures current volatility regime (high/low vol periods)
   - More responsive than flat lookback windows

4. **Regime blend** ($\beta$):
   - Controls how much to weight current regime vs seasonal baseline
   - $\beta \approx 0.83$ means ~83% weight on $\sigma_{\text{rv}}$, ~17% on $\sigma_{\text{tod}}$

### Calibrated Parameters

Current fitted values (on ~2,666 samples):

| Parameter | Value | Description |
|-----------|-------|-------------|
| $a_0$ | 0.318 | Base scale |
| $a_1$ | 0.026 | τ-dependence |
| $\beta$ | 0.831 | Regime blend |

**Performance**: Log loss = 0.505 (+27% improvement vs constant rate baseline)

## Directory Structure

```
pricer_calibration/
├── config.yaml            # Pipeline configuration
├── config.py              # Config loader
├── README.md              # This file
├── data/
│   ├── ingest.py          # Load Binance BBO data from S3
│   ├── grid.py            # Build 100ms calendar-time grid
│   └── labels.py          # Generate binary labels (did S_T > K?)
├── features/
│   ├── seasonal_vol.py    # Compute σ_tod (time-of-day vol)
│   └── ewma_shock.py      # EWMA features (legacy)
├── model/
│   ├── pricer.py          # Core pricing function
│   └── sensitivities.py   # dp/dS and related derivatives
├── run/
│   ├── build_dataset.py   # Build calibration dataset (used internally)
│   └── run_calibration.py # ONE-CLICK CALIBRATION SCRIPT
├── tests/                 # Unit tests
└── output/                # Cached data and results
    ├── params_final.json            # Fitted parameters
    ├── calibration_diagnostics.png  # Diagnostic plots
    └── calibration_dataset.parquet  # Calibration data
```

## Quick Start

### 1. Configure

Edit `config.yaml` to set your data range:

```yaml
start_date: "2026-01-18"
end_date: "2026-01-27"
asset: "BTC"
```

### 2. Run Calibration (One Command)

```bash
python -m pricer_calibration.run.run_calibration
```

This single command will:
- Load/build calibration dataset (with S3 caching)
- Compute σ_tod (seasonal) and σ_rv (EWMA) features
- Fit model parameters (a0, a1, β) via maximum likelihood
- Generate diagnostic plots
- Save results to `output/`

### Options

```bash
# Use existing dataset (default - fast recalibration)
python -m pricer_calibration.run.run_calibration

# Force rebuild dataset from scratch
python -m pricer_calibration.run.run_calibration --rebuild

# Use custom config file
python -m pricer_calibration.run.run_calibration --config path/to/config.yaml
```

## Recalibration Workflow

To recalibrate with new data:

1. Update `config.yaml` with new date range
2. Run: `python -m pricer_calibration.run.run_calibration --rebuild`

To recalibrate on existing data (e.g., after code changes):

1. Run: `python -m pricer_calibration.run.run_calibration`

The cache system automatically invalidates when config changes.

## Volatility Estimation Details

### Time-of-Day Volatility (σ_tod)

Computed using **tick-time MAD** (Median Absolute Deviation):

1. Extract consecutive mid-price changes from raw BBO
2. Compute log returns: $r_i = \ln(S_i / S_{i-1})$
3. Normalize to per-√sec: $r_{\text{norm}} = r_i / \sqrt{\Delta t_i}$
4. Bucket by time-of-day (5-min buckets)
5. For each bucket: $\sigma_{\text{tod}} = 1.4826 \times \text{MAD}(r_{\text{norm}})$

The 1.4826 factor makes MAD consistent with Gaussian standard deviation.

### Realized Volatility (σ_rv)

Computed using **EWMA on tick-time returns**:

$$v_k = (1 - \alpha_k) \cdot v_{k-1} + \alpha_k \cdot r_k^2$$

$$\sigma_{\text{rv},k} = \sqrt{v_k}$$

where $\alpha_k = 1 - \exp\left(-\frac{\ln 2 \cdot \Delta t_k}{H}\right)$ and $H = 300$ seconds (5-min half-life).

This gives more weight to recent returns, making $\sigma_{\text{rv}}$ responsive to regime changes.

## Calibration Objective

The model is fit by minimizing **Bernoulli log-loss**:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(p_i) + (1-y_i) \ln(1-p_i) \right]$$

where $y_i \in \{0, 1\}$ is the realized outcome and $p_i$ is the predicted probability.

## Diagnostic Plots

The calibration script generates `calibration_diagnostics.png` with:

1. **Calibration plot**: Predicted vs actual (binned)
2. **Performance by τ**: Log loss by time-to-expiry buckets
3. **Performance by |x|**: Log loss by distance from strike
4. **Performance by σ_rel**: Log loss by volatility regime
5. **Prediction distribution**: Histograms by outcome
6. **Residual analysis**: Mean residual vs x

### The "Dead Zone"

When $|x| = \left|\frac{\ln(K/S)}{\sigma\sqrt{\tau}}\right| < 1$, the outcome is nearly random (close to 50/50). This is the "dead zone" where the model has limited predictive power. Approximately 72% of samples fall in this zone.

The model adds most value when $|x| > 1$ (price is meaningfully far from strike), achieving log loss of ~0.16 in this region.

## Using the Model

```python
from pricer_calibration.model import price_probability

# Load fitted parameters
import json
with open("pricer_calibration/output/params_final.json") as f:
    params = json.load(f)

# Price a binary option
p = price_probability(
    S=100000,           # Current BTC price
    K=101000,           # Strike price
    tau=3600,           # 1 hour to expiry (seconds)
    sigma_base=0.0001,  # Your σ_eff estimate (per √sec)
    z=0,                # Shock statistic (set to 0)
    gamma=0,            # Shock parameter (set to 0)
    dist="normal",      # Use Normal distribution
)
print(f"Probability: {p:.2%}")
```

Note: You'll need to compute `sigma_base` (i.e., $\sigma_{\text{eff}}$) using your own $\sigma_{\text{tod}}$ and $\sigma_{\text{rv}}$ estimates with the fitted $a_0, a_1, \beta$ parameters.

## Tests

Run unit tests:

```bash
pytest pricer_calibration/tests/
```
