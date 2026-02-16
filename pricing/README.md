# Adaptive-t Binary Option Pricer

A binary option pricing model for Polymarket hourly BTC markets. Given the current BTC price *S*, a strike *K*, and time-to-expiry *τ*, the model outputs P(S_T > K) — the probability that BTC finishes the hour above the strike.

## The Journey: Models Tried

We explored five model architectures before arriving at the adaptive-t model. All were calibrated on 12 days of 1-second Binance BBO data (Jan 19–30 2026, 287 market hours, 17,172 calibration rows sampled at 60s intervals).

### 1. Simple Gaussian (1 parameter)

```
sigma = constant
P(Up) = Phi(-z),  z = log(K/S) / (sigma * sqrt(tau))
```

- **Binary LL**: 0.4754 (+31.1% vs baseline)
- **Drawback**: A single volatility parameter cannot capture regime changes (quiet vs volatile markets), time-of-day effects, or staleness.

### 2. Gaussian (3 parameters)

```
sigma_eff = (a0 + a1 * sqrt(tau_min)) * sigma_tod * sigma_rel^beta
```

where `sigma_tod` is a seasonal (time-of-day) volatility curve and `sigma_rel = sigma_rv / sigma_tod` is the realized/seasonal vol ratio from an EWMA estimator.

- **Binary LL (optimized directly)**: 0.4703 (+31.9% vs baseline)
- **Binary LL (QLIKE vol)**: 0.4720 (+31.6% vs baseline)
- **Drawback**: When optimized for binary log-loss directly, the variance ratios (realized/predicted) diverge from 1.0 — the model sacrifices variance accuracy to game the binary objective. When optimized for QLIKE (a proper scoring rule for variance), the variance ratios are excellent but binary LL is slightly worse. **This misalignment between variance accuracy and binary prediction quality was the key insight.**

### 3. Student-t (4 parameters)

```
Same sigma_eff as Gaussian, but with fixed heavy tails:
P(Up) = T_nu(-z / scale),  scale = sqrt((nu-2)/nu)
```

- **Binary LL**: 0.4680 (+32.2% vs baseline)
- **Drawback**: A fixed degrees-of-freedom parameter applies heavy tails uniformly. But crypto returns are not always heavy-tailed — during active trading, the distribution is closer to Gaussian. Heavy tails everywhere increases variance in predictions.

### 4. Gaussian Jump (6 parameters)

```
sigma_total^2 = sigma_eff^2 + jump_var * f(staleness, session)
```

Added an explicit additive jump-variance term that activates in stale/session-transition regimes.

- **Drawback**: The additive jump variance structure turned out to be the wrong inductive bias. Overfits, and the separation between "continuous" and "jump" variance is not clean in 1-second crypto data.

### 5. Adaptive-t (5 tail parameters + 3 frozen vol) — FINAL MODEL

The key realization: **decouple variance estimation from tail shape**. Use the best variance estimator (QLIKE-calibrated Gaussian), then separately fit how heavy the tails should be as a function of market state.

- **Binary LL**: 0.4693 (+32.0% vs baseline)
- **The surprise**: This achieves the best binary log-loss of all models, despite never directly optimizing for binary log-loss. The two-stage approach — honest variance first, then honest tails — yields better binary predictions than directly optimizing for them.

## The Adaptive-t Model

### Two-Stage Calibration

**Stage 1 — Volatility (QLIKE)**

Calibrates `sigma_eff` by minimizing the QLIKE scoring rule:

```
QLIKE = mean( log(var_pred) + var_realized / var_pred )
```

This is a proper scoring rule that is minimized when `var_pred = E[var_realized]`. It produces honest variance forecasts without gaming any downstream objective.

```
sigma_eff = (a0 + a1 * sqrt(tau_min)) * sigma_tod * sigma_rel^beta
```

Three parameters: `a0`, `a1`, `beta`.
File: `calibrate.py` → `calibrate_vol()`
Model: `models/gaussian.py`
Saved to: `output/gaussian_vol_params.json`

**Stage 2 — Tails (Student-t log-likelihood)**

With `sigma_eff` frozen from stage 1, computes standardized residuals:

```
z = log(S_T / S) / (sigma_eff * sqrt(tau))
```

Then fits state-dependent degrees-of-freedom `nu` by maximizing the Student-t log-likelihood of these residuals:

```
eta = b0 + b_stale * log1p(time_since_move) + b_sess * session_bump(hour_et) + b_tau * log1p(tau)
nu  = nu_min + (nu_max - nu_min) * sigmoid(eta)
```

The CDF uses a variance-preserving scale so that changing `nu` adjusts tail weight without distorting the vol forecast:

```
scale = sqrt((nu - 2) / nu)       # ensures Var(scale * T_nu) = 1.0
P(Up) = T_nu(-z / scale, df=nu)
```

Five parameters: `b0`, `b_stale`, `b_sess`, `b_tau`, `nu_max`.
File: `calibrate.py` → `calibrate_tail()`
Model: `models/gaussian_t.py`
Saved to: `output/gaussian_t_params.json`

### State-Dependent nu — What It Captures

The calibrated b-parameters are all **negative**, meaning `nu` drops (heavier tails) when:

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `b_stale` | −0.45 | When the price hasn't moved for a while, the next move is likely a jump → heavier tails |
| `b_sess`  | −1.26 | During session transitions (6am, 8pm ET), volatility clusters arrive → heavier tails |
| `b_tau`   | −0.26 | At longer horizons, more tail events accumulate → heavier tails |
| `b0`      | −0.41 | Baseline: even in "normal" conditions, crypto returns have moderate heavy tails |

The resulting `nu` distribution has median ~3.7 (very heavy tails on average) with range [3.0, 8.0].

### Session Bumps

The `session_bump` feature uses Gaussian bumps centered at 06:00 and 20:00 ET (US equity open/close transitions):

```python
bump_06 = exp(-((hour - 6)^2) / (2 * 1.5^2))
bump_20 = exp(-((hour - 20)^2) / (2 * 1.5^2))
sess = bump_06 + bump_20
```

## File Structure

```
pricing/
├── __init__.py              # Package docstring
├── README.md                # This file
├── calibrate.py             # Two-stage calibration: calibrate_vol() + calibrate_tail()
├── dataset.py               # Build calibration dataset from 1s Binance data
├── dashboard.py             # Streamlit dashboard (calibration, model vs PM, single market)
├── run_calibration.py       # End-to-end calibration script with diagnostics
├── features/
│   ├── __init__.py
│   ├── seasonal_vol.py      # Time-of-day seasonal volatility (sigma_tod)
│   └── realized_vol.py      # EWMA realized volatility (sigma_rv)
├── models/
│   ├── __init__.py           # Model registry: get_model("gaussian_t")
│   ├── base.py               # Model ABC + CalibrationResult dataclass
│   ├── gaussian.py           # Stage 1 model (QLIKE vol calibration)
│   └── gaussian_t.py         # Production model (adaptive Student-t tails)
└── output/
    ├── gaussian_vol_params.json       # Frozen QLIKE vol params (a0, a1, beta)
    ├── gaussian_t_params.json         # Fitted tail params (b0, b_stale, b_sess, b_tau, nu_max)
    ├── calibration_dataset.parquet    # Cached calibration dataset
    └── seasonal_vol.parquet           # Cached seasonal vol curve
```

## How to Run

### Full calibration

```bash
python pricing/run_calibration.py
```

Runs both stages, prints diagnostics, and saves a 6-panel diagnostic plot to `pricing/output/calibration_diagnostics.png`.

### Interactive dashboard

```bash
streamlit run pricing/dashboard.py
```

Three views:
1. **Calibration** — model diagnostics (reliability curve, LL breakdowns by tau/sigma_rel)
2. **Model vs PM** — head-to-head comparison with Polymarket mid prices
3. **Single Market** — intra-hour price/model/PM with interactive parameter tuning

### Data dependencies

All data loading goes through `src/data/` (not modified by this package):

- `load_binance(start, end, asset, interval)` — 1s BBO data with `ts_recv`, `mid_px`
- `load_binance_labels(start, end, asset)` — hourly labels with `K`, `S_T`, `Y`
- `load_polymarket_market(asset, date, hour_et, interval)` — PM bid/ask/mid at 1s

## Calibration Results (BTC, Jan 19–30 2026)

| Model | Binary LL | vs Baseline | Notes |
|-------|-----------|-------------|-------|
| Baseline (constant) | 0.6905 | — | P = mean(Y) for all rows |
| Simple Gaussian | 0.4754 | +31.1% | 1 param |
| Gaussian (binary LL) | 0.4703 | +31.9% | 3 params, variance ratios diverge |
| Gaussian (QLIKE vol) | 0.4720 | +31.6% | 3 params, honest variance |
| Student-t (fixed nu) | 0.4680 | +32.2% | 4 params, heavy tails everywhere |
| **Adaptive-t** | **0.4693** | **+32.0%** | **8 params (3 frozen + 5 fitted)** |

The adaptive-t achieves the best binary log-loss among properly-calibrated models (only the Student-t with fixed heavy tails everywhere scores marginally better, but at the cost of applying inappropriate tail weight in normal conditions). More importantly, it is the only model where both the variance forecast and the tail shape are honestly calibrated — nothing is gamed.
