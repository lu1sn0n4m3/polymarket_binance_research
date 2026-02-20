# CLAUDE.md

## Quick Reference

```bash
# Run calibration (builds dataset + fits model + prints diagnostics)
venv/bin/python3 pricing/run_calibration.py

# Walk-forward cross-validation (24 folds, ~5 min)
venv/bin/python3 pricing/cross_validate.py

# Interactive dashboard
venv/bin/python3 -m streamlit run pricing/dashboard.py

# Run tests
venv/bin/python3 -m pytest tests/ -v

# Generate paper figures
venv/bin/python3 pricing/paper/generate_figures.py

# Sync data cache from S3
venv/bin/python3 scripts/sync_resampled_cache.py --venue binance --asset BTC --interval 1s --start 2026-01-15 --end 2026-02-01
```

## Environment

- **Python**: `venv/bin/python3` (always use this, never system python)
- **matplotlib**: Needs `pip install matplotlib` (not in base deps, used by diagnostics/figures)
- **S3 credentials**: `.env` file in project root with `S3_ACCESS_KEY` and `S3_SECRET_KEY`

## Repository Layout

```
marketdata/          Data infrastructure (S3 loading, caching, alignment, features, viz)
pricing/             Model research (Gaussian pricer, calibration, diagnostics)
scripts/             CLI utilities (cache sync, data checks)
tests/               Unit tests
docs/                USER_GUIDE.md, UPDATING_CACHE.md
data/                Local Parquet cache (git-ignored)
```

## Package: marketdata/

Data infrastructure layer. Do not modify unless explicitly asked.

**Primary API** (re-exported from `marketdata/__init__.py`):
```python
from marketdata import load_binance, load_binance_labels, load_polymarket_market, align_timestamps
```

| Function | File | Purpose |
|---|---|---|
| `load_binance()` | `marketdata/data/easy_api.py` | Load Binance BBO for any time range |
| `load_binance_labels()` | `marketdata/data/easy_api.py` | Hourly labels (K, S_T, Y) |
| `load_polymarket_market()` | `marketdata/data/easy_api.py` | Single PM hourly market |
| `load_resampled_polymarket()` | `marketdata/data/resampled_polymarket.py` | Raw PM data (internal, used by dataset.py) |
| `load_resampled_bbo()` | `marketdata/data/resampled_bbo.py` | Raw Binance data (internal) |

**Data quirks**:
- Timestamps are milliseconds since epoch, UTC
- `load_binance()` returns BBO with `ts_recv`/`mid_px` columns. `mid_px` can have NaN at day boundaries (first row of each day). Always filter NaN before computing log returns
- Polymarket prices are auto-normalized to P(Up)
- Polymarket markets are hourly in ET (Eastern Time), not UTC

## Package: pricing/

Model research framework. This is where most edits happen.

### The Model

```
v = c² · σ_tod² · τ^α · [m + (1-m) · σ_rel²]
m = sigmoid(k₀ + k₁ · ln τ)
p = Φ(-ln(K/S) / √v)
```

4 parameters: `c`, `alpha`, `k0`, `k1`. Calibrated via QLIKE.

### Key Files

| What | File | Key function/class |
|---|---|---|
| **Model definition** | `pricing/models/gaussian.py` | `GaussianModel` (implements `predict`, `predict_variance`, `qlike_gradient`) |
| **Model ABC** | `pricing/models/base.py` | `Model` (abstract base), `CalibrationResult` dataclass |
| **Model registry** | `pricing/models/__init__.py` | `get_model("gaussian")` → `GaussianModel()` |
| **Calibration** | `pricing/calibrate.py` | `calibrate_vol(model, dataset)` → `CalibrationResult` |
| **Dataset builder** | `pricing/dataset.py` | `build_dataset(cfg)` → DataFrame, `DatasetConfig` dataclass |
| **Diagnostics** | `pricing/diagnostics.py` | `variance_ratio_diagnostics()`, `tail_diagnostics()` |
| **Online pricer** | `pricing/pricer.py` | `Pricer.from_calibration("pricing/output")`, `.price(S, K, tau, sigma_rv, t_ms)` |
| **Seasonal vol** | `pricing/features/seasonal_vol.py` | `compute_seasonal_vol_split(bbo)` → `WeekdayWeekendSplit` |
| **Realized vol** | `pricing/features/realized_vol.py` | `compute_rv_ewma(bbo, half_life_sec)` → `(ts, sigma_rv)` |
| **Cross-validation** | `pricing/cross_validate.py` | `run_cv()` — walk-forward, expanding window |
| **End-to-end runner** | `pricing/run_calibration.py` | Script: builds dataset → calibrates → diagnostics |
| **Dashboard** | `pricing/dashboard.py` | Streamlit app |
| **Paper figures** | `pricing/paper/generate_figures.py` | 6 publication figures |
| **LaTeX spec** | `pricing/paper/model_spec.tex` | Full mathematical specification |

### Calibration Output

All in `pricing/output/` (git-ignored):
- `gaussian_vol_params.json` — calibrated params `{c, alpha, k0, k1, qlike, binary_log_loss, ...}`
- `calibration_dataset.parquet` — cached dataset (42,934 rows)
- `seasonal_vol_weekday.parquet` / `seasonal_vol_weekend.parquet` — σ_tod curves

### Current Results (BTC, Jan 19 – Feb 18 2026)

- 720 markets, 42,934 observations
- **LL = 0.4483** (+35.1% vs baseline 0.6905)
- **OOS LL = 0.4315 ± 0.0131** (+37.9%, negative overfit gap)
- Params: c=0.685, α=1.141, k₀=-2.576, k₁=0.258

## How To: Common Tasks

### Add a new model

1. Create `pricing/models/my_model.py`, subclass `Model` from `pricing/models/base.py`
2. Implement: `predict()`, `param_names()`, `initial_params()`, `param_bounds()`
3. Optionally implement: `predict_variance()` (for QLIKE), `qlike_gradient()` (for analytic gradients)
4. Register in `pricing/models/__init__.py` → `MODEL_REGISTRY`
5. Test: `get_model("my_model").predict(params, S, K, tau, features)`

### Change calibration date range

Edit `pricing/run_calibration.py` line 23:
```python
cfg = DatasetConfig(start_date=date(2026, 1, 19), end_date=date(2026, 2, 18))
```

### Change dataset sampling or features

Edit `pricing/dataset.py` → `DatasetConfig` (sample interval, EWMA half-life, etc.) or `build_dataset()` for the row assembly logic.

### Change the variance formula

Edit `pricing/models/gaussian.py` → `predict_variance()` and `qlike_gradient()`. The gradient must match the formula or calibration will be wrong — if unsure, return `None` from `qlike_gradient()` to fall back to numerical gradients.

### Add a new feature to the model

1. Compute it in `pricing/dataset.py` → `build_dataset()` and add to the row dict
2. Add it to `required_features()` in the model class
3. Use it in `predict_variance()` / `predict()`

### Update the data cache

```bash
# Sync specific date range
venv/bin/python3 scripts/sync_resampled_cache.py --venue binance --asset BTC --interval 1s --start 2026-02-01 --end 2026-02-20

# Check cache status
venv/bin/python3 -c "from marketdata import get_cache_status; print(get_cache_status('binance', 'BTC', '1s'))"
```

### Use the pricer in production

```python
from pricing.pricer import Pricer

pricer = Pricer.from_calibration("pricing/output")
p = pricer.price(S=104000, K=103500, tau=1800, sigma_rv=3.5e-5, t_ms=1706745600000)
# p is a numpy array of P(Up) probabilities
```

The `Pricer` class is self-contained — loads params from JSON and seasonal curves from Parquet. No pandas at runtime.

## Conventions

- All scripts use `sys.path.insert(0, ".")` so they can be run from the repo root
- Feature columns in the dataset: `sigma_tod`, `sigma_rv`, `sigma_rel`, `time_since_move`, `hour_et`, `pm_mid`
- `sigma_tod` is in per-√s units (multiply by √τ to get per-horizon vol)
- `tau` is in seconds (not minutes)
- Market ID format: `BTC_YYYYMMDD_HH` (UTC hour)
- The calibration dataset is observation-level (one row per market per sample time), not market-level
