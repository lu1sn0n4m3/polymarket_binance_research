# Polymarket-Binance Research

Research framework for pricing Polymarket hourly binary options on BTC/ETH using tick-level Binance data. The repo has two main packages:

- **`marketdata/`** -- Data infrastructure: loads tick-level Binance and Polymarket data from S3, resamples to regular grids, caches locally as Parquet, and provides aligned DataFrames for research.
- **`pricing/`** -- Model research: a Gaussian binary option pricer calibrated via QLIKE, with walk-forward cross-validation and diagnostic tooling.

## The Problem

Polymarket runs hourly "Up or Down" binary option markets on BTC and ETH. Each market pays $1 if the asset closes the hour above its opening price, $0 otherwise. The goal is to produce accurate real-time probability forecasts P(S_T > K) from the current price, strike, time-to-expiry, and volatility regime.

## The Model

Four-parameter Gaussian diffusion with horizon-dependent shrinkage:

```
v = c^2 * sigma_tod^2 * tau^alpha * [m + (1-m) * sigma_rel^2]
m = sigmoid(k0 + k1 * log(tau))
p = Phi(-log(K/S) / sqrt(v))
```

- `sigma_tod` -- seasonal (time-of-day) volatility, integrated over remaining time to expiry
- `sigma_rel = sigma_rv / sigma_tod` -- real-time regime from EWMA realized vol
- `m` -- shrinkage: at short horizons variance tracks the live regime; at long horizons it reverts to seasonal
- `alpha` -- horizon exponent (alpha != 1 is doing real work)

Calibrated via QLIKE, a proper scoring rule for conditional variance. See [pricing/README.md](pricing/README.md) for full details.

## Results (BTC, Jan 19 -- Feb 18 2026)

720 markets, 42,934 observations.

| | Binary Log-Loss | vs Baseline |
|---|---|---|
| Baseline (constant) | 0.6905 | -- |
| **Gaussian (QLIKE)** | **0.4483** | **+35.1%** |

Walk-forward cross-validation (24 folds, 7-day warm-up): OOS LL = 0.4315 +/- 0.0131 (+37.9%). Negative overfit gap (-6.8%) confirms the model generalizes well.

Calibrated parameters: `c=0.685`, `alpha=1.141`, `k0=-2.576`, `k1=0.258`.

## Quick Start

### Installation

```bash
cd polymarket_binance_research
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Create a `.env` file with S3 credentials:

```
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
```

### Load data

```python
from marketdata import load_binance, load_polymarket_market

# Binance BBO (any time range, auto-cached)
bbo = load_binance(start="2026-01-19 14:00", end="2026-01-19 15:00", asset="BTC", interval="1s")

# Single Polymarket hourly market (normalized to P(Up))
pm = load_polymarket_market(asset="BTC", date="2026-01-19", hour_et=9, interval="1s")
```

See [marketdata/README.md](marketdata/README.md) for the full data API.

### Run calibration

```bash
python pricing/run_calibration.py       # full calibration + diagnostics
python pricing/cross_validate.py        # walk-forward CV
streamlit run pricing/dashboard.py      # interactive dashboard
```

### Use the pricer

```python
from pricing.pricer import Pricer

pricer = Pricer.from_calibration("pricing/output")
p = pricer.price(S=104000, K=103500, tau=1800, sigma_rv=3.5e-5, t_ms=1706745600000)
```

## Project Structure

```
polymarket_binance_research/
├── marketdata/              # Data infrastructure (S3, caching, alignment, features, viz)
│   ├── data/                #   Loading & caching API
│   ├── features/            #   Microstructure, volatility, historical features
│   ├── viz/                 #   Plotly visualizations
│   └── config.py            #   S3 & cache configuration
├── pricing/                 # Model research (Gaussian pricer)
│   ├── models/              #   Model definitions (Gaussian)
│   ├── features/            #   Model-specific features (seasonal vol, realized vol)
│   ├── paper/               #   LaTeX model specification
│   └── output/              #   Calibrated parameters & cached datasets
├── scripts/                 # CLI utilities (cache sync, data checks, visualization)
├── tests/                   # Unit tests
├── docs/                    # USER_GUIDE.md, UPDATING_CACHE.md
└── data/                    # Local Parquet cache (git-ignored)
```

## Further Reading

- [marketdata/README.md](marketdata/README.md) -- Data loading API, caching, S3 architecture
- [pricing/README.md](pricing/README.md) -- Model specification, calibration details, results
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) -- Full API reference with recipes
- [docs/UPDATING_CACHE.md](docs/UPDATING_CACHE.md) -- Cache sync guide

## License

MIT
