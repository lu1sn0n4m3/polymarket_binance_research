# Polymarket-Binance Research Framework

A Python framework for analyzing Polymarket hourly binary options on Bitcoin and Ethereum using synchronized Binance market data.

## Overview

This framework is designed for researching and backtesting trading strategies on Polymarket's hourly "Up or Down" binary option markets. These markets pay $1 if the underlying asset (BTC or ETH) closes the hour higher than it opened, and $0 otherwise.

**Key Features:**
- Load Polymarket and Binance data aligned by receive timestamp (`ts_recv`)
- Automatic normalization to always show "Up" probability (inferred from resolution)
- Lookback window support for volatility estimation
- Interactive Plotly visualizations with dual-axis charts
- Extensible interfaces for volatility estimators and pricing models

## Quick Start

```python
from datetime import date
from src.data import load_session
from src.viz import plot_session

# Load a market session
session = load_session(
    asset="BTC",
    market_date=date(2026, 1, 19),
    hour_et=9,  # 9am Eastern Time
    lookback_hours=3,
)

# Access aligned data
df = session.aligned
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")

# Check outcome
print(f"Outcome: {session.outcome}")

# Visualize
fig = plot_session(session)
fig.show()
```

For a complete walkthrough, see **[`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)**.

---

## Installation

### 1. Create and activate a virtual environment

```bash
cd polymarket_binance_research
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

### 3. Configure S3 credentials

Create a `.env` file in the project root:

```
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
```

The framework uses these defaults (configurable in `src/config.py`):
- **Endpoint:** `nbg1.your-objectstorage.com`
- **Bucket:** `marketdata-archive`
- **Region:** `eu-central`

---

## Project Structure

```
polymarket_binance_research/
├── src/
│   ├── __init__.py
│   ├── config.py              # S3 and data configuration
│   │
│   ├── data/                  # Data loading and alignment
│   │   ├── connection.py      # DuckDB + S3 setup
│   │   ├── loaders.py         # Raw data loaders
│   │   ├── alignment.py       # ASOF join, time bucketing
│   │   └── session.py         # HourlyMarketSession (core abstraction)
│   │
│   ├── features/              # Feature computation
│   │   ├── microstructure.py  # Microprice, VWAP, imbalance
│   │   ├── volatility.py      # Realized volatility estimators
│   │   └── historical.py      # Cross-session features
│   │
│   ├── pricing/               # Pricing models
│   │   └── base.py            # Pricer interface + examples
│   │
│   └── viz/                   # Visualization
│       ├── timeseries.py      # Dual-axis price charts
│       └── book.py            # Order book depth charts
│
├── notebooks/
│   └── 01_data_exploration.ipynb  # Getting started notebook
│
├── tests/                     # Unit tests
├── docs/
│   └── DATA_GUIDE.md          # Data schema documentation
├── pyproject.toml
└── requirements.txt
```

---

## Core Concepts

### The `HourlyMarketSession`

The central abstraction is `HourlyMarketSession` - it represents one hourly Polymarket market (e.g., "BTC Up/Down, Jan 19, 9am-10am ET") with all associated data.

```python
from datetime import date
from src.data import load_session

session = load_session(
    asset="BTC",           # "BTC" or "ETH"
    market_date=date(2026, 1, 19),
    hour_et=9,             # Hour in Eastern Time (0-23)
    lookback_hours=3,      # Hours of Binance data before market for volatility
)
```

**Key Properties:**

| Property | Description |
|----------|-------------|
| `session.aligned` | Main DataFrame with Polymarket + Binance data joined on `ts_recv` |
| `session.outcome` | `MarketOutcome` with open/close prices and result |
| `session.token_is_up` | `True` if loaded token is "Up", `False` if "Down" (inferred) |
| `session.polymarket_bbo` | Raw Polymarket BBO data |
| `session.polymarket_book` | L2 order book snapshots |
| `session.binance_bbo` | Binance BBO for the market hour |
| `session.binance_lookback_trades` | Binance trades including lookback period |

**Time Properties:**

| Property | Description |
|----------|-------------|
| `session.utc_start` | Market start in UTC |
| `session.utc_end` | Market end in UTC |
| `session.lookback_start` | Start of lookback window in UTC |

### The Aligned DataFrame

`session.aligned` is the main data structure you'll work with. It contains:

**Polymarket columns (normalized to "Up" probability):**
- `pm_bid`, `pm_ask` - Best bid/ask prices (0-1 probability)
- `pm_bid_sz`, `pm_ask_sz` - Best bid/ask sizes
- `pm_mid` - Mid price
- `pm_spread` - Bid-ask spread
- `pm_microprice` - Size-weighted mid price

**Binance columns (ASOF-joined):**
- `bnc_bid`, `bnc_ask` - Best bid/ask prices in USD
- `bnc_mid` - Mid price
- `bnc_spread` - Spread

**Important:** Polymarket prices are **automatically normalized** to always represent P(Up). If the loaded token was "Down", prices are flipped using `Up_bid = 1 - Down_ask`.

### Market Outcome

```python
outcome = session.outcome

print(f"Open: ${outcome.open_price:,.2f}")
print(f"Close: ${outcome.close_price:,.2f}")
print(f"Return: {outcome.return_pct:+.3f}%")
print(f"Result: {outcome.outcome}")  # "up", "down", or "flat"
```

The open price is the **first trade** after the hour begins; the close price is the **last trade** in the hour.

---

## Data Loading

### Loading Raw Data

For more control, use the low-level loaders:

```python
from datetime import datetime, timezone
from src.data import (
    load_binance_bbo,
    load_binance_trades,
    load_polymarket_bbo,
    load_polymarket_trades,
    load_polymarket_book,
)

start = datetime(2026, 1, 19, 14, 0, tzinfo=timezone.utc)
end = datetime(2026, 1, 19, 15, 0, tzinfo=timezone.utc)

# Load Binance BBO
bnc_bbo = load_binance_bbo(start, end, asset="BTC")

# Load Polymarket order book
pm_book = load_polymarket_book(start, end, asset="BTC")
```

### Loading Multiple Sessions

```python
from src.data.session import load_sessions_range

sessions = load_sessions_range(
    asset="BTC",
    start_date=date(2026, 1, 16),
    end_date=date(2026, 1, 19),
    hours_et=[9, 10, 11],  # Specific hours, or None for all 24
    preload=False,         # Don't load data until accessed
)

for session in sessions:
    if session.outcome:
        print(f"{session.market_date} {session.hour_et}:00 ET -> {session.outcome.outcome}")
```

---

## Alignment Strategies

The framework provides multiple ways to align Polymarket and Binance data:

### ASOF Join (Default)

For each Polymarket update, get the latest Binance state:

```python
from src.data.alignment import align_asof

aligned = align_asof(
    left=polymarket_df,
    right=binance_df,
    direction="backward",  # Latest Binance at or before Polymarket timestamp
)
```

### Time Bucketing

Aggregate both sources into fixed time buckets:

```python
from src.data.alignment import align_bucketed

aligned = align_bucketed(
    left=polymarket_df,
    right=binance_df,
    bucket_ms=1000,        # 1-second buckets
    agg_method="last",     # Take last value in each bucket
)
```

### Resampling to Grid

Resample to a regular time grid (useful for volatility calculation):

```python
from src.data.alignment import resample_to_grid

resampled = resample_to_grid(
    df=binance_bbo,
    grid_ms=100,           # 100ms grid
    method="ffill",        # Forward-fill missing values
)
```

---

## Feature Computation

### Microstructure Features

```python
from src.features import (
    compute_microprice,
    compute_book_imbalance,
    compute_spread_bps,
    compute_vwap,
)

df = session.aligned

# Microprice (size-weighted mid)
microprice = compute_microprice(df["pm_bid"], df["pm_ask"], df["pm_bid_sz"], df["pm_ask_sz"])

# Book imbalance: (bid_sz - ask_sz) / (bid_sz + ask_sz)
imbalance = compute_book_imbalance(df["pm_bid_sz"], df["pm_ask_sz"])

# Spread in basis points
spread_bps = compute_spread_bps(df["pm_bid"], df["pm_ask"])
```

### Volatility Estimation

```python
from src.features import SimpleRealizedVol, TradeBasedVol

# Simple realized vol from resampled prices
vol_estimator = SimpleRealizedVol(sample_interval_ms=1000)
vol = vol_estimator.compute(session.binance_lookback_trades, price_col="price")

# Rolling volatility
rolling_vol = vol_estimator.compute_rolling(
    session.binance_lookback_trades,
    window_seconds=300,  # 5-minute window
)
```

### Historical Features

```python
from src.features.historical import (
    compute_hourly_returns,
    get_historical_hourly_stats,
    compute_same_hour_features,
)

# Get stats for 9am hour over multiple days
stats = get_historical_hourly_stats(
    asset="BTC",
    hour_et=9,
    start_date=date(2026, 1, 16),
    end_date=date(2026, 1, 19),
)
print(f"Up rate: {stats['up_rate']:.1%}")
print(f"Mean return: {stats['mean_return']:.3f}%")
```

---

## Visualization

### Session Chart (Dual-Axis)

```python
from src.viz import plot_session

fig = plot_session(
    session,
    pm_fields=["pm_bid", "pm_ask", "pm_mid", "pm_microprice"],
    bnc_fields=["bnc_mid"],
    show_outcome=True,
)
fig.show()
```

This creates an interactive Plotly chart with:
- **Left Y-axis:** Polymarket probability (0-100%)
- **Right Y-axis:** Binance price (USD)
- **Vertical line:** Market close with outcome annotation

### Order Book Visualization

```python
from src.viz.book import plot_book_depth, plot_book_depth_over_time

# Single snapshot
book = session.polymarket_book
row = book.iloc[len(book) // 2]

fig = plot_book_depth(
    bid_prices=row["bid_prices"],
    bid_sizes=row["bid_sizes"],
    ask_prices=row["ask_prices"],
    ask_sizes=row["ask_sizes"],
)
fig.show()

# Depth over time
fig = plot_book_depth_over_time(book, depth=5, sample_interval=20)
fig.show()
```

---

## Pricing Models

The framework provides an abstract `Pricer` interface for implementing pricing models:

```python
from src.pricing import Pricer, PricerOutput

class MyPricer(Pricer):
    def price(
        self,
        time_to_expiry_sec: float,
        realized_vol: float,
        current_price: float | None = None,
        strike_price: float | None = None,
        **features,
    ) -> PricerOutput:
        # Your pricing logic here
        up_prob = 0.5  # Placeholder
        
        return PricerOutput.from_up_prob(
            up_prob,
            spread=0.02,  # 2% spread for fair bid/ask
        )
```

**Example with the built-in MoneynessPricer:**

```python
from src.pricing.base import MoneynessPricer

pricer = MoneynessPricer(sensitivity=100)

output = pricer.price(
    time_to_expiry_sec=1800,  # 30 minutes left
    realized_vol=0.5,
    current_price=session.outcome.open_price * 1.001,
    strike_price=session.outcome.open_price,
)

print(f"P(Up): {output.up_prob:.2%}")
print(f"Fair bid: {output.up_fair_bid:.4f}")
print(f"Fair ask: {output.up_fair_ask:.4f}")
```

---

## Configuration

### S3 Configuration

```python
from src.config import set_config, DataConfig, S3Config

set_config(DataConfig(
    s3=S3Config(
        endpoint="your-endpoint.com",
        bucket="your-bucket",
        region="your-region",
        access_key="...",
        secret_key="...",
    )
))
```

### Default Lookback

```python
from src.config import get_config

config = get_config()
config.default_lookback_hours = 2  # Change default from 3 to 2
```

---

## Data Schemas

See **[`docs/DATA_GUIDE.md`](docs/DATA_GUIDE.md)** for detailed documentation on:
- S3 bucket structure and partitioning
- Binance BBO and trade schemas
- Polymarket BBO, trade, and L2 book schemas
- Timestamp handling (`ts_event` vs `ts_recv`)
- DuckDB query examples

---

## Important Notes

### Timestamp Convention

All timestamps are **milliseconds since Unix epoch (UTC)**. Use `ts_recv` for backtesting as it reflects what you would have seen in real-time including network latency.

### Polymarket Price Normalization

The framework **automatically normalizes** Polymarket prices to always represent P(Up):
- Detects whether the loaded token is "Up" or "Down" based on resolution
- If "Down" token was loaded, flips prices: `Up_bid = 1 - Down_ask`

Check which token was loaded:
```python
print(f"Token is Up: {session.token_is_up}")
```

### Eastern Time Handling

Polymarket hourly markets are based on **Eastern Time (ET)**. The framework handles timezone conversion automatically:
- `hour_et=9` means 9:00-10:00 AM Eastern
- Internally converted to UTC for data loading
- During EST: ET = UTC-5; During EDT: ET = UTC-4

---

## Examples

For hands-on examples covering all functionality, see the Jupyter notebook:

**[`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb)**

Topics covered:
1. Loading and inspecting a session
2. Visualizing prices and order book
3. Computing volatility
4. Microstructure features
5. Loading multiple sessions
6. Using the pricer interface

---

## License

MIT
