# Polymarket-Binance Research Framework

A Python framework for analyzing Polymarket hourly binary options on Bitcoin and Ethereum using synchronized Binance market data.

## Overview

Polymarket runs hourly "Up or Down" binary option markets: they pay $1 if the underlying asset (BTC or ETH) closes the hour higher than it opened, and $0 otherwise. This framework loads tick-level Binance and Polymarket data from an S3 bucket, resamples it to regular intervals, caches it locally as Parquet files, and provides aligned DataFrames for research and backtesting.

## How Data Flows

```
S3 Bucket (Hetzner)                Local Cache                    Your Code
┌──────────────────┐          ┌─────────────────────┐       ┌──────────────┐
│ Parquet files     │  fetch   │ data/resampled_data/ │ load  │  DataFrame   │
│ partitioned by    │ ──────► │  ├─ binance/          │ ────► │  (pandas)    │
│ venue/stream/     │ resample│  │  └─ asset=BTC/     │       └──────────────┘
│ date/hour         │  + cache│  │     └─ interval=1s/│
│                   │         │  │        └─ date=*.parquet
│                   │         │  └─ polymarket/
│                   │         │     └─ ...
└──────────────────┘          └─────────────────────┘
```

### S3 bucket structure

Raw data lives in a Hetzner-hosted S3-compatible bucket. Each file is a single hour of data:

```
s3://marketdata-archive/prod/
  venue=binance/stream_id=BTCUSDT/event_type=bbo/date=2026-01-19/hour=14/data.parquet
  venue=binance/stream_id=BTCUSDT/event_type=trades/date=2026-01-19/hour=14/data.parquet
  venue=polymarket/stream_id=bitcoin-up-or-down/event_type=bbo/date=2026-01-19/hour=14/data.parquet
  venue=polymarket/stream_id=bitcoin-up-or-down/event_type=trades/...
  venue=polymarket/stream_id=bitcoin-up-or-down/event_type=l2_book/...
```

The framework reads these via DuckDB's `httpfs` extension using S3 credentials from your `.env` file.

### Local cache structure

When data is fetched from S3, it's resampled to a regular time grid (500ms, 1s, or 5s intervals) and saved locally as Parquet:

```
data/resampled_data/
  binance/
    asset=BTC/
      interval=1s/
        date=2026-01-15.parquet
        date=2026-01-16.parquet
        .metadata.json            # tracks cached dates, row counts, file sizes
  polymarket/
    asset=BTC/
      interval=1s/
        date=2026-01-19.parquet
        .metadata.json
```

Subsequent loads for the same date/interval hit the local cache instead of S3. You can bulk-populate the cache using `scripts/sync_resampled_cache.py` (see [docs/UPDATING_CACHE.md](docs/UPDATING_CACHE.md)).

---

## Installation

```bash
cd polymarket_binance_research
python3.12 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Create a `.env` file in the project root with your S3 credentials:

```
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
```

The S3 endpoint, bucket, and region are configured in `marketdata/config.py` (defaults: `nbg1.your-objectstorage.com`, `marketdata-archive`, `eu-central`).

---

## Loading Data

The framework provides two levels of API: a **resampled easy API** for loading continuous time series, and a **session API** for working with individual hourly markets.

### Load Binance data (continuous time series)

Use `load_binance()` to fetch Binance BBO data for any arbitrary time range. The data is automatically fetched from S3 (if not cached), resampled to your chosen interval, and cached locally.

```python
from marketdata import load_binance

bnc = load_binance(
    start="2026-01-15 09:00:00",   # UTC
    end="2026-01-20 17:00:00",
    asset="BTC",                    # "BTC" or "ETH"
    interval="1s",                  # "500ms", "1s", or "5s"
    columns=["ts_recv", "mid_px", "spread"],  # optional column selection
)
```

Returned columns: `ts_recv` (epoch ms), `bid_px`, `ask_px`, `bid_sz`, `ask_sz`, `mid_px`, `spread`.

### Load a Polymarket hourly market

Each Polymarket market is independent and covers a single hour (in Eastern Time). Use `load_polymarket_market()` to load one:

```python
from marketdata import load_polymarket_market

pm = load_polymarket_market(
    asset="BTC",
    date="2026-01-19",
    hour_et=9,          # 9am-10am Eastern Time
    interval="1s",
)
```

Returned columns: `ts_recv` (epoch ms), `bid`, `ask`, `bid_sz`, `ask_sz`, `mid`, `spread`, `microprice`.

All prices are automatically normalized to represent P(Up) — if the underlying token was "Down", prices are flipped (`Up_bid = 1 - Down_ask`).

### Combine Binance + Polymarket

```python
from marketdata import load_binance, load_polymarket_market, align_timestamps

pm = load_polymarket_market("BTC", "2026-01-19", 9, "1s")
bnc = load_binance("2026-01-19 14:00:00", "2026-01-19 15:00:00", "BTC", "1s")

combined = align_timestamps(
    left=pm,
    right=bnc,
    method="asof_backward",  # for each PM update, get latest Binance state
)
```

### Load a full market session (alternative API)

`load_session()` bundles a Polymarket market with its corresponding Binance data (including a lookback window for volatility estimation) and provides an aligned DataFrame:

```python
from datetime import date
from marketdata import load_session

session = load_session(
    asset="BTC",
    market_date=date(2026, 1, 19),
    hour_et=9,
    lookback_hours=3,
)

df = session.aligned          # Polymarket + Binance joined on ts_recv
outcome = session.outcome     # MarketOutcome with open/close/return/result
print(f"Result: {outcome.outcome}, Return: {outcome.return_pct:+.3f}%")
```

The `aligned` DataFrame has columns prefixed `pm_` (Polymarket) and `bnc_` (Binance).

### Load multiple sessions

```python
from marketdata import load_sessions_range

sessions = load_sessions_range(
    asset="BTC",
    start_date=date(2026, 1, 16),
    end_date=date(2026, 1, 19),
    hours_et=[9, 10, 11],  # or None for all 24 hours
    preload=False,
)
```

---

## Cache Management

```python
from marketdata import get_cache_status, clear_cache_easy

# Check what's cached
status = get_cache_status("binance", "BTC", "1s")
print(f"Cached: {len(status['dates_cached'])} days, {status['size_mb']:.1f} MB")

# Bulk sync from S3 (see docs/UPDATING_CACHE.md for full options)
# python scripts/sync_resampled_cache.py --venue binance --asset BTC --interval 1s \
#     --start 2026-01-15 --end 2026-02-01
```

---

## Project Structure

```
polymarket_binance_research/
├── marketdata/                  # Data infrastructure layer
│   ├── config.py                # S3 and data configuration
│   ├── data/
│   │   ├── easy_api.py          # Primary API: load_binance, load_polymarket_market, align_timestamps
│   │   ├── session.py           # HourlyMarketSession abstraction
│   │   ├── resampled_bbo.py     # Binance resampling + cache logic
│   │   ├── resampled_polymarket.py  # Polymarket resampling + cache logic
│   │   ├── cache_manager.py     # Local Parquet cache I/O
│   │   ├── alignment.py         # ASOF join, time bucketing, grid resampling
│   │   ├── loaders.py           # Raw S3 data loading (internal)
│   │   └── connection.py        # DuckDB + S3 setup (internal)
│   ├── features/                # Microstructure, volatility, historical features
│   └── viz/                     # Plotly visualizations (session charts, order books)
├── pricing/                     # Model research framework (Gaussian pricer)
├── scripts/                     # CLI utilities for cache sync, data checks, analysis
├── data/resampled_data/         # Local Parquet cache (git-ignored)
├── docs/
│   ├── USER_GUIDE.md            # Full API reference with recipes
│   └── UPDATING_CACHE.md        # Cache sync guide
└── pyproject.toml
```

---

## Important Notes

- **Timestamps** are milliseconds since Unix epoch, UTC. Use `ts_recv` (receive time) for backtesting.
- **Polymarket prices** are automatically normalized to P(Up). Check `session.token_is_up` to see which token was loaded.
- **Eastern Time**: Polymarket markets are hourly in ET. `hour_et=9` means 9:00–10:00 AM Eastern (UTC-5 in winter, UTC-4 in summer). The framework handles conversion.
- **Column selection**: Pass `columns=["ts_recv", "mid_px"]` to `load_binance()` to reduce memory usage on large queries.

## License

MIT
