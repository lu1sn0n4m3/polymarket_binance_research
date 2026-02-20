# marketdata

Data infrastructure for the Polymarket-Binance research framework. Handles loading tick-level data from S3, resampling to regular grids, local Parquet caching, time alignment, and feature computation.

## Primary API

All imports work from the top-level package:

```python
from marketdata import load_binance, load_polymarket_market, align_timestamps
```

### `load_binance(start, end, asset, interval)`

Load Binance BBO data for any time range. Data is fetched from S3 on first call, then served from local Parquet cache.

```python
bbo = load_binance(
    start="2026-01-15 09:00:00",   # UTC (str or datetime)
    end="2026-01-20 17:00:00",
    asset="BTC",                    # "BTC" or "ETH"
    interval="1s",                  # "500ms", "1s", or "5s"
    columns=["ts_recv", "mid_px"],  # optional column selection
)
```

Columns: `ts_recv` (epoch ms), `bid_px`, `ask_px`, `bid_sz`, `ask_sz`, `mid_px`, `spread`.

### `load_binance_labels(start, end, asset)`

Load hourly market labels (strike price, terminal price, outcome).

```python
labels = load_binance_labels(start=start_dt, end=end_dt, asset="BTC")
```

Columns: `hour_start_ms`, `hour_end_ms`, `K` (open price), `S_T` (close price), `Y` (1 if up, 0 if down).

### `load_polymarket_market(asset, date, hour_et, interval)`

Load a single Polymarket hourly market. Prices are automatically normalized to P(Up).

```python
pm = load_polymarket_market(asset="BTC", date="2026-01-19", hour_et=9, interval="1s")
```

Columns: `ts_recv` (epoch ms), `bid`, `ask`, `bid_sz`, `ask_sz`, `mid`, `spread`, `microprice`.

### `align_timestamps(left, right, method)`

Join two DataFrames on timestamps.

```python
combined = align_timestamps(left=pm, right=bbo, method="asof_backward")
```

### Session API

Bundle a Polymarket market with its Binance data into a single object:

```python
from marketdata import load_session, HourlyMarketSession

session = load_session(asset="BTC", market_date=date(2026, 1, 19), hour_et=9)
df = session.aligned          # joined DataFrame
outcome = session.outcome     # MarketOutcome with open/close/return/result
```

## Data Flow

```
S3 Bucket (Hetzner)              Local Cache                     Your Code
┌──────────────────┐        ┌─────────────────────┐        ┌──────────────┐
│ Parquet files     │ fetch  │ data/resampled_data/ │  load  │  DataFrame   │
│ partitioned by    │──────> │  ├─ binance/          │──────> │  (pandas)    │
│ venue/stream/     │resample│  │  └─ asset=BTC/     │        └──────────────┘
│ date/hour         │+ cache │  │     └─ interval=1s/│
│                   │        │  │        └─ date=*.parquet
│                   │        │  └─ polymarket/
│                   │        │     └─ ...
└──────────────────┘        └─────────────────────┘
```

### S3 bucket structure

Raw tick data lives in a Hetzner-hosted S3-compatible bucket, partitioned by venue/stream/event/date/hour:

```
s3://marketdata-archive/prod/
  venue=binance/stream_id=BTCUSDT/event_type=bbo/date=2026-01-19/hour=14/data.parquet
  venue=polymarket/stream_id=bitcoin-up-or-down/event_type=bbo/date=2026-01-19/hour=14/data.parquet
```

### Local cache

When data is fetched from S3, it's resampled to a regular grid and saved locally:

```
data/resampled_data/
  binance/asset=BTC/interval=1s/
    date=2026-01-19.parquet
    .metadata.json
  polymarket/asset=BTC/interval=1s/
    date=2026-01-19.parquet
```

Subsequent loads for the same date/interval hit the local cache. Bulk-populate with:

```bash
python scripts/sync_resampled_cache.py --venue binance --asset BTC --interval 1s --start 2026-01-15 --end 2026-02-01
```

## Cache Management

```python
from marketdata.data import get_cache_status, clear_cache

status = get_cache_status("binance", "BTC", "1s")
print(f"Cached: {len(status['dates_cached'])} days, {status['size_mb']:.1f} MB")
```

## Features

Generic market microstructure and volatility features (`marketdata.features`):

| Function | Description |
|---|---|
| `compute_microprice()` | Size-weighted mid price |
| `compute_vwap()` | Volume-weighted average price |
| `compute_spread_bps()` | Spread in basis points |
| `compute_book_imbalance()` | Order book imbalance |
| `compute_trade_imbalance()` | Trade flow imbalance |
| `compute_realized_vol()` | Simple realized volatility |
| `compute_yang_zhang_vol()` | Yang-Zhang volatility estimator |
| `compute_parkinson_vol()` | Parkinson (high-low) volatility |

## Visualization

Interactive Plotly charts (`marketdata.viz`):

- `plot_session(session)` -- dual-axis Polymarket + Binance
- `plot_aligned_prices(df)` -- aligned price series
- `plot_book_snapshot(book)` -- order book depth
- `animate_book(session)` -- animated book over time

## Configuration

S3 and cache settings are in `marketdata/config.py`. Credentials come from environment variables (`S3_ACCESS_KEY`, `S3_SECRET_KEY`) or a `.env` file in the project root.

## File Structure

```
marketdata/
├── __init__.py              # Top-level re-exports (load_binance, etc.)
├── config.py                # S3Config, DataConfig
├── data/
│   ├── __init__.py          # Full API exports (primary + internal)
│   ├── easy_api.py          # Primary user-facing API
│   ├── session.py           # HourlyMarketSession, MarketOutcome
│   ├── resampled_bbo.py     # Binance: S3 fetch + resample + cache
│   ├── resampled_polymarket.py  # Polymarket: S3 fetch + resample + cache
│   ├── resampled_labels.py  # Hourly labels (K, S_T, Y)
│   ├── cache_manager.py     # Parquet cache I/O + metadata
│   ├── alignment.py         # ASOF join, time bucketing, grid resampling
│   ├── loaders.py           # Raw DuckDB S3 queries (internal)
│   └── connection.py        # DuckDB + S3 connection (internal)
├── features/
│   ├── microstructure.py    # microprice, VWAP, spread, imbalance
│   ├── volatility.py        # Realized vol estimators
│   └── historical.py        # Cross-session patterns
└── viz/
    ├── timeseries.py        # Session and price series plots
    └── book.py              # Order book visualization
```

## Notes

- All timestamps are milliseconds since Unix epoch, UTC.
- Polymarket prices are automatically normalized to P(Up). If the underlying token is "Down", prices are flipped.
- Eastern Time: Polymarket markets are hourly in ET. `hour_et=9` = 9:00--10:00 AM Eastern.
