# Resampled Market Data - User Guide

Complete guide for loading and analyzing Binance and Polymarket market data.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Common Recipes](#common-recipes)
- [LLM Agent Guide](#llm-agent-guide)
- [Performance Tips](#performance-tips)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Installation

Ensure your cache is populated (see [UPDATING_CACHE.md](UPDATING_CACHE.md)):

```bash
python scripts/sync_resampled_cache.py \
    --venue binance,polymarket \
    --asset BTC \
    --interval 1s \
    --last-days 7 \
    --incremental
```

### Example 1: Load Binance Price History

```python
from marketdata import load_binance

# Load Bitcoin prices for any time range
bnc = load_binance(
    start="2026-01-19 00:00:00",
    end="2026-01-20 00:00:00",
    asset="BTC",
    interval="1s",
)

print(f"Loaded {len(bnc):,} rows")
print(f"Price range: ${bnc['mid_px'].min():.2f} - ${bnc['mid_px'].max():.2f}")
print(f"Columns: {list(bnc.columns)}")
```

**Output:**
```
Loaded 86,400 rows
Price range: $104,523.10 - $105,821.50
Columns: ['ts_recv', 'bid_px', 'ask_px', 'bid_sz', 'ask_sz', 'mid_px', 'spread']
```

### Example 2: Load Polymarket Market

```python
from marketdata import load_polymarket_market
from datetime import date

# Load ONE specific hourly market
pm = load_polymarket_market(
    asset="BTC",
    date=date(2026, 1, 19),
    hour_et=9,  # 9am-10am ET market
    interval="1s",
)

print(f"Market duration: {len(pm)} seconds")
print(f"Opening Up probability: {pm['mid'].iloc[0]:.3f}")
print(f"Closing Up probability: {pm['mid'].iloc[-1]:.3f}")
```

**Output:**
```
Market duration: 3600 seconds
Opening Up probability: 0.520
Closing Up probability: 0.780
```

### Example 3: Combine Binance + Polymarket

```python
from marketdata import load_binance, load_polymarket_market, align_timestamps
from datetime import date

# Load Polymarket market
pm = load_polymarket_market("BTC", date(2026, 1, 19), 9, "1s")

# Load Binance data (wider range for context)
bnc = load_binance(
    start="2026-01-19 13:00:00",  # 9am ET = 2pm UTC
    end="2026-01-19 15:00:00",
    asset="BTC",
    interval="1s",
    columns=["ts_recv", "mid_px", "spread"],  # Column selection for efficiency
)

# Align: For each PM update, get latest Binance state
combined = align_timestamps(
    left=pm,
    right=bnc,
    method="asof_backward",
    left_suffix="_pm",
    right_suffix="_bnc",
)

print(f"Combined: {len(combined)} rows")
print(combined[["ts_recv", "mid_pm", "mid_px_bnc"]].head())
```

---

## Core Concepts

### 1. Binance vs Polymarket - Different Use Cases

**Binance:**
- Continuous spot price data for BTC/ETH
- Load arbitrary time ranges (seconds, days, weeks)
- Use for: price history, volatility analysis, reference prices

**Polymarket:**
- Hourly binary options markets ("Will BTC close higher?")
- Load ONE market at a time (markets are independent)
- Use for: probability forecasts, market sentiment, strategy testing

**Key Insight:** Binance and Polymarket serve different purposes. Always load them separately, then combine with utilities.

### 2. Resampling - Why 1s Instead of High-Frequency?

Raw data from S3 is high-frequency (tick-by-tick):
- Binance: ~10-100 updates per second
- Polymarket: ~1-10 updates per second

**Resampled data** is:
- Aligned to fixed intervals (500ms, 1s, 5s)
- Forward-filled (last quote persists until next update)
- Cached locally for fast access (10-50x faster than S3)

**When to use each interval:**
- `500ms`: High-frequency analysis, market microstructure
- `1s`: Standard granularity, most use cases ⭐
- `5s`: Low-frequency, longer timeframes

### 3. Timestamp Alignment - Combining Data Correctly

Both Binance and Polymarket data use `ts_recv` (receive timestamp) in epoch milliseconds.

**Alignment methods:**
- `asof_backward`: For each left timestamp, find latest right timestamp ≤ left (most common)
- `asof_forward`: For each left timestamp, find earliest right timestamp ≥ left
- `inner`: Only exact timestamp matches
- `outer`: All timestamps from both datasets (fills NaN where missing)

**Typical pattern:**
```python
combined = align_timestamps(
    left=pm,  # Polymarket updates (sparse)
    right=bnc,  # Binance reference (dense)
    method="asof_backward",  # Get latest BNC state for each PM update
)
```

### 4. Up Normalization - Polymarket Prices

**Important:** All Polymarket prices represent "Up" probability (0-1 scale).

The framework automatically normalizes:
- If you loaded the "Down" token, prices are flipped (1 - price)
- You always see P(Up) in your DataFrames
- Normalization happens transparently based on market outcome

**Example:**
```python
pm = load_polymarket_market("BTC", date(2026, 1, 19), 9, "1s")

# pm["mid"] is always P(Up), whether we loaded Up or Down token
print(f"P(Up) at start: {pm['mid'].iloc[0]:.3f}")  # e.g., 0.52
print(f"P(Up) at end: {pm['mid'].iloc[-1]:.3f}")   # e.g., 0.78 (if market went up)
```

### 5. No Forward-Looking Bias

**Critical for backtesting:** The framework prevents lookahead bias:

1. **Resampling:** Only uses past data (forward-fill)
2. **Alignment:** ASOF join ensures you only see past Binance state
3. **Normalization:** Uses alphabetically first token (no outcome peeking)

**Timestamp ordering:**
```python
combined = align_timestamps(pm, bnc, "asof_backward")

# Verify timestamps are monotonic (no lookahead)
assert (combined["ts_recv"].diff().dropna() >= 0).all()
```

---

## API Reference

### Load Binance Data

```python
load_binance(
    start: str | datetime,
    end: str | datetime,
    asset: Literal["BTC", "ETH"] = "BTC",
    interval: Literal["500ms", "1s", "5s"] = "1s",
    columns: list[str] | None = None,
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame
```

**Use for:** Continuous price history across any time range.

**Available columns:**
- `ts_recv`: Timestamp (epoch milliseconds)
- `bid_px`, `ask_px`: Best bid/ask prices (USD)
- `bid_sz`, `ask_sz`: Best bid/ask sizes
- `mid_px`: Mid price = (bid + ask) / 2
- `spread`: Spread = ask - bid

**Column selection example:**
```python
# Only load what you need
bnc = load_binance(
    "2026-01-19", "2026-01-20",
    "BTC", "1s",
    columns=["ts_recv", "mid_px", "spread"],  # Faster, less memory
)
```

**Time formats:**
```python
# String format
load_binance("2026-01-19 00:00:00", "2026-01-20 12:30:45", "BTC", "1s")

# Date only (assumes 00:00:00)
load_binance("2026-01-19", "2026-01-20", "BTC", "1s")

# Datetime objects
from datetime import datetime, timezone
load_binance(
    datetime(2026, 1, 19, tzinfo=timezone.utc),
    datetime(2026, 1, 20, tzinfo=timezone.utc),
    "BTC", "1s"
)
```

---

### Load Polymarket Market

```python
load_polymarket_market(
    asset: Literal["BTC", "ETH"],
    date: str | date | datetime,
    hour_et: int,  # 0-23, Eastern Time
    interval: Literal["500ms", "1s", "5s"] = "1s",
    cache_dir: Path | str | None = None,
    force_reload: bool = False,
) -> pd.DataFrame
```

**Use for:** Loading ONE specific hourly market.

**Important:** Markets are hourly and independent. Load one at a time.

**Available columns:**
- `ts_recv`: Timestamp (epoch milliseconds)
- `bid`, `ask`: Best bid/ask prices (0-1 probability)
- `bid_sz`, `ask_sz`: Best bid/ask sizes
- `mid`: Mid price (Up probability)
- `spread`: Bid-ask spread
- `microprice`: Size-weighted mid price

**Date formats:**
```python
from datetime import date, datetime

# String
load_polymarket_market("BTC", "2026-01-19", 9, "1s")

# date object
load_polymarket_market("BTC", date(2026, 1, 19), 9, "1s")

# datetime (uses .date())
load_polymarket_market("BTC", datetime(2026, 1, 19, 9, 0), 9, "1s")
```

**Hour conversion (ET to UTC):**
- 9am ET = 2pm UTC (winter) or 1pm UTC (summer)
- Framework handles timezone conversion automatically
- Just specify `hour_et` in Eastern Time

---

### Align Timestamps

```python
align_timestamps(
    left: pd.DataFrame,
    right: pd.DataFrame,
    method: Literal["asof_backward", "asof_forward", "inner", "outer"] = "asof_backward",
    left_suffix: str = "_left",
    right_suffix: str = "_right",
) -> pd.DataFrame
```

**Use for:** Combining Binance and Polymarket data by timestamp.

**Method guide:**
- `"asof_backward"`: Most common - get latest right value for each left timestamp
- `"asof_forward"`: Get next right value for each left timestamp
- `"inner"`: Only exact matches (usually results in empty DataFrame)
- `"outer"`: All timestamps from both (fills NaN where missing)

**Typical pattern:**
```python
combined = align_timestamps(
    left=pm,   # Sparse Polymarket updates
    right=bnc,  # Dense Binance reference
    method="asof_backward",
    left_suffix="_pm",
    right_suffix="_bnc",
)

# Result columns: ts_recv, bid_pm, ask_pm, mid_pm, mid_px_bnc, spread_bnc
```

---

### Cache Management

#### Check Cache Status

```python
get_cache_status(
    venue: Literal["binance", "polymarket"],
    asset: Literal["BTC", "ETH"] = "BTC",
    interval: Literal["500ms", "1s", "5s"] = "1s",
) -> dict
```

**Returns:**
- `dates_cached`: List of dates (YYYY-MM-DD strings)
- `date_range`: Tuple of (first_date, last_date)
- `total_rows`: Total cached rows
- `size_mb`: Total cache size in megabytes

**Example:**
```python
from marketdata import get_cache_status

status = get_cache_status("binance", "BTC", "1s")
print(f"Cached: {len(status['dates_cached'])} days")
print(f"Range: {status['date_range']}")
print(f"Size: {status['size_mb']:.1f} MB")
```

#### Clear Cache

```python
clear_cache(
    venue: Literal["binance", "polymarket"],
    asset: Literal["BTC", "ETH"] = "BTC",
    interval: Literal["500ms", "1s", "5s"] = "1s",
    before_date: str | date | None = None,
) -> int  # Number of files deleted
```

**Example:**
```python
from datetime import date, timedelta
from marketdata import clear_cache

# Keep last 30 days, delete older
cutoff = date.today() - timedelta(days=30)
deleted = clear_cache("binance", "BTC", "1s", before_date=cutoff)
print(f"Deleted {deleted} files")
```

---

## Common Recipes

### Recipe 1: Compute Bitcoin Realized Volatility

```python
import numpy as np
from marketdata import load_binance

# Load Bitcoin 1s data for one day
bnc = load_binance("2026-01-19", "2026-01-20", "BTC", "1s", columns=["ts_recv", "mid_px"])

# Compute log returns
returns = np.log(bnc["mid_px"]).diff().dropna()

# Realized volatility (annualized)
rv_daily = returns.std() * np.sqrt(86400)  # 86400 seconds in a day
rv_annual = rv_daily * np.sqrt(365)

print(f"Daily RV: {rv_daily:.4f} ({rv_daily*100:.2f}%)")
print(f"Annual RV: {rv_annual:.4f} ({rv_annual*100:.2f}%)")
```

### Recipe 2: Load Polymarket Market and Check Outcome

```python
from datetime import date
from marketdata import load_session

# Load session (includes outcome information)
session = load_session("BTC", date(2026, 1, 19), hour_et=9)

# Get Polymarket data
pm = session.polymarket_resampled("1s")

# Get market outcome
outcome = session.outcome

print(f"Market outcome: {outcome.outcome}")  # 'up', 'down', or 'flat'
print(f"Open: ${outcome.open_price:.2f}")
print(f"Close: ${outcome.close_price:.2f}")
print(f"Return: {outcome.return_pct:.2f}%")
print()
print(f"PM opening probability: {pm['mid'].iloc[0]:.3f}")
print(f"PM closing probability: {pm['mid'].iloc[-1]:.3f}")

# Verify Up normalization
if outcome.is_up:
    print("✓ Market went up, PM mid should increase")
    assert pm['mid'].iloc[-1] > pm['mid'].iloc[0], "PM mid should increase for up market"
```

### Recipe 3: Compare PM Probability vs BTC Returns

```python
from datetime import date
from marketdata import load_session
import pandas as pd

# Load multiple markets
markets = []
for hour in [9, 10, 11, 14, 15]:
    session = load_session("BTC", date(2026, 1, 19), hour)

    outcome = session.outcome
    if outcome is None:
        continue

    pm = session.polymarket_resampled("1s")
    if pm.empty:
        continue

    markets.append({
        "hour_et": hour,
        "opening_prob": pm["mid"].iloc[0],
        "closing_prob": pm["mid"].iloc[-1],
        "actual_return": outcome.return_pct,
        "outcome": outcome.outcome,
    })

df = pd.DataFrame(markets)
print(df)

# Correlation between opening prob and actual return
corr = df["opening_prob"].corr(df["actual_return"])
print(f"\nCorrelation: {corr:.3f}")
```

### Recipe 4: Build Simple Trading Signal

```python
from datetime import date
from marketdata import load_session

session = load_session("BTC", date(2026, 1, 19), hour_et=9)

# Get aligned data
aligned = session.aligned_resampled("1s")

# Simple signal: PM probability crosses 0.5
aligned["signal"] = 0
aligned.loc[aligned["mid_pm"] > 0.5, "signal"] = 1  # Long when P(Up) > 50%
aligned.loc[aligned["mid_pm"] < 0.5, "signal"] = -1  # Short when P(Up) < 50%

# Compute forward returns (for backtesting)
aligned["fwd_return"] = aligned["mid_px_bnc"].pct_change().shift(-1)

# Signal performance
signal_returns = aligned.groupby("signal")["fwd_return"].mean()
print("Signal performance:")
print(signal_returns)
```

### Recipe 5: Handle Missing Polymarket Markets

```python
from datetime import date
from marketdata import load_polymarket_market

def safe_load_market(asset, date, hour_et, interval="1s"):
    """Load market with graceful handling of missing data."""
    try:
        pm = load_polymarket_market(asset, date, hour_et, interval)
        if pm.empty:
            print(f"⚠️ No market for {date} hour {hour_et}")
            return None
        return pm
    except Exception as e:
        print(f"✗ Error loading {date} hour {hour_et}: {e}")
        return None

# Load markets, handling missing gracefully
markets = []
for hour in range(24):  # Try all hours
    pm = safe_load_market("BTC", date(2026, 1, 19), hour)
    if pm is not None:
        markets.append((hour, pm))

print(f"Found {len(markets)} markets")
```

### Recipe 6: Efficient Column Selection for Large Ranges

```python
from marketdata import load_binance

# Loading 30 days of all columns (slow, large memory)
# bnc_full = load_binance("2026-01-01", "2026-01-31", "BTC", "1s")  # DON'T DO THIS

# Better: Load only needed columns
bnc = load_binance(
    "2026-01-01", "2026-01-31",
    "BTC", "1s",
    columns=["ts_recv", "mid_px"],  # Only what you need
)

print(f"Loaded {len(bnc):,} rows")
print(f"Memory: {bnc.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Compute what you need
import numpy as np
returns = np.log(bnc["mid_px"]).diff()
volatility = returns.rolling(window=3600).std()  # 1-hour rolling vol

print(f"Mean volatility: {volatility.mean():.6f}")
```

---

## LLM Agent Guide

### Quick Reference

**If you need to...**

| Goal | Use |
|------|-----|
| Load Bitcoin price history | `load_binance()` |
| Load single Polymarket market | `load_polymarket_market()` |
| Combine Binance + Polymarket | `align_timestamps()` |
| Check what's cached | `get_cache_status()` |
| Clear old cache | `clear_cache()` |

### Common Mistakes to Avoid

❌ **Loading multiple Polymarket markets at once:**
```python
# WRONG - Polymarket loads ONE market
pm = load_polymarket_market("BTC", "2026-01-19", hours=[9,10,11], "1s")
```

✅ **Correct:**
```python
markets = []
for hour in [9, 10, 11]:
    pm = load_polymarket_market("BTC", date(2026, 1, 19), hour, "1s")
    markets.append(pm)
```

❌ **Using bid/ask instead of bid_px/ask_px for Binance:**
```python
# Column names differ between venues
bnc["bid"]  # Wrong - KeyError
pm["bid"]   # Correct
```

✅ **Correct:**
```python
bnc["bid_px"]  # Binance uses _px suffix
pm["bid"]      # Polymarket uses no suffix
```

❌ **Forgetting column selection for large ranges:**
```python
# Loads all 7 columns, wastes memory
bnc = load_binance("2026-01-01", "2026-01-31", "BTC", "1s")
```

✅ **Correct:**
```python
# Only load what you need
bnc = load_binance("2026-01-01", "2026-01-31", "BTC", "1s", columns=["ts_recv", "mid_px"])
```

### Example Prompts That Work Well

**Good prompts for LLM agents:**

1. "Load Bitcoin prices from Jan 19-20 2026 at 1s intervals and compute daily volatility"
2. "Load the 9am ET Polymarket market for BTC on Jan 19 2026 and show opening/closing probabilities"
3. "Align Polymarket and Binance data for the 9am market and show how PM probability evolved vs BTC price"
4. "Check what's cached for Binance BTC 1s data"

**Prompts that need clarification:**

1. "Load Polymarket data for Jan 19" - *Which hour? Markets are hourly*
2. "Get Bitcoin data" - *What time range? What interval?*
3. "Combine the data" - *Which Polymarket market? What alignment method?*

### Chaining Operations

**Pattern 1: Load → Analyze → Visualize**
```python
# Load
bnc = load_binance("2026-01-19", "2026-01-20", "BTC", "1s", columns=["ts_recv", "mid_px"])

# Analyze
returns = np.log(bnc["mid_px"]).diff()
vol = returns.std() * np.sqrt(86400)

# Visualize
import matplotlib.pyplot as plt
plt.plot(bnc["mid_px"])
plt.title(f"BTC Price (Daily Vol: {vol:.2%})")
plt.show()
```

**Pattern 2: Load Multiple Markets → Compare**
```python
# Load
markets = []
for hour in [9, 10, 11]:
    pm = load_polymarket_market("BTC", date(2026, 1, 19), hour, "1s")
    markets.append({"hour": hour, "opening_prob": pm["mid"].iloc[0]})

# Compare
df = pd.DataFrame(markets)
print(df)
```

---

## Performance Tips

### 1. Use Appropriate Interval

| Use Case | Recommended Interval |
|----------|---------------------|
| Quick prototyping | 5s |
| Standard analysis | 1s ⭐ |
| Market microstructure | 500ms |

**Example - 5s is 5x faster:**
```python
# Fast prototyping
bnc_5s = load_binance("2026-01-01", "2026-01-31", "BTC", "5s")  # ~500K rows, fast
# vs
bnc_1s = load_binance("2026-01-01", "2026-01-31", "BTC", "1s")  # ~2.5M rows, slower
```

### 2. Column Selection

**Memory savings:**
```python
# All columns: ~200 MB
bnc_full = load_binance("2026-01-01", "2026-01-31", "BTC", "1s")

# Selected columns: ~80 MB (60% savings)
bnc_mini = load_binance("2026-01-01", "2026-01-31", "BTC", "1s", columns=["ts_recv", "mid_px"])
```

### 3. Cache Management

**Keep cache size manageable:**
```python
from datetime import date, timedelta
from marketdata import clear_cache

# Monthly cleanup: keep last 60 days
cutoff = date.today() - timedelta(days=60)
clear_cache("binance", "BTC", "1s", before_date=cutoff)
clear_cache("polymarket", "BTC", "1s", before_date=cutoff)
```

### 4. Benchmark: Cache vs S3

**Typical speedups:**
| Data Source | Method | Time |
|-------------|--------|------|
| Binance 1 day, 1s | S3 (raw) | ~30s |
| Binance 1 day, 1s | Cache | ~0.5s |
| **Speedup** | | **60x** |

```python
import time
from marketdata import load_binance

# Cache hit (fast)
start = time.time()
bnc = load_binance("2026-01-19", "2026-01-20", "BTC", "1s")
print(f"Cached: {time.time() - start:.2f}s")

# Force reload (slow - hits S3)
start = time.time()
bnc = load_binance("2026-01-19", "2026-01-20", "BTC", "1s", force_reload=True)
print(f"S3: {time.time() - start:.2f}s")
```

---

## Troubleshooting

### Empty DataFrame Returned

**Symptom:**
```python
pm = load_polymarket_market("BTC", date(2026, 1, 19), 3, "1s")
print(len(pm))  # 0
```

**Causes & Fixes:**

1. **No market for that hour** (most common for Polymarket)
   - Polymarket markets only exist for certain hours (typically NYSE hours)
   - Check which hours have data:
   ```python
   for hour in range(24):
       pm = load_polymarket_market("BTC", date(2026, 1, 19), hour, "1s")
       if not pm.empty:
           print(f"Hour {hour}: {len(pm)} rows")
   ```

2. **Cache not populated**
   ```python
   # Sync cache first
   # See UPDATING_CACHE.md
   ```

3. **Date out of range**
   ```python
   status = get_cache_status("polymarket", "BTC", "1s")
   print(f"Available: {status['date_range']}")
   ```

### KeyError: 'ts_recv'

**Symptom:**
```python
combined = align_timestamps(pm, bnc, "asof_backward")
# KeyError: 'ts_recv'
```

**Cause:** One DataFrame doesn't have `ts_recv` column.

**Fix:**
```python
# Check columns
print("PM columns:", list(pm.columns))
print("BNC columns:", list(bnc.columns))

# Ensure both have ts_recv
assert "ts_recv" in pm.columns
assert "ts_recv" in bnc.columns
```

### Market Doesn't Exist Error

**Symptom:**
```python
pm = load_polymarket_market("BTC", date(2026, 1, 19), 23, "1s")
# Returns empty DataFrame or error
```

**Cause:** Market doesn't exist for that hour.

**Solution:** Use try/except and check:
```python
try:
    pm = load_polymarket_market("BTC", date(2026, 1, 19), 23, "1s")
    if pm.empty:
        print("No market for hour 23")
except Exception as e:
    print(f"Error: {e}")
```

### S3 Connection Errors

**Symptom:**
```python
# IO Error, Failed to read, Connection error
```

**Solutions:**

1. Check internet connection
2. Verify S3 credentials are configured
3. Try again (transient errors)
4. Sync script has built-in retry logic

---

## Related Documentation

- [UPDATING_CACHE.md](UPDATING_CACHE.md) - How to keep cache up to date
- [README.md](../README.md) - Project overview
- [Plan](../.claude/plans/) - Implementation plan details

---

## Questions?

For issues or questions:
1. Check this guide's troubleshooting section
2. Review UPDATING_CACHE.md for cache-related issues
3. Verify cache status with `get_cache_status()`
4. Check cache metadata files in `data/resampled_data/`
