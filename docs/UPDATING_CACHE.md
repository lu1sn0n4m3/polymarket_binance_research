# Updating Your Local Resampled Data Cache

This guide explains how to periodically update your local resampled data cache from S3.

## Quick Start - Daily Update

Run this daily (manually or via cron) to keep your cache up to date:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance,polymarket \
    --asset BTC \
    --interval 1s \
    --last-days 7 \
    --incremental
```

This fetches the last 7 days, only downloading missing data (incremental mode).

---

## Common Use Cases

### 1. Initial Bulk Load

First-time setup to load historical data:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance,polymarket \
    --asset BTC \
    --interval 1s \
    --start-date 2026-01-01 \
    --end-date 2026-02-11 \
    --incremental \
    --delay 2.0
```

**Options:**
- `--venue`: Which data sources to sync (`binance`, `polymarket`, or both)
- `--asset`: Asset symbol (`BTC` or `ETH`)
- `--interval`: Resampling interval (`500ms`, `1s`, or `5s`)
- `--incremental`: Skip already cached dates
- `--delay`: Seconds to wait between days (helps avoid S3 throttling)

### 2. Sync Specific Date Range

Load data for a specific period:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance \
    --asset BTC \
    --interval 1s,5s \
    --start-date 2026-01-19 \
    --end-date 2026-01-25 \
    --incremental
```

### 3. Daily Maintenance (Last Week)

Keep your cache fresh with recent data:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance,polymarket \
    --asset BTC \
    --interval 1s \
    --last-days 7 \
    --incremental
```

### 4. Force Reload (Cache Corruption)

If you suspect cache corruption, force reload:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance \
    --asset BTC \
    --interval 1s \
    --start-date 2026-01-19 \
    --end-date 2026-01-20 \
    --force
```

**Warning:** `--force` ignores the cache and reloads everything from S3.

---

## Cron Job Setup (Automated Daily Updates)

### Linux/Mac

Add to your crontab (runs daily at 2am):

```bash
# Open crontab editor
crontab -e

# Add this line (adjust paths):
0 2 * * * cd /home/user/polymarket_binance_research && /home/user/venv/bin/python scripts/sync_resampled_cache.py --venue binance,polymarket --asset BTC --interval 1s --last-days 7 --incremental >> logs/sync.log 2>&1
```

### Create logs directory

```bash
mkdir -p logs
```

### View sync logs

```bash
tail -f logs/sync.log
```

---

## Checking Cache Status

### Python API

```python
from src.data import get_cache_status

# Check Binance cache
status = get_cache_status("binance", "BTC", "1s")
print(f"Cached: {len(status['dates_cached'])} days")
print(f"Range: {status['date_range']}")
print(f"Size: {status['size_mb']:.1f} MB")

# Check Polymarket cache
pm_status = get_cache_status("polymarket", "BTC", "1s")
print(f"\nPolymarket cached: {len(pm_status['dates_cached'])} days")
```

### View cache directory

```bash
# See cached Binance data
ls -lh data/resampled_data/binance/asset=BTC/interval=1s/

# See cached Polymarket data
ls -lh data/resampled_data/polymarket/asset=BTC/interval=1s/

# Check disk usage
du -sh data/resampled_data/
```

### Metadata files

Each cache directory has a `.metadata.json` file with coverage information:

```bash
# View Binance metadata
cat data/resampled_data/binance/asset=BTC/interval=1s/.metadata.json | jq

# View Polymarket metadata
cat data/resampled_data/polymarket/asset=BTC/interval=1s/.metadata.json | jq
```

---

## Troubleshooting

### S3 Throttling Errors

**Symptom:** `IO Error`, `SlowDown`, or `429` errors

**Solution:** Increase the delay between days:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance,polymarket \
    --asset BTC \
    --interval 1s \
    --last-days 7 \
    --incremental \
    --delay 5.0  # Increased from default 2.0s
```

### Missing Polymarket Hours

**Symptom:** Polymarket has fewer rows than expected

**This is normal!** Polymarket markets only exist for certain hours (typically NYSE trading hours). Not all 24 hours have markets.

Check which hours have data:

```python
from src.data import get_cache_status

status = get_cache_status("polymarket", "BTC", "1s")
# Check available_dates to see coverage
```

### Cache Corruption

**Symptom:** Load errors or unexpected data

**Solution 1:** Delete corrupted files and re-sync:

```bash
# Delete specific date
rm data/resampled_data/binance/asset=BTC/interval=1s/date=2026-01-19.parquet

# Re-fetch
python scripts/sync_resampled_cache.py \
    --venue binance \
    --asset BTC \
    --interval 1s \
    --start-date 2026-01-19 \
    --end-date 2026-01-20 \
    --incremental
```

**Solution 2:** Clear and rebuild entire cache:

```bash
# Clear cache for specific venue/asset/interval
rm -rf data/resampled_data/binance/asset=BTC/interval=1s/

# Rebuild
python scripts/sync_resampled_cache.py \
    --venue binance \
    --asset BTC \
    --interval 1s \
    --start-date 2026-01-01 \
    --end-date 2026-02-11 \
    --incremental
```

### Network Errors

**Symptom:** `Connection error`, `Failed to read` errors

**Solution:** The script has built-in retry logic (3 attempts per day). If errors persist:

1. Check your internet connection
2. Verify S3 credentials are configured
3. Try again later (temporary S3 issues)

---

## Advanced Usage

### Multiple Assets and Intervals

Sync BTC and ETH at multiple intervals:

```bash
python scripts/sync_resampled_cache.py \
    --venue binance \
    --asset BTC,ETH \
    --interval 1s,5s \
    --last-days 7 \
    --incremental
```

This syncs 4 combinations:
- BTC 1s
- BTC 5s
- ETH 1s
- ETH 5s

### Parallel Syncing

For faster syncing, run multiple processes in parallel:

```bash
# Terminal 1: Binance
python scripts/sync_resampled_cache.py \
    --venue binance \
    --asset BTC \
    --interval 1s \
    --last-days 30 \
    --incremental &

# Terminal 2: Polymarket
python scripts/sync_resampled_cache.py \
    --venue polymarket \
    --asset BTC \
    --interval 1s \
    --last-days 30 \
    --incremental &

# Wait for both to complete
wait
```

---

## Cache Management

### Check Cache Size

```bash
# Total cache size
du -sh data/resampled_data/

# Per-venue size
du -sh data/resampled_data/binance/
du -sh data/resampled_data/polymarket/

# Per-asset size
du -sh data/resampled_data/*/asset=BTC/
du -sh data/resampled_data/*/asset=ETH/
```

### Clear Old Data

```python
from datetime import date, timedelta
from src.data import clear_cache

# Keep last 30 days, delete older
cutoff = date.today() - timedelta(days=30)

# Clear Binance old data
deleted = clear_cache("binance", "BTC", "1s", before_date=cutoff)
print(f"Deleted {deleted} Binance files")

# Clear Polymarket old data
deleted = clear_cache("polymarket", "BTC", "1s", before_date=cutoff)
print(f"Deleted {deleted} Polymarket files")
```

### Expected Cache Sizes

Approximate sizes per day:

| Venue      | Interval | Size/Day |
|------------|----------|----------|
| Binance    | 500ms    | ~5 MB    |
| Binance    | 1s       | ~3 MB    |
| Binance    | 5s       | ~0.6 MB  |
| Polymarket | 500ms    | ~2 MB    |
| Polymarket | 1s       | ~1 MB    |
| Polymarket | 5s       | ~0.2 MB  |

**Note:** Polymarket is smaller because only certain hours have markets.

---

## Best Practices

1. **Run incremental updates daily** - Keeps cache fresh without re-downloading everything
2. **Use appropriate delay** - Default 2s is good, increase if you hit throttling
3. **Monitor logs** - Set up cron job to log output for debugging
4. **Check cache status periodically** - Ensure coverage is complete
5. **Clean old data** - If disk space is limited, clear data older than your analysis window

---

## Related Documentation

- [USER_GUIDE.md](USER_GUIDE.md) - Complete guide to using the resampled data API
- [README.md](../README.md) - Project overview and setup

---

## Questions?

If you encounter issues:
1. Check this troubleshooting section
2. Review sync script output for error messages
3. Verify S3 credentials and connectivity
4. Check cache metadata files for corruption
