# Debug Audit Report

**Date**: 2026-01-27
**Dataset**: BTC Binance, 2026-01-18 13:00 → 2026-01-27 20:00 UTC
**Samples**: 159,942 calibration rows across 223 market hours

---

## Check Results

| # | Check | Result |
|---|-------|--------|
| 1 | Unit consistency of τ, Δ, σ | **FAIL** — σ_base was ~3× too small |
| 2 | Correct probability tail | PASS |
| 3 | Settlement/label alignment | PASS |
| 4 | Feature leakage / lookahead | PASS |
| 5 | Dataset sampling | PASS (~720 samples/hour, uniform) |
| 6 | Student-t implementation | PASS |
| 7 | Scale mismatch diagnostic | **FAIL** — confirmed σ_eff / empirical ratio ≈ 0.32 |

---

## Root Cause

**σ_tod was computed on 100ms BBO mid returns, which are 97% zeros** due to tick-size quantization. This caused:

1. MAD estimator returned 0 for most buckets → hit floor (1e-10)
2. Switching to RV (`sqrt(mean(r²))`) still underestimated because zero-inflated 100ms returns don't reflect true variance
3. EWMA `v_k` decayed toward 0 (most `u_k² ≈ 0`), so `σ_rel` deflated σ_tod further

The optimizer compensated by pushing ν → 2.14 (maximum fat tails), which artificially widened the distribution to match observed outcomes — but at the cost of poor calibration.

---

## Patch Summary

Three changes fixed the issue:

### 1. Seasonal vol: 10-second sub-intervals with RV estimator
`features/seasonal_vol.py` — Changed from MAD on 100ms returns to RV on 10-second sub-interval returns. Added `sub_interval_sec` field to `SeasonalVolCurve` dataclass for unit tracking.

### 2. EWMA: proper unit conversion
`features/ewma_shock.py` — σ_tod is now in per-√(10s) units. Added conversion:
```
s_per_sec = s_tod / sqrt(sub_interval_sec)
s_step = s_per_sec * sqrt(delta_sec)
```

### 3. EWMA: floor σ_rel at 1.0
`features/ewma_shock.py` — Prevents zero-inflated 100ms returns from deflating vol below the seasonal baseline:
```python
sr = max(np.sqrt(v_k), 1.0)
```

---

## Updated Metrics

|  | Before fix | After fix |
|--|-----------|-----------|
| Log loss | 0.591 | **0.494** |
| Brier score | — | 0.168 |
| Accuracy | — | 74.6% |
| ν | 2.14 (abused) | 82.5 (near-Gaussian) |
| γ | ≈0 | 4.5e-5 |
| c | 3.5 | 3.51 |

The near-Gaussian ν confirms the scale bug was the primary issue — the optimizer no longer needs fat tails to compensate for undersized σ.

---

## Calibration Plot

Updated plot saved to `pricer_calibration/output/predicted_vs_actual.html`.
