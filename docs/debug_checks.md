# debug_checks.md — Request to Audit Likely Implementation Errors

Context:
- 98M Binance BBO rows → 8M 100ms grid points → 223 market hours → ~160K calibration rows
- Fit results: ν ≈ 2.14 (very heavy tails), γ ≈ 0, c ≈ 3.5
- Calibration curve shows classic “too-extreme probabilities” (overconfident tails)
- Log loss improved vs γ=0 baseline, but reliability remains poor in extreme bins

This pattern strongly suggests a **scale/units/label alignment bug** or a systematic mis-specification that forces the optimizer to abuse ν while γ becomes irrelevant. Please audit the following items **in order**, and add explicit tests/prints to confirm each one.

---

## 1) Unit consistency of τ, Δ, and σ

### What to check
- `Δ` grid step is 0.1 seconds everywhere.
- `τ = T - t` is expressed in **seconds**, not minutes or milliseconds.
- Seasonal volatility `σ_tod` is defined as **stdev of 100ms log returns**.
- `σ_base` used in pricing is per √second: `σ_base = σ_Δ / sqrt(Δ)`.

### Required assertions / prints
For random sample rows:
- print `Δ`, `τ`, `σ_tod`, `σ_rel`, `σ_Δ`, `σ_base`.
- compute implied stdev to expiry: `stdev_to_expiry = σ_eff * sqrt(τ)` (must be in log-return units).
- sanity: for τ near 3600s, stdev_to_expiry should be on the order of typical hourly log-move (not 10× too small).

### Common failure modes
- Using τ in minutes with σ in per √sec (or vice versa).
- Forgetting the `/sqrt(Δ)` conversion (or doing it twice).
- Using σ_tod that already embeds √Δ scaling but still dividing by √Δ again.

---

## 2) Correct probability tail (no sign / inversion bugs)

### What to check
Pricing must implement:
- `x = ln(K/S) / (σ_eff * sqrt(τ))`
- `p = P(S_T > K) = P(X > x) = 1 - F(x)`

### Required tests
- Monotonicity: hold `(K, τ, σ_eff)` fixed; increase `S` → `p` must increase.
- Edge cases:
  - If `S >> K` and τ small, `p` should be near 1.
  - If `S << K` and τ small, `p` should be near 0.
- Compare with known Normal special case when ν→∞.

### Common failure modes
- Using `ln(S/K)` instead of `ln(K/S)` (or swapping and also swapping tail, masking the error).
- Using `p = F(x)` instead of `1 - F(x)`.

---

## 3) Settlement / label alignment and timestamp correctness

### What to check
- Settlement timestamp `T` (UTC) matches the contract’s definition exactly.
- Label uses the same rule as contract:
  - `>` vs `>=`
  - rounding rules (if any)
  - index price vs mid price
- `S_T` is sampled correctly on the grid:
  - `S_T = S(t = T^-)` via previous-tick sampling (no future leakage).

### Required diagnostics
- For each market hour, record:
  - actual settlement grid timestamp used
  - time delta between true T and nearest grid point
- Count “near-boundary” cases where `|S_T - K|/K` is tiny; label noise here can be huge.

### Common failure modes
- Off-by-one hour/day due to timezone conversion.
- Using the *next* tick after T instead of the previous tick.
- Using mid vs mark/index mismatch relative to contract.

---

## 4) Feature leakage or unintended lookahead in live features

### What to check
- Grid construction uses only backward asof join.
- EWMA uses `u_k` at time k and past state only (no centered rolling windows).
- Shock `z_k` is rolling max over the **past** M points including current point only.

### Required tests
- Ensure rolling window uses `.rolling(window=M).max()` aligned right, not centered.
- Verify no `.shift(-1)` or future data enters u/v/z.

---

## 5) Dataset sampling and “fake sample size”

### What to check
- Per-hour sampling schedule for calibration rows:
  - e.g. every 5s (recommended), not every 100ms.
- Ensure each market hour contributes reasonably, not dominated by certain hours with denser sampling.

### Required diagnostics
- Print histogram: samples per market hour.
- Optionally weight each hour equally (or cap per-hour samples) and re-fit to see if parameters change materially.

---

## 6) Student-t implementation details

### What to check
- Using `scipy.stats.t.cdf(x, df=ν)` and `t.pdf` correctly.
- Numerical stability:
  - clip p to [1e-9, 1-1e-9] before log.
- Ensure ν constraints are enforced (ν > 2 if assumed finite variance) or reparameterize.

### Diagnostic
- Evaluate p for representative x values: x ∈ {-5, -2, 0, 2, 5} for fitted ν and compare with Normal.

---

## 7) Quick “scale mismatch” diagnostic against empirical terminal moves

### What to check
For a coarse set of τ buckets (e.g. 0–5m, 5–15m, 15–30m, 30–60m):
- compute empirical distribution of `R = ln(S_T/S_t)` from calibration rows.
- compute model-implied stdev: `σ_eff * sqrt(τ)` for same rows.
- compare empirical stdev (or robust scale like MAD) vs model-implied.

If empirical scale >> model scale, the pricer will produce overly extreme p (exactly what we see).

---

## Deliverable
After auditing, produce a short report:
- Which checks passed/failed
- Any discovered unit/scale/label mismatch
- A minimal patch (code diff) for the top issue(s)
- Re-run calibration and attach updated metrics + calibration plot
