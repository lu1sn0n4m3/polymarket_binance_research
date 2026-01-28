# task.md — Implementation Plan for `estimation.tex`

This document decomposes the whitepaper in `estimation.tex` into concrete engineering tasks suitable for a coding agent. The goal is an end-to-end, reproducible pipeline that:

1) constructs a 100ms mid-price grid from Binance L1 (bid/ask),
2) estimates a seasonal volatility curve using MAD,
3) computes live EWMA volatility and a shock statistic on the same grid,
4) prices an hourly digital `1{S_T > K}` using an abstract CDF specialized to Student‑t,
5) calibrates a small parameter set with Bernoulli log-likelihood + optional L2 penalty,
6) outputs live probability and sensitivities.

Assumptions:
- You have two weeks of Binance best bid/ask updates with timestamps.
- You know contract definitions: for each “market hour” you have `(T, K)` and can label `Y = 1{S_T > K}` from Binance.
- Everything is in log space; returns are log returns.
- All estimators run on a fixed 100ms grid.
- Seasonal volatility is computed once on the full dataset (intentional look-ahead).

---

## Repository layout (suggested)

```
/pricer_calibration
  /data
    ingest.py
    grid.py
    labels.py
  /features
    seasonal_vol.py
    ewma_shock.py
  /model
    pricer.py
    calibration.py
    metrics.py
  /run
    build_dataset.py
    fit_params.py
    backtest_pricer.py
/config
  config.yaml
/tests
  test_grid.py
  test_seasonal_vol.py
  test_ewma_shock.py
  test_pricer.py
  test_calibration.py
task.md
README.md
```

Implement in Python. Use `numpy`, `pandas`, and `scipy` (`scipy.stats.t`, `scipy.optimize.minimize`), plus `pyarrow` for parquet if desired.

---

## Global conventions

### Time and grid
- All timestamps in UTC.
- Grid interval: `Δ = 0.1 seconds`.
- Grid times: `t_k = t0 + k*Δ` for each continuous segment (or full range).

### Prices and returns
- Mid price: `m(t) = (bid(t) + ask(t))/2`.
- Grid mid: `S_k = m(t_k^-)` via previous-tick sampling (forward-fill).
- Log mid: `ℓ_k = log(S_k)`.
- Log return: `r_k = ℓ_k - ℓ_{k-1}`.

### Bucketing time-of-day
- Choose a time-of-day bucket width: default `5 minutes`.
- Total buckets per day: `B = 24*60/5 = 288`.
- Bucket index: `b(t_k)` based on `(hour, minute)`.

### Units
- Seasonal vol `σ_tod(b)` is the **standard deviation of 100ms log returns**.
- Normalized return `u_k = r_k / σ_tod(b_k)` (dimensionless).
- EWMA tracks `v_k ≈ E[u^2]` so `σ_rel,k = sqrt(v_k)` (dimensionless).
- Live σ for 100ms return: `σ_Δ,k = σ_tod(b_k) * σ_rel,k`.
- For pricing over horizon τ (seconds), convert to per-√sec:
  - `σ_base(t_k) = σ_Δ,k / sqrt(Δ)`.

---

## Task 0 — Config & CLI scaffolding

### Goal
Centralize constants and make runs reproducible.

### Deliverables
- `config/config.yaml` with:
  - grid interval `delta_ms: 100`
  - tod bucket `tod_bucket_minutes: 5`
  - ewma half-life `ewma_half_life_seconds: 10|20|30` (pick one default)
  - shock window steps `shock_M: 5` (500ms default)
  - shock hinge `c: 3.5` default
  - pricer: `dist: student_t`, `nu: 6` default (or `inf` for normal)
  - calibration: `lambda_l2`, bounds, initial guesses
  - dataset sampling schedule (e.g., 5s).
- CLI entrypoints:
  - `build_dataset.py`
  - `fit_params.py`
  - `backtest_pricer.py`

### Acceptance
- Single command can rebuild derived data from raw input with deterministic output.

---

## Task 1 — Binance data ingestion and cleaning

### Goal
Load Binance best bid/ask updates into a canonical dataframe.

### Inputs
Raw file(s) with at least:
- timestamp (ns/us/ms resolution; include timezone or assume UTC),
- best bid price,
- best ask price.

### Steps
1. Parse timestamps and sort ascending.
2. Drop rows with missing/invalid bid/ask.
3. Enforce `bid <= ask` (drop or fix obvious glitches).
4. Compute mid `m = (bid + ask)/2`.
5. Optional: de-duplicate identical consecutive bid/ask updates.

### Outputs
- `binance_l1.parquet` (or equivalent) with columns:
  - `ts` (datetime64[ns], UTC),
  - `bid`, `ask`, `mid`.

### Acceptance
- No negative or zero mids.
- Time strictly non-decreasing.
- Basic sanity stats printed (rows, start/end time, mean spread).

---

## Task 2 — Build the 100ms grid (previous-tick sampling)

### Goal
Create `(t_k, S_k, ℓ_k, r_k)` at 100ms.

### Steps
1. Determine `t0` and `t_end`:
   - recommended: align to whole 100ms boundaries (floor/ceil).
2. Build grid timestamps `t_k` using `pd.date_range` with `freq="100ms"`.
3. Previous-tick sample:
   - use `merge_asof` (left join grid onto events, direction="backward") to get latest `mid` at or before `t_k`.
4. Forward-fill any gaps at the start (or drop initial rows until first observation exists).
5. Compute `ℓ_k = log(S_k)`.
6. Compute `r_k = diff(ℓ_k)` and drop the first row (or set `r_0 = 0`).

### Outputs
- `grid_100ms.parquet` with columns:
  - `t`, `S`, `logS`, `r`.

### Acceptance
- Grid has constant spacing.
- No NaNs after the warmup cut.
- Returns series is finite.

---

## Task 3 — Compute seasonal volatility curve σ_tod via MAD

### Goal
Estimate baseline intraday (time-of-day) volatility on the 100ms return series.

### Steps
1. Define bucket index `b_k = floor((hour*60 + minute)/bucket_minutes)` (0..B-1).
2. For each bucket `b`:
   - collect all returns `r_k` where `b_k == b`.
   - compute `med = median(r)`
   - compute `MAD = median(|r - med|)`
   - compute `σ_tod(b) = 1.4826 * MAD`
3. Handle edge cases:
   - if a bucket has too few samples, interpolate from neighbors.
4. Optional smoothing:
   - circular moving average across buckets, e.g., window 3–7 buckets.

### Outputs
- `seasonal_vol.parquet` or `.json` with:
  - `bucket`, `sigma_tod`, optionally `sigma_tod_smoothed`.
- Utility function:
  - `sigma_tod_at_time(t)` returning σ_tod for that time’s bucket.

### Acceptance
- σ_tod strictly positive (apply small floor if needed).
- Clear intraday shape in quick plot (optional).
- Matches the whitepaper: MAD * 1.4826.

---

## Task 4 — Compute normalized returns u_k, EWMA v_k, σ_rel,k, σ_Δ,k

### Goal
Live-style volatility estimation on the same 100ms grid.

### Inputs
- `grid_100ms` (with `r_k`),
- `σ_tod(b_k)` function.

### Steps
1. For each row, compute `b_k` and `σ_tod_k = σ_tod(b_k)`.
2. Compute normalized returns:
   - `u_k = r_k / σ_tod_k`.
3. EWMA recursion on `u_k^2`:
   - choose half-life `H` seconds; with `Δ=0.1`:
     - `α = 1 - exp(-(ln2)*Δ/H)`
   - initialize `v_0 = 1`
   - update: `v_k = (1-α)*v_{k-1} + α*(u_k^2)`
4. Compute:
   - `σ_rel,k = sqrt(v_k)`
   - `σ_Δ,k = σ_tod_k * σ_rel,k`
   - optionally `σ_base,k = σ_Δ,k / sqrt(Δ)` (per √sec)
5. Recommended safety:
   - winsorize `u_k^2` at a high quantile (e.g., cap at 100) so data spikes don’t pollute v_k for too long.

### Outputs
- `features_100ms.parquet` with:
  - `t, S, logS, r, bucket, sigma_tod, u, v, sigma_rel, sigma_delta, sigma_base`

### Acceptance
- In calm hours `sigma_rel` hovers around 1.
- `sigma_rel` rises during volatile episodes.
- No NaNs.

---

## Task 5 — Compute shock statistic z_k

### Goal
Fast instability proxy on normalized returns.

### Steps
1. Choose `M` grid steps (e.g., `M=5` for 500ms).
2. Compute rolling maximum:
   - `z_k = rolling_max(|u_k|, window=M)`
   - align so z_k at time k uses data up to and including k (no lookahead).
3. Fill first `M-1` z-values sensibly (e.g., increasing window or NaN then drop).

### Outputs
- Add `z` column to `features_100ms`.

### Acceptance
- z spikes quickly on jumps.
- No additional sqrt(Δ) factor is used (u already dimensionless).

---

## Task 6 — Define the pricer p(t) with abstract CDF and Student‑t specialization

### Goal
Compute fair probability `p*` for an hourly digital contract at any time.

### Inputs (at time t_k)
- Current price `S`,
- Time-to-expiry `τ = T - t_k` (seconds),
- Strike `K`,
- `σ_eff(t_k)` from estimator + shock inflation,
- Distribution choice: Student‑t ν (or Normal).

### Steps
1. Shock inflation hinge:
   - `κ(z) = 1 + γ*max(0, z - c)`
2. Effective volatility:
   - `σ_eff = σ_base * κ(z)`  (σ_base is per √sec)
3. Standardized threshold:
   - `x = ln(K/S) / (σ_eff * sqrt(τ))`
4. Probability:
   - **Correct tail**: `p = P(S_T > K) = P(X > x) = 1 - F(x)`
   - Student‑t: `p = 1 - t_cdf(x; ν)`
   - Normal: `p = 1 - Φ(x)`

### Outputs
- Function `price_probability(S, K, tau, sigma_base, z, params) -> p`

### Acceptance
- p increases with S.
- p in (0,1), stable for large |x| (clip x if needed).

---

## Task 7 — Sensitivities: dp/dS, dp/dlnS, and finite shocks Δp±(h)

### Goal
Provide risk proxies needed by quoting logic.

### Local derivative
Given `p = 1 - F(x)` and `x = ln(K/S)/(σ_eff*sqrt(τ))` with σ_eff treated constant w.r.t. infinitesimal S:
- `dx/dS = -1 / (S * σ_eff * sqrt(τ))`
- `dp/dS = f(x) / (S * σ_eff * sqrt(τ))` where `f` is PDF of the chosen distribution.
- `dp/dlnS = S * dp/dS = f(x) / (σ_eff * sqrt(τ))`.

Implement:
- Student‑t pdf: `scipy.stats.t.pdf(x, df=ν)`
- Normal pdf: `scipy.stats.norm.pdf(x)`.

### Finite-horizon one-sided shocks
For horizon `h` seconds:
1. Obtain adverse log-move magnitudes `q_+(h), q_-(h)`:
   - simplest first pass: empirical constants based on normalized return quantiles.
2. Compute:
   - `Δp_+ = p(S*exp(q_+)) - p(S)`
   - `Δp_- = p(S*exp(-q_-)) - p(S)`

Deliver functions:
- `dp_dS(...)`, `dp_dlogS(...)`, `delta_p_one_sided(...)`

Acceptance:
- dp/dS positive.
- Δp_+ ≥ 0, Δp_- ≤ 0 (typically).

---

## Task 8 — Build calibration dataset (sampling schedule per hour)

### Goal
Create independent-ish training rows to fit θ.

### Inputs
- `features_100ms` with `S, sigma_base, z, t`.
- Contract table for each market hour: `(market_id, K, T, hour_start, hour_end)`.

### Steps
For each market hour:
1. Determine candidate times `t_k` in `(hour_start, T)` (or `(hour_start, hour_end)` depending on definition).
2. Downsample decision times to avoid fake sample size:
   - default: every 5 seconds (i.e., take every 50th 100ms row),
   - optionally: time-to-expiry schedule (5s early, 1–2s in last 10 minutes).
3. For each sampled time:
   - compute `tau = (T - t).total_seconds()`
   - record `S, tau, sigma_base, z, K, market_id`
4. Label:
   - compute `Y = 1{S_T > K}` using grid price at settlement time:
     - find `S_T` by previous-tick sampling at `T` from the grid.
5. Store rows.

### Outputs
- `calibration_dataset.parquet` with:
  - `market_id, t, S, tau, K, sigma_base, z, y`

### Acceptance
- No lookahead in features (only seasonal curve uses intentional lookahead).
- Sample count per hour matches config (e.g., ~720 rows per hour at 5s).
- Label is constant within each market_id.

---

## Task 9 — Calibration optimizer: Bernoulli log-likelihood + L2

### Goal
Fit θ to minimize penalized log loss.

### Parameters (up to 3)
Use the whitepaper’s recommended small set:
- `ν` (df of Student‑t), optionally fixed
- `γ` (shock inflation slope)
- `c` (shock threshold)

Recommendation for first pass:
- Fix `c` (e.g., 3.5) and maybe fix `ν` (e.g., 6 or ∞), optimize only `γ`.

### Loss
For each row i:
- compute `κ(z_i)`, `σ_eff_i = σ_base_i * κ`
- compute `x_i = ln(K_i/S_i)/(σ_eff_i*sqrt(tau_i))`
- compute `p_i = 1 - F(x_i; ν)`
- log loss:
  - `ℓ_i = -[ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]`
Add L2:
- `λ * ||θ - θ0||^2` or `λ * ||θ||^2`.

### Implementation details
- Clip probabilities: `p = clip(p, 1e-9, 1-1e-9)` to avoid log(0).
- Parameter constraints:
  - `γ >= 0`
  - `c > 0`
  - `ν > 2` (if using Student‑t with finite variance; optional but recommended)
- Use `scipy.optimize.minimize` with bounds.
- Consider reparameterizing:
  - optimize `log_gamma = log(γ + eps)` and map back
  - optimize `log_nu_minus2 = log(ν - 2)`.

### Outputs
- `params.json` with fitted values and metadata (train window, config hash).
- Print log loss on validation window.

### Acceptance
- Optimization converges.
- Parameters are stable under small data perturbations.
- Out-of-sample calibration improves vs baseline (e.g., γ=0).

---

## Task 10 — Live pricer module

### Goal
Given a stream of 100ms grid updates, output `p*` and sensitivities per market.

### Interface
- Inputs per tick:
  - `t, bid, ask`
- State:
  - seasonal curve σ_tod(b)
  - EWMA v_k
  - shock z_k (rolling max)
  - fitted parameters θ
- Methods:
  - `update_tick(t, bid, ask)` updates grid series and features
  - `price_market(K, T)` returns:
    - `p*`, `dp/dS`, `dp/dlnS`, `Δp_+(h)`, `Δp_-(h)` (if q± configured)

### Acceptance
- Deterministic, low-latency computations.
- No dependence on future data.
- Matches backtest results when replaying the feed.

---

## Task 11 — Tests & validation checks

### Must-have unit tests
- Grid correctness:
  - previous-tick sampling matches a reference implementation.
- MAD computation:
  - known small vector returns expected MAD and sigma scaling.
- EWMA:
  - α derived from half-life matches formula; recursion matches expected values.
- Shock:
  - rolling max window works, no lookahead.
- Pricer:
  - monotonicity in S (higher S → higher p)
  - correct tail direction: p = 1 - F(x)
  - dp/dS positive
- Calibration:
  - log loss decreases vs baseline on train
  - parameter bounds enforced.

### Diagnostics scripts (optional but recommended)
- Plot σ_tod curve (intraday seasonality).
- Plot σ_rel over time and distribution by hour-of-day.
- Reliability curve / calibration plot for predicted p vs realized y.
- Log loss by time-to-expiry bucket.

---

## Notes for future extensions (do NOT implement in v1 unless needed)
- Subsampling offsets for vol estimation (multiple 100ms phase offsets) to further reduce grid-alignment artifacts.
- Separate calm vs shock quantiles for q±(h).
- Allow a second inflation slope or a smooth κ(z) (e.g., exp(γ(z-c))) if hinge is too crude.
- Incorporate drift only if calibration shows systematic bias; start with zero drift.

---

## Definition of done (v1)
You are done when:
- You can run a full pipeline from raw Binance bid/ask to:
  - `seasonal_vol`, `features_100ms`, `calibration_dataset`, fitted `params`, and
  - a callable live pricer returning `p*` and sensitivities for any `(K,T)` hour market.
- The probability behaves sensibly:
  - monotone in S,
  - stable in calm conditions,
  - responds to shocks via σ_eff inflation.
- Calibration uses Bernoulli log-likelihood (not least squares) and can fit 1–3 parameters.
