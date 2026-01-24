# Simple Gaussian Pricer (Capped Fast/Slow EWMA + Time-of-Day Prior)
*Implementation-focused spec (no code), designed so an engineer can build it directly.*
 
## 0) Goal and payoff
You have a 1-hour market with:
- **Open** price \(O = S_0\) (Binance reference at hour start)
- **Close** price \(C = S_T\) at \(T=1\text{ hour}\)
- **Up** contract pays \(1\) if \(C \ge O\), else \(0\)
 
Work in log prices:
- \(X_t = \log S_t\)
- Realized log return since open: \(r(t)=X_t - X_0=\log(S_t/O)\)
 
At any time \(t\in[0,T]\), the fair probability is:
\[
p_t = \mathbb{P}(C\ge O \mid \mathcal{F}_t)=\mathbb{P}(r(T)\ge 0\mid\mathcal{F}_t)
= \mathbb{P}(r_{t\to T} \ge -r(t)\mid \mathcal{F}_t)
\]
 
**Model choice (simple Gaussian):**
\[
r_{t\to T}\mid\mathcal{F}_t \sim \mathcal{N}(0,\,V_{\text{rem}}(t))
\]
so
\[
p_t = \Phi\!\left(\frac{r(t)}{\sqrt{V_{\text{rem}}(t)}}\right)
\]
where \(\Phi\) is the standard normal CDF.
 
Everything reduces to estimating **remaining variance** \(V_{\text{rem}}(t)\) robustly online.
 
---
 
## 1) Inputs and timing conventions (critical for implementation)
### 1.1 Price stream used for state
You need a **reference price** stream for Bitcoin:
- A mid-price or microprice derived from Binance L1 (best bid/ask), or a consolidated mid from your feed.
- Denote the per-second sampled reference as \(m_k\) for second index \(k\).
 
### 1.2 Two clocks
You should distinguish:
- **Event time**: exchange timestamps (nice for research, not tradable).
- **Received time**: what your system actually knew at that moment (tradable).
 
For online pricing, everything should be computed off **received time**.
 
### 1.3 Update grid
**Recommended:** maintain volatility state on a **1-second grid**:
- Every second boundary, finalize that second’s return and update volatility states.
- You can still compute \(r(t)\) (the “distance from open”) at higher frequency (100ms etc.), but keep volatility on 1s for robustness.
 
Define:
- Hour open at received time \(t=0\)
- Second boundaries: \(t_k = k\cdot \Delta t\) with \(\Delta t = 1\text{ s}\)
- Remaining time: \(\tau(t)=T-t\)
 
---
 
## 2) State variables (what you store in memory)
At minimum, store:
 
### 2.1 “Distance from open”
- \(X_0 = \log(O)\)
- At current time \(t\), compute \(X(t)=\log(S(t))\) from latest price
- \(r(t)=X(t)-X_0\)
 
### 2.2 Volatility states (variance-per-second)
Maintain two online estimates of **variance rate** (variance per second):
- \(v_{\text{fast}}(k)\): reacts quickly to changes
- \(v_{\text{slow}}(k)\): stabilizes / acts as anchor
 
These are updated once per second from capped squared returns (below).
 
### 2.3 Time-of-day baseline (prior)
A precomputed baseline volatility rate:
- \(v_{\text{tod}}(h)\) where \(h\in\{0,\dots,23\}\) is hour-of-day
- Units: **variance per second** (consistent with \(v_{\text{fast}}, v_{\text{slow}}\))
 
This acts as a “prior” early in the hour or during quiet periods.
 
---
 
## 3) Computing returns (per-second)
At each second boundary \(k\ge 1\):
1. Sample the reference price \(m_k\) (mid/microprice) at that boundary.
2. Define log return:
\[
\Delta x_k = \log(m_k) - \log(m_{k-1})
\]
3. Raw squared increment: \(\Delta x_k^2\)
 
---
 
## 4) The key robustness feature: **jump/spike capping**
You observed that fast price moves can cause markets to “pull to 50%” by inflating implied remaining vol.
You want your volatility to react to **persistent** regime changes, not to every single “price level jump”.
 
### 4.1 Cap definition
Define a capped squared return:
\[
u_k = \min\left(\Delta x_k^2,\; c^2 \cdot v_{\text{slow}}(k-1)\cdot \Delta t\right)
\]
Where:
- \(c\) is the cap multiplier (typical range: 6–10)
- \(v_{\text{slow}}(k-1)\cdot \Delta t\) is the slow estimate of variance over one second
 
**Intuition:**
- If a one-second move is “too large” relative to your slow regime estimate, treat it mostly as a level shift and **do not** let it fully reprice diffusion variance immediately.
- The price level still affects \(r(t)\) strongly, which correctly moves the probability, but the *remaining variance* won’t explode on one spike.
 
### 4.2 Practical guardrails
To avoid bad behavior when initializing:
- Ensure \(v_{\text{slow}}(k-1)\) has a floor:
  \[
  v_{\text{slow}}(k-1)\leftarrow \max(v_{\text{slow}}(k-1), v_{\min})
  \]
- Choose \(v_{\min}\) small but nonzero (e.g. based on typical quiet-hour variance).
 
---
 
## 5) Updating fast/slow EWMA variance rates
### 5.1 EWMA form
Update once per second:
\[
v_{\text{fast}}(k) = (1-\lambda_f)\,v_{\text{fast}}(k-1) + \lambda_f \cdot \frac{u_k}{\Delta t}
\]
\[
v_{\text{slow}}(k) = (1-\lambda_s)\,v_{\text{slow}}(k-1) + \lambda_s \cdot \frac{u_k}{\Delta t}
\]
where \(\Delta t=1\text{ s}\) and \(u_k/\Delta t\) is the estimated variance rate for that second.
 
### 5.2 Choosing \(\lambda\) via half-life (recommended)
Pick half-lives:
- \(\text{HL}_f\): fast half-life in seconds (e.g. 30–90s)
- \(\text{HL}_s\): slow half-life in seconds (e.g. 600–1800s)
 
Convert to EWMA weight:
\[
\lambda = 1 - 2^{-\Delta t / \text{HL}}
\]
 
This is easier to reason about than picking \(\lambda\) directly.
 
### 5.3 Combine fast and slow into a single “current” estimate
\[
v_{\text{ewma}}(k) = \alpha\,v_{\text{fast}}(k) + (1-\alpha)\,v_{\text{slow}}(k)
\]
with \(\alpha\in[0,1]\) (common: 0.3–0.7).
 
---
 
## 6) Time-of-day prior (TOD) and online blending
### 6.1 Why you want TOD here
- Early in the hour you have little intrahour data, so volatility estimates are noisy.
- Crypto volatility varies by global sessions; TOD helps stop systematic bias.
 
### 6.2 Blending rule
Let \(h\) be the hour-of-day at the hour start. Use:
\[
v_{\text{blend}}(t) = w(t)\,v_{\text{ewma}}(k(t)) + (1-w(t))\,v_{\text{tod}}(h)
\]
where:
- \(k(t)=\lfloor t/\Delta t\rfloor\) is the last completed second index
- \(w(t)\in[0,1]\) increases as the hour progresses
 
A simple, effective ramp:
\[
w(t)=\min\left(1,\frac{t}{t_{\text{ramp}}}\right)
\]
with \(t_{\text{ramp}}\approx 10\text{ minutes}\).
 
**Interpretation:** for the first ~10 minutes you shrink toward the TOD baseline; after that you mostly trust your live estimate.
 
---
 
## 7) From variance rate to **remaining variance**
### 7.1 Core approximation
Assume the variance rate stays roughly constant over the remaining window:
\[
V_{\text{rem}}(t) = v_{\text{blend}}(t)\cdot \tau(t)
\]
where \(\tau(t)=T-t\) measured in seconds.
 
### 7.2 Add a microstructure / stability floor (important near expiry)
Very near expiry, your effective uncertainty isn’t zero because:
- reference price definition and discretization
- feed jitter / last-second prints
- microstructure effects
 
Add a small floor:
\[
V_{\text{rem}}(t)\leftarrow \max(V_{\text{rem}}(t), V_{\text{floor}}(\tau))
\]
 
Keep it simple:
- Constant floor: \(V_{\text{floor}} = \sigma_{\text{floor}}^2\)
- Or time-proportional: \(V_{\text{floor}}(\tau)=v_{\text{floor}}\cdot \tau\)
 
Choose floors from quiet-market diagnostics (don’t overthink).
 
---
 
## 8) Probability and fair price
### 8.1 Probability
Compute:
\[
z(t)=\frac{r(t)}{\sqrt{V_{\text{rem}}(t)}}
\quad\Rightarrow\quad
p_t=\Phi(z(t))
\]
 
### 8.2 Mapping to a “fair” contract price
For a binary paying \(1\) on Up and \(0\) otherwise, ignoring fees:
\[
\text{FairPrice}_t^{\text{Up}} \approx p_t
\quad\text{and}\quad
\text{FairPrice}_t^{\text{Down}} \approx 1-p_t
\]
 
If the venue has fees or non-1 payout conventions, you adjust this mapping separately. The pricer’s job is the probability.
 
---
 
## 9) Rolling algorithm: what to do at each step (implementation checklist)
Below is the full online loop a system would implement.
 
### 9.1 At the start of each hourly market
1. Record hour start time (received) and determine hour-of-day \(h\).
2. Set open price \(O\) from Binance reference used for settlement.
3. Set \(X_0=\log(O)\).
4. Initialize volatility states:
   - \(v_{\text{fast}} \leftarrow v_{\text{tod}}(h)\)
   - \(v_{\text{slow}} \leftarrow v_{\text{tod}}(h)\)
5. Initialize price sampling:
   - set \(m_0\) at the first second boundary (or immediately; be consistent).
 
### 9.2 Every second boundary (volatility update)
At time \(t_k\), for \(k=1,2,\dots\):
1. Sample \(m_k\) (mid/microprice at this second boundary).
2. Compute \(\Delta x_k=\log(m_k)-\log(m_{k-1})\).
3. Compute cap threshold:
   \[
   \text{capVar} = c^2\cdot v_{\text{slow}}(k-1)\cdot \Delta t
   \]
4. Compute capped increment:
   \[
   u_k = \min(\Delta x_k^2,\;\text{capVar})
   \]
5. EWMA updates:
   \[
   v_{\text{fast}}(k) \leftarrow (1-\lambda_f)v_{\text{fast}}(k-1) + \lambda_f\frac{u_k}{\Delta t}
   \]
   \[
   v_{\text{slow}}(k) \leftarrow (1-\lambda_s)v_{\text{slow}}(k-1) + \lambda_s\frac{u_k}{\Delta t}
   \]
6. Optionally compute \(v_{\text{ewma}}(k)=\alpha v_{\text{fast}}(k)+(1-\alpha)v_{\text{slow}}(k)\).
 
### 9.3 At any time you want to quote a probability (can be sub-second)
At current received time \(t\):
1. Pull latest tradeable price \(S(t)\) and compute:
   \[
   r(t)=\log(S(t)/O)
   \]
2. Determine \(\tau(t)=T-t\) (in seconds).
3. Compute \(w(t)=\min(1, t/t_{\text{ramp}})\).
4. Use latest completed volatility state \(v_{\text{ewma}}(k(t))\) and compute:
   \[
   v_{\text{blend}}(t)=w(t)v_{\text{ewma}}(k(t))+(1-w(t))v_{\text{tod}}(h)
   \]
5. Remaining variance:
   \[
   V_{\text{rem}}(t)=v_{\text{blend}}(t)\tau(t)
   \]
   Apply floor if desired.
6. Probability:
   \[
   p_t=\Phi\!\left(\frac{r(t)}{\sqrt{V_{\text{rem}}(t)}}\right)
   \]
7. Output \(p_t\) as your fair Up probability.
 
---
 
## 10) Estimating the time-of-day baseline \(v_{\text{tod}}(h)\)
This is an offline process you update periodically (daily/weekly).
 
### 10.1 Simple estimator (works well)
For each historical hour interval with hour-of-day \(h\):
1. Compute 1-second log returns \(\Delta x_k\) across that hour.
2. Compute realized variance for that hour:
   \[
   RV_{\text{hour}}=\sum_{k=1}^{3600}\Delta x_k^2
   \]
3. Convert to variance-per-second:
   \[
   v_{\text{hour}}=\frac{RV_{\text{hour}}}{3600}
   \]
Then set:
\[
v_{\text{tod}}(h)=\text{robust average of } v_{\text{hour}} \text{ over all hours with TOD }h
\]
Use a robust average:
- median, trimmed mean, or winsorized mean (recommended) to reduce jump contamination.
 
### 10.2 Smoothing across \(h\) (optional but nice)
Volatility across adjacent hours is correlated. You can smooth \(v_{\text{tod}}(h)\) by:
- a small moving average across \(h-1,h,h+1\), or
- exponential smoothing day-to-day.
 
Keep it stable; don’t chase noise.
 
---
 
# Calibration layer (simple and focused)
The pricer produces raw probabilities \(p_t^{\text{raw}}\). Calibration makes them match empirical frequencies better without changing the core mechanics.
 
## 11) What calibration is (and is not)
- **Calibration**: when you predict 0.70, the event should happen about 70% of the time.
- We calibrate against **market resolution outcomes** \(y\in\{0,1\}\) where \(y=1\) if Up resolved.
 
This section is **not** about trading/PnL optimization. It’s about probabilistic correctness.
 
---
 
## 12) Data you should collect for calibration
For each hourly market (each hour), choose a small set of snapshot times (fixed remaining times):
- Example: \(\tau\in\{50\text{m},30\text{m},15\text{m},5\text{m},2\text{m}\}\)
 
At each snapshot:
1. Record your model probability \(p^{\text{raw}}\).
2. Record the market-implied probability \(q\) (decide and document whether it’s mid, last, best bid/ask midpoint, etc.).
3. After resolution, record outcome \(y\in\{0,1\}\).
 
**Why fixed \(\tau\):**
- Observations within an hour are highly correlated (same eventual \(y\)).
- Fixed \(\tau\) makes comparisons cleaner and more stable.
 
---
 
## 13) Calibration evaluation: reliability buckets (your decile idea)
### 13.1 Bucket procedure
For a given snapshot time \(\tau\) (evaluate each \(\tau\) separately):
1. Collect all \((p_i^{\text{raw}},y_i)\) at that \(\tau\).
2. Sort by \(p_i^{\text{raw}}\).
3. Split into 10 equal-count buckets (deciles).
4. For each bucket \(b\):
   - predicted mean: \(\bar p_b = \frac{1}{n_b}\sum p_i^{\text{raw}}\)
   - empirical win rate: \(\hat y_b = \frac{1}{n_b}\sum y_i\)
 
Plot \(\hat y_b\) vs \(\bar p_b\). Perfect calibration lies on the 45° line.
 
Repeat the exact same procedure for market probabilities \(q_i\) to compare calibration quality.
 
### 13.2 Confidence intervals (so you don’t fool yourself)
Within bucket \(b\), approximate standard error:
\[
\mathrm{SE}_b \approx \sqrt{\frac{\hat y_b(1-\hat y_b)}{n_b}}
\]
Use this to add error bars (or Wilson intervals). This tells you whether differences are meaningful.
 
---
 
## 14) A very simple parametric calibration: Platt scaling (logit-linear)
### 14.1 Why this is a good minimal choice
- It’s only two parameters.
- It corrects systematic overconfidence/underconfidence.
- It usually improves log loss and reliability without overfitting badly.
 
### 14.2 Model
Define the logit function:
\[
\mathrm{logit}(p)=\log\frac{p}{1-p}
\]
Calibration map:
\[
\mathrm{logit}(p^{\text{cal}}) = a + b\,\mathrm{logit}(p^{\text{raw}})
\]
Then
\[
p^{\text{cal}} = \sigma\!\left(a + b\,\mathrm{logit}(p^{\text{raw}})\right)
\]
where \(\sigma(\cdot)\) is the logistic sigmoid.
 
### 14.3 Fitting procedure (offline, rolling)
For each snapshot time \(\tau\) (recommended: fit separate \(a_\tau,b_\tau\)):
1. Gather a training set of recent observations \((p_i^{\text{raw}}, y_i)\) at that \(\tau\).
2. Fit \(a,b\) by minimizing log loss:
\[
\min_{a,b}\sum_i \left[-y_i\log(p_i^{\text{cal}})-(1-y_i)\log(1-p_i^{\text{cal}})\right]
\]
3. Apply the fitted transform to future \(p^{\text{raw}}\) at the same \(\tau\).
 
**Implementation detail:** clip probabilities before logit:
- \(p^{\text{raw}}\leftarrow\min(\max(p^{\text{raw}},\epsilon),1-\epsilon)\) with \(\epsilon\sim10^{-6}\).
 
### 14.4 Avoiding overfitting in calibration
- Use a **rolling window** (e.g., last 30–90 days) and refit periodically.
- Keep it per-\(\tau\) or use a small number of \(\tau\)-bins (e.g., “early/mid/late”) if data is limited.
- If \(b\) tries to go extreme, constrain it (or add weak regularization).
 
---
 
## 15) Comparing your calibrated model to the market (still “probability-only”)
At each fixed \(\tau\), compare:
- your \(p^{\text{cal}}\) vs market \(q\)
- using proper scoring rules against realized \(y\)
 
### 15.1 Log loss (recommended)
\[
\ell(p,y) = -y\log p-(1-y)\log(1-p)
\]
Compute average log loss for your model and for the market at each \(\tau\).
 
### 15.2 Brier score (nice secondary metric)
\[
(p-y)^2
\]
 
### 15.3 Important practical note (dependence)
If you record many snapshots per hour, they are not independent. For clean stats:
- prefer a few fixed \(\tau\) values per hour, or
- aggregate performance by hour first, then average.
 
---
 
# Parameter defaults (reasonable starting point)
These are good “first run” values that avoid pathological overreaction:
 
- Sampling for volatility: \(\Delta t = 1s\)
- Fast half-life: \(\text{HL}_f = 60s\)
- Slow half-life: \(\text{HL}_s = 15m\)
- Combine: \(\alpha=0.5\)
- Cap multiplier: \(c=8\)
- TOD ramp: \(t_{\text{ramp}}=10m\)
- Floors: small \(v_{\min}\) and/or \(V_{\text{floor}}\) based on quiet periods
 
Tune later, but this gets you a stable, implementable baseline quickly.
 
---
 
# Summary: what your coder should implement
1. **Online per-second loop**:
   - compute \(\Delta x_k\), cap to \(u_k\), update \(v_{\text{fast}}, v_{\text{slow}}\), compute \(v_{\text{ewma}}\).
2. **On-demand pricing**:
   - compute \(r(t)\), blend vol with TOD, forecast \(V_{\text{rem}}(t)\), output \(p_t=\Phi(r/\sqrt{V_{\text{rem}}})\).
3. **Offline TOD estimation**:
   - compute \(v_{\text{tod}}(h)\) from historical 1s RV, robustly averaged.
4. **Calibration layer (simple)**:
   - reliability buckets + Platt scaling per fixed \(\tau\), evaluated on log loss/Brier vs market.
 
This spec is intentionally minimal and operational: it’s a base you can ship and stress-test.