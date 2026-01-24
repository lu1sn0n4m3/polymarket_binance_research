# Gaussian EWMA Pricer: Design, Analysis & Comparison with Polymarket

This document explains the Gaussian EWMA Pricer, how it prices hourly binary options on Bitcoin, and provides a detailed comparison with Polymarket's crowd-sourced probabilities.

---

## Table of Contents

1. [The Problem: Pricing Hourly Binary Options](#the-problem)
2. [The Gaussian EWMA Pricer](#the-pricer)
3. [Calibration Analysis Framework](#calibration-analysis)
4. [Results: Model vs Polymarket](#results)
5. [Where Polymarket Beats the Model](#pm-wins)
6. [Where the Model Beats Polymarket](#model-wins)
7. [Key Insights & Actionable Takeaways](#insights)
8. [Configuration & Tuning Guide](#tuning)

---

<a name="the-problem"></a>
## 1. The Problem: Pricing Hourly Binary Options

### What We're Pricing

Polymarket offers hourly binary options on Bitcoin with the question: **"Will BTC be UP or DOWN at the end of this hour compared to the start?"**

- **Strike price**: The first trade price at the start of the hour (the "open")
- **Expiry**: End of the hour (60 minutes)
- **Payout**: 100% if correct, 0% if wrong
- **Current price**: Polymarket shows a probability (e.g., 65% chance of UP)

### The Challenge

To profitably trade these markets, we need to:
1. Estimate the **true probability** that BTC will be above the open at expiry
2. Compare our estimate to Polymarket's price
3. Trade when there's a meaningful **edge** (our price ≠ market price)

The core challenge is **volatility estimation** — how much will BTC move in the remaining time?

---

<a name="the-pricer"></a>
## 2. The Gaussian EWMA Pricer

### Core Idea

The pricer assumes BTC log-returns follow a **Gaussian distribution** (or optionally Student-t for fatter tails). Given:

- **Current price** relative to open: `r = ln(price / open_price)`
- **Remaining variance**: `V_rem = σ² × τ` (variance rate × time remaining)

The probability of finishing above the open is:

```
P(UP) = Φ(r / √V_rem)
```

Where `Φ` is the standard normal CDF.

### Why EWMA?

The key question is: **what variance rate σ² should we use?**

Simple approaches fail:
- Historical volatility: Too slow to react to changing conditions
- Recent volatility: Too noisy, overreacts to single moves

**EWMA (Exponentially Weighted Moving Average)** provides a balance:
- Recent observations have more weight
- But extreme moves don't dominate

### Dual EWMA with Capping

The pricer uses two EWMA estimates:

| EWMA | Half-life | Purpose |
|------|-----------|---------|
| **Fast** | 5 min | Reacts quickly to regime changes |
| **Slow** | 15-30 min | Anchor, prevents overreaction |

These are blended: `v_ewma = α × v_fast + (1-α) × v_slow`

**Capping**: Large price jumps are capped to prevent a single move from spiking volatility:
```
r²_capped = min(r², c² × v_slow × Δt)
```

Where `c = 8` (cap multiplier) means moves >8σ are truncated.

### Time-of-Day (TOD) Prior

Volatility varies by hour of day (e.g., higher during US market hours). The pricer uses a **TOD prior** estimated from the same hour on previous days.

Early in the hour (first ~10 minutes), the TOD prior dominates because we don't have enough in-market data. As time passes, we ramp to trust the live EWMA estimate more.

```
v_blend = w × v_tod + (1-w) × v_ewma
```

Where `w` starts at 1 and decays to 0 over the ramp period.

### Recent Fixes

Two additional adjustments address systematic biases:

#### Fix 1: Volatility Floor
Prevents overconfidence in calm markets by setting a minimum volatility:
```
v_blend = max(v_blend, vol_floor)
```

#### Fix 2: ITM Volatility Boost
When price has moved above the open (in-the-money for UP), adds extra volatility to account for reversal risk:
```
if r > 0:
    V_rem *= (1 + boost × moneyness_factor × time_factor)
```

This reduces overconfidence when predicting "almost certain" outcomes.

---

<a name="calibration-analysis"></a>
## 3. Calibration Analysis Framework

### What is Calibration?

A model is **well-calibrated** if when it says "70% probability", the event happens 70% of the time.

We measure this by:
1. Bucketing all predictions (0-10%, 10-20%, ..., 90-100%)
2. Computing the actual hit rate in each bucket
3. Comparing to the bucket midpoint

**Calibration error** = Actual hit rate - Predicted probability

### Key Metrics

| Metric | What It Measures | Better |
|--------|------------------|--------|
| **Brier Score** | Overall accuracy (MSE of probabilities) | Lower |
| **Reliability** | Calibration error | Lower |
| **Resolution** | Ability to separate UP from DOWN | Higher |
| **Sharpness** | Confidence (distance from 50%) | Depends |
| **AUC** | Discrimination ability | Higher |

**Brier Score = Reliability - Resolution + Uncertainty**

You want: Low reliability (well-calibrated) + High resolution (good separation)

### Win Rate Analysis

Beyond aggregate metrics, we track **per-sample win rate**: for each prediction, did the model or Polymarket have lower squared error?

This reveals if the model wins more often even if aggregate Brier is worse (happens when model loses big on a few samples).

---

<a name="results"></a>
## 4. Results: Model vs Polymarket

### Overall Summary (872 samples, 73 market hours)

| Metric | Polymarket | Gaussian Model | Student-t Model |
|--------|------------|----------------|-----------------|
| **Brier Score** | **0.162** | 0.170 | 0.170 |
| **Accuracy** | 73.7% | **74.5%** | **74.5%** |
| **AUC** | **0.845** | 0.825 | 0.825 |
| **Sharpness** | 0.222 | **0.237** | **0.246** |
| **Resolution** | **0.088** | 0.080 | 0.079 |

**Key finding**: Polymarket has better Brier score (4.9% better) and resolution, but the model has higher accuracy and sharpness.

### Calibration by Probability Bucket

```
Bucket      | PM Actual | PM Error | Model Actual | Model Error
------------|-----------|----------|--------------|------------
0-10%       |   0.0%    |  -5.0pp  |     1.5%     |   -3.5pp ✓
10-20%      |   2.5%    | -12.5pp  |    12.3%     |   -2.7pp ✓
20-30%      |  17.4%    |  -7.6pp  |    20.4%     |   -4.6pp ✓
30-40%      |  29.1%    |  -5.9pp  |    36.1%     |   +1.1pp ✓
40-50%      |  37.4%    |  -7.6pp  |    34.1%     |  -10.9pp
50-60%      |  46.0%    |  -9.0pp  |    58.8%     |   +3.8pp ✓
60-70%      |  66.7%    |  +1.7pp  |    61.3%     |   -3.7pp
70-80%      |  75.0%    |  ±0.0pp ✓|    57.6%     |  -17.4pp ✗
80-90%      |  87.1%    |  +2.1pp  |    86.5%     |   +1.5pp ✓
90-100%     |  94.6%    |  -0.4pp ✓|    92.4%     |   -2.6pp
```

---

<a name="pm-wins"></a>
## 5. Where Polymarket Beats the Model

### 5.1 High-Probability Predictions (70-80% bucket)

**The Problem**: When the model predicts 75% chance of UP, it only happens 57.6% of the time. That's a massive -17.4pp error.

**Why This Happens**:
- Price has moved significantly above open (model is bullish)
- Model sees low recent volatility (price is stable after the move)
- Model becomes overconfident that UP will persist
- But there's **reversal risk** the model underestimates

**Polymarket's Advantage**: Human traders have intuition about reversals. They've seen rallies fade. They don't just extrapolate recent calm periods.

### 5.2 Medium Volatility Regime

| Vol Regime | PM Brier | Model Brier | Gap |
|------------|----------|-------------|-----|
| Low (<3.6%) | 0.170 | 0.177 | PM +4.7% |
| **Medium (3.6-4.7%)** | **0.191** | **0.216** | **PM +12.9%** |
| High (>4.7%) | 0.127 | 0.120 | Model +6.0% |

The model struggles most in medium volatility — not calm enough for the floor to help, not volatile enough to show the model's strength.

### 5.3 Near-ATM Positions

When price is close to the open (|return| < 0.1%), outcomes are essentially coin flips. Polymarket handles this better:

| Moneyness | PM Brier | Model Brier |
|-----------|----------|-------------|
| ATM (<0.1%) | **0.216** | 0.222 |
| Near (0.1-0.3%) | **0.137** | 0.153 |
| Deep (>0.3%) | 0.035 | **0.031** |

### 5.4 Resolution (Separating UP from DOWN)

Polymarket has higher resolution (0.088 vs 0.080), meaning it's better at assigning high probabilities to actual UPs and low probabilities to actual DOWNs.

This suggests Polymarket participants have **informational advantages**:
- Order flow signals
- Sentiment from other markets
- Whale activity detection
- News/event awareness

---

<a name="model-wins"></a>
## 6. Where the Model Beats Polymarket

### 6.1 Low-Probability Predictions (0-30%)

When predicting **bearish outcomes** (UP unlikely), the model is much better calibrated:

| Bucket | PM Error | Model Error | Winner |
|--------|----------|-------------|--------|
| 0-10% | -5.0pp | -3.5pp | **Model** |
| 10-20% | -12.5pp | -2.7pp | **Model** |
| 20-30% | -7.6pp | -4.6pp | **Model** |

**Insight**: Polymarket traders are **systematically too bullish** on low-probability UP moves. When PM says "15% chance of UP", it only happens 2.5% of the time!

The model's volatility-based approach doesn't have this bullish bias.

### 6.2 High Volatility Regime

When lookback volatility exceeds 4.7% (annualized), the model significantly outperforms:

| Metric | PM | Model | Advantage |
|--------|-----|-------|-----------|
| Brier | 0.127 | **0.120** | Model +6.0% |
| Accuracy | 83.0% | **88.0%** | Model +5pp |
| Win Rate | - | **65.0%** | Model wins 65% of samples |

**Why**: In high-vol environments, the model's volatility estimation shines. It correctly predicts larger moves and adjusts probabilities appropriately. Polymarket traders may underreact to volatility spikes.

### 6.3 Deep In-The-Money Positions

When price has moved significantly from open (|return| > 0.3%), the model excels:

| Moneyness | PM Brier | Model Brier | Winner |
|-----------|----------|-------------|--------|
| Deep (>0.3%) | 0.035 | **0.031** | **Model** |

The model correctly assigns near-certain probabilities (>90%) to positions with large leads.

### 6.4 Per-Sample Win Rate

Despite worse aggregate Brier, the model wins **61.4% of individual samples**:

```
Model wins: 535 (61.4%)
PM wins:    337 (38.6%)
```

This means when you disagree with PM, you're right more often than wrong. The aggregate Brier is worse because when the model loses, it loses bigger (overconfidence in the 70-80% bucket).

### 6.5 Bearish Edge is Profitable

When the model is **more bearish** than Polymarket (predicts lower P(UP)):
- 196 samples with edge < -5pp
- **71.9% accuracy** on these samples

When the model is **more bullish** than Polymarket:
- 107 samples with edge > +5pp  
- Only **45.8% accuracy**

**Actionable**: Trust the model's bearish calls, be skeptical of its bullish calls.

---

<a name="insights"></a>
## 7. Key Insights & Actionable Takeaways

### Insight 1: Polymarket Has a Bullish Bias

PM consistently overprices low-probability UP moves. When PM says 15%, reality is closer to 5%. This creates a **systematic short opportunity** on UP tokens when PM is bearish.

### Insight 2: The Model Overestimates Certainty After Rallies

When price rallies and the model becomes very bullish (70-80%), it's too confident. Reversal risk is higher than the model estimates. 

**Fix**: The ITM volatility boost helps but needs tuning. Consider also blending toward PM when model confidence is extreme.

### Insight 3: Volatility Regime Matters

| Regime | Strategy |
|--------|----------|
| **Low Vol** | Trust PM more, or skip trading |
| **Medium Vol** | Careful, model's weakest regime |
| **High Vol** | Model has edge, trade confidently |

### Insight 4: Moneyness × Volatility Interaction

The model's edge depends on BOTH where the price is AND volatility:

```
                    | Low Vol | Med Vol | High Vol
-------------------------------------------------
ATM (near open)     |    PM   |    PM   |  Model
Near (small move)   |  Model  |    PM   |  Model  
Deep (large move)   |  Model  |    PM   |  Model
```

**Best regime for model**: High volatility, any moneyness.

### Insight 5: Use Bearish Signals

The model's bearish predictions are more reliable than bullish ones:
- **Bearish edge**: 71.9% accurate
- **Bullish edge**: 45.8% accurate

Consider asymmetric sizing: larger positions on bearish signals.

---

<a name="tuning"></a>
## 8. Configuration & Tuning Guide

### Current Configuration

```python
PRICER_CONFIG = {
    # EWMA parameters
    "fast_halflife_sec": 300.0,    # 5 min fast EWMA
    "slow_halflife_sec": 1800.0,   # 30 min slow EWMA
    "alpha": 0.3,                   # 30% fast, 70% slow
    
    # Jump capping
    "cap_multiplier": 8.0,         # Cap at 8σ
    "enable_capping": True,
    
    # TOD prior
    "tod_ramp_sec": 60.0,          # Ramp to live EWMA over 1 min
    "use_tod_prior": True,
    "tod_blend_beta": 0.0,         # No permanent TOD blend
    
    # Fixes
    "vol_floor_annual": 0.036,     # Min 3.6% annualized vol
    "itm_vol_boost": 0.3,          # +30% vol when ITM
    "itm_vol_time_scaling": True,  # Scale boost by time
    
    # Distribution
    "student_t_nu": 8.0,           # Student-t degrees of freedom
}
```

### Tuning Recommendations

| To Address | Adjust | Try Values |
|------------|--------|------------|
| 70-80% overconfidence | `itm_vol_boost` | 0.15-0.25 (lower) |
| Low vol underperformance | `vol_floor_annual` | 0.04-0.05 (higher) |
| Medium vol issues | `slow_halflife_sec` | 2400-3600 (longer) |
| More conservative overall | `vol_multiplier` | 1.1-1.3 |
| Fatter tails | `student_t_nu` | 4-6 (lower) |

### Parameter Sweep Template

To systematically find optimal parameters:

```python
PARAM_GRID = {
    "vol_floor_annual": [0.0, 0.03, 0.036, 0.04, 0.05],
    "itm_vol_boost": [0.0, 0.15, 0.3, 0.5],
    "slow_halflife_sec": [900, 1800, 2700, 3600],
}
```

---

## Conclusion

The Gaussian EWMA Pricer provides a **fundamentals-based alternative** to Polymarket's crowd-sourced probabilities. While PM has better overall Brier score, the model has distinct advantages:

1. **Better calibrated for bearish predictions** (0-30% bucket)
2. **Significantly outperforms in high volatility** (+6% Brier advantage)
3. **Wins 61% of individual samples** (valuable for per-trade decisions)
4. **No bullish bias** unlike Polymarket

The model's main weakness is **overconfidence after rallies** (70-80% bucket), which can be addressed through the ITM volatility boost and further tuning.

For practical trading:
- **Trust bearish signals** from the model
- **Be skeptical of bullish signals** especially in moderate vol
- **Trade aggressively in high volatility** where the model shines
- **Consider blending with PM** in low/medium volatility regimes
