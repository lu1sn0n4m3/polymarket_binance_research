# Estimator + Pricer for BTC-Linked Hourly Digitals

This repo implements a calibrated binary option pricer for hourly BTC digital options on Polymarket, using Binance BBO tick data.

See `estimation.tex` for the mathematical framework and `final_model.tex` for the final model specification.

## What it does

- Ingests Binance best bid/ask data from S3, builds a 100ms mid-price grid.
- Estimates intraday seasonal volatility sigma_tod via tick-time MAD (5-min buckets).
- Computes realized volatility sigma_rv via EWMA of sum(dx^2)/sum(dt) with 5-min half-life.
- Prices hourly digitals P(S_T > K) using a two-stage model:
  - **Stage 1**: sigma_eff = a(tau) * sigma_tod^(1-beta) * sigma_rv^beta
  - **Stage 2**: Probit calibration layer correcting for tau and vol-regime effects
- Outputs fair probability p and sensitivities (dp/dS, dp/dlnS).

## Quickstart

```bash
# Configure data range in pricer_calibration/config.yaml, then:
python -m pricer_calibration.run.run_calibration --rebuild
```

See `pricer_calibration/README.md` for full documentation.

## Key files

- `estimation.tex` -- mathematical whitepaper
- `final_model.tex` -- final model specification
- `task.md` -- engineering plan
- `pricer_calibration/` -- implementation (see its README)
