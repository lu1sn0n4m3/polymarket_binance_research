"""Visualize model predictions on a single hourly market.

Usage:
    python -m pricing.scripts.view_market --model gaussian --date 2026-01-19 --hour 14
    python -m pricing.scripts.view_market --model simple_gaussian --date 2026-01-20 --hour 9 --save
"""

import argparse
import json
from datetime import date
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Visualize model on a single market")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name or module.ClassName")
    parser.add_argument("--date", type=str, required=True,
                        help="Market date (YYYY-MM-DD)")
    parser.add_argument("--hour", type=int, required=True,
                        help="UTC hour (0-23)")
    parser.add_argument("--asset", type=str, default="BTC",
                        help="Asset (BTC or ETH)")
    parser.add_argument("--params-file", type=str, default=None,
                        help="Path to params JSON (default: pricing/output/{model}_params.json)")
    parser.add_argument("--output-dir", type=str, default="pricing/output",
                        help="Output directory for params and plots")
    parser.add_argument("--save", action="store_true",
                        help="Save plot to file instead of showing")
    parser.add_argument("--ewma-half-life", type=float, default=300.0,
                        help="EWMA half-life for sigma_rv (seconds)")
    args = parser.parse_args()

    from pricing.scripts.calibrate import get_model
    from pricing.market_view import plot_market

    # Load model
    model = get_model(args.model)

    # Load params
    params_file = args.params_file or f"{args.output_dir}/{model.name}_params.json"
    params_path = Path(params_file)
    if not params_path.exists():
        print(f"Params file not found: {params_path}")
        print(f"Run calibration first: python -m pricing.scripts.calibrate --model {args.model}")
        return

    with open(params_path) as f:
        params_data = json.load(f)

    # Extract just the model parameters
    params = {k: params_data[k] for k in model.param_names() if k in params_data}

    market_date = date.fromisoformat(args.date)
    output_path = None
    if args.save:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{model.name}_{args.date}_{args.hour:02d}.png"

    plot_market(
        model=model,
        params=params,
        asset=args.asset,
        market_date=market_date,
        hour_utc=args.hour,
        ewma_half_life_sec=args.ewma_half_life,
        output_path=output_path,
        show=not args.save,
    )


if __name__ == "__main__":
    main()
