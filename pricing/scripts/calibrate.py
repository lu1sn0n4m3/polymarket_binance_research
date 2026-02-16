"""One-click model calibration.

Usage:
    python -m pricing.scripts.calibrate --model gaussian
    python -m pricing.scripts.calibrate --model simple_gaussian --start 2026-01-19 --end 2026-01-30
    python -m pricing.scripts.calibrate --model gaussian --rebuild
"""

import argparse
import importlib
from datetime import date
from pathlib import Path

import pandas as pd


# Registry of built-in models (name -> module path, class name)
MODEL_REGISTRY = {
    "simple_gaussian": ("pricing.models.simple_gaussian", "SimpleGaussianModel"),
    "gaussian": ("pricing.models.gaussian", "GaussianModel"),
}


def get_model(name: str):
    """Load a model by name from the registry or by module path."""
    if name in MODEL_REGISTRY:
        module_path, class_name = MODEL_REGISTRY[name]
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()

    # Try loading as "module.ClassName"
    if "." in name:
        parts = name.rsplit(".", 1)
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])()

    raise ValueError(
        f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}\n"
        f"Or use 'module.path.ClassName' for custom models."
    )


def main():
    parser = argparse.ArgumentParser(description="Calibrate a pricing model")
    parser.add_argument("--model", type=str, required=True,
                        help=f"Model name ({', '.join(MODEL_REGISTRY)}) or module.ClassName")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Default: earliest cached date")
    parser.add_argument("--end", type=str, default=None,
                        help="End date (YYYY-MM-DD). Default: latest cached date")
    parser.add_argument("--asset", type=str, default="BTC",
                        help="Asset (BTC or ETH)")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild dataset from scratch")
    parser.add_argument("--output-dir", type=str, default="pricing/output",
                        help="Output directory")
    parser.add_argument("--sample-interval", type=float, default=60.0,
                        help="Sampling interval within each hour (seconds)")
    parser.add_argument("--l2", type=float, default=0.0,
                        help="L2 regularization strength")
    args = parser.parse_args()

    from pricing.dataset import build_dataset, DatasetConfig
    from pricing.calibrate import calibrate
    from pricing.diagnostics import generate_diagnostics

    print("=" * 60)
    print("BINARY OPTION PRICER CALIBRATION")
    print("=" * 60)

    # Load model
    model = get_model(args.model)
    print(f"\nModel: {model.name}")
    print(f"Parameters: {model.param_names()}")
    print(f"Features: {model.required_features()}")

    # Build or load dataset
    output_dir = Path(args.output_dir)
    dataset_path = output_dir / "calibration_dataset.parquet"

    if dataset_path.exists() and not args.rebuild:
        print(f"\nLoading cached dataset from {dataset_path}")
        dataset = pd.read_parquet(dataset_path)
        print(f"  {len(dataset):,} calibration samples")
    else:
        # Auto-detect date range from cache if not specified
        start = args.start
        end = args.end
        if start is None or end is None:
            from src.data import get_cache_status
            status = get_cache_status("binance", args.asset, "1s")
            available = status.get("available_dates", [])
            if not available:
                print("No cached 1s data found. Specify --start and --end manually.")
                return
            if start is None:
                start = available[0]
            if end is None:
                end = available[-1]
            print(f"\nAuto-detected date range: {start} to {end} ({len(available)} days cached)")

        cfg = DatasetConfig(
            start_date=date.fromisoformat(start),
            end_date=date.fromisoformat(end),
            asset=args.asset,
            sample_interval_sec=args.sample_interval,
            output_dir=args.output_dir,
        )
        dataset = build_dataset(cfg)

    if dataset.empty:
        print("No data to calibrate. Exiting.")
        return

    # Calibrate
    result = calibrate(model, dataset, l2_lambda=args.l2, output_dir=args.output_dir)

    # Generate diagnostics
    print("\nGenerating diagnostic plots ...")
    generate_diagnostics(model, dataset, result, output_dir)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    print(f"\nOutputs in {args.output_dir}/:")
    print(f"  - {model.name}_params.json")
    print(f"  - {model.name}_diagnostics.png")
    print(f"  - {model.name}_conditional.png")


if __name__ == "__main__":
    main()
