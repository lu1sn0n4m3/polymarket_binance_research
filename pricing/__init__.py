"""Lightweight pricing framework for binary option model research.

Workflow:
    1. Define a model (subclass Model in pricing/models/)
    2. Build a calibration dataset from 1s Binance data
    3. Calibrate via MLE (minimize log loss)
    4. Generate diagnostic plots
    5. Visualize on single markets

Example:
    from datetime import date
    from pathlib import Path
    from pricing.dataset import build_dataset, DatasetConfig
    from pricing.calibrate import calibrate
    from pricing.diagnostics import generate_diagnostics
    from pricing.models.simple_gaussian import SimpleGaussianModel

    dataset = build_dataset(DatasetConfig(
        start_date=date(2026, 1, 19),
        end_date=date(2026, 1, 30),
    ))
    model = SimpleGaussianModel()
    result = calibrate(model, dataset)
    generate_diagnostics(model, dataset, result, Path("pricing/output"))
"""
