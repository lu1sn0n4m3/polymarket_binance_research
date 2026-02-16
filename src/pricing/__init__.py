"""Legacy pricing models -- archived.

For new model development, use the top-level pricing/ framework:
    from pricing.models.base import Model
    from pricing.calibrate import calibrate
    from pricing.diagnostics import generate_diagnostics

See pricing/models/simple_gaussian.py for a minimal example.
"""

import warnings

warnings.warn(
    "src.pricing is deprecated. Use the top-level pricing/ framework instead. "
    "Old models moved to src/pricing/archive/.",
    DeprecationWarning,
    stacklevel=2,
)

from src.pricing.base import Pricer, PricerOutput, NaivePricer, MoneynessPricer
