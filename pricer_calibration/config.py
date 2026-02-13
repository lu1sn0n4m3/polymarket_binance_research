"""Configuration loader for pricer calibration pipeline."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    """All pipeline parameters."""

    # Data range
    start_date: date
    end_date: date
    start_hour: int = 0
    end_hour: int = 23
    asset: str = "BTC"

    # Grid interval (matches src.data.easy_api)
    interval: str = "500ms"

    # TOD seasonal vol
    tod_bucket_minutes: int = 5
    tod_smoothing_window: int = 3
    sigma_tod_floor: float = 1e-10

    # EWMA
    ewma_half_life_seconds: float = 20.0
    ewma_u_sq_cap: float = 100.0

    # Shock
    shock_M: int = 5
    shock_c: float = 3.5

    # Calibration
    gamma_init: float = 0.0
    lambda_l2: float = 0.01
    sample_interval_sec: float = 5.0
    fix_c: bool = True

    # Output
    output_dir: str = "pricer_calibration/output"

    _INTERVAL_MS = {"500ms": 500, "1s": 1000, "5s": 5000}

    @property
    def delta_ms(self) -> int:
        """Grid step in milliseconds, derived from interval string."""
        return self._INTERVAL_MS[self.interval]

    @property
    def delta_sec(self) -> float:
        return self.delta_ms / 1000.0

    @property
    def n_tod_buckets(self) -> int:
        return 24 * 60 // self.tod_bucket_minutes


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load config from YAML file."""
    if path is None:
        path = Path(__file__).parent / "config.yaml"
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    # Parse dates
    start = raw.pop("start_date")
    end = raw.pop("end_date")
    if isinstance(start, str):
        start = date.fromisoformat(start)
    if isinstance(end, str):
        end = date.fromisoformat(end)

    start_hour = raw.pop("start_hour", 0)
    end_hour = raw.pop("end_hour", 23)

    # Migrate old delta_ms config to interval string
    if "delta_ms" in raw:
        raw.pop("delta_ms")
        if "interval" not in raw:
            raw["interval"] = "500ms"

    return PipelineConfig(start_date=start, end_date=end, start_hour=start_hour, end_hour=end_hour, **raw)
