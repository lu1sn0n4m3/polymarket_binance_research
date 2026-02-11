"""Configuration for S3 and data access."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


@dataclass
class S3Config:
    """S3 connection configuration."""
    
    bucket: str = "marketdata-archive"
    prefix: str = "prod"
    endpoint: str = "nbg1.your-objectstorage.com"
    region: str = "eu-central"
    access_key: str = field(default_factory=lambda: os.environ.get("S3_ACCESS_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get("S3_SECRET_KEY", ""))
    url_style: str = "path"
    
    def validate(self) -> None:
        """Raise if credentials are missing."""
        if not self.access_key or not self.secret_key:
            raise ValueError(
                "S3 credentials not set. Set S3_ACCESS_KEY and S3_SECRET_KEY "
                "environment variables or pass them to S3Config."
            )


@dataclass
class DataConfig:
    """Data loading configuration."""
    
    s3: S3Config = field(default_factory=S3Config)
    
    # Polymarket stream IDs
    btc_stream_id: str = "bitcoin-up-or-down"
    eth_stream_id: str = "ethereum-up-or-down"
    
    # Binance stream IDs
    btc_binance_stream: str = "BTCUSDT"
    eth_binance_stream: str = "ETHUSDT"
    
    # Default lookback for volatility estimation (hours)
    default_lookback_hours: int = 3

    # Cache configuration for resampled BBO data
    cache_dir: Path = field(default_factory=lambda: Path("data/resampled_bbo"))
    cache_enabled: bool = True
    cache_max_size_gb: float = 10.0

    def stream_id_for_asset(self, asset: str, venue: str) -> str:
        """Get stream_id for an asset and venue."""
        asset = asset.upper()
        venue = venue.lower()
        
        if venue == "polymarket":
            return self.btc_stream_id if asset == "BTC" else self.eth_stream_id
        elif venue == "binance":
            return self.btc_binance_stream if asset == "BTC" else self.eth_binance_stream
        else:
            raise ValueError(f"Unknown venue: {venue}")


# Global default config (can be overridden)
_config: DataConfig | None = None


def get_config() -> DataConfig:
    """Get the global configuration."""
    global _config
    if _config is None:
        _config = DataConfig()
    return _config


def set_config(config: DataConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config
