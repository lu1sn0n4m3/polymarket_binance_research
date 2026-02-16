"""Abstract base class for binary option pricing models.

A Model maps (S, K, tau, features) -> P(Up).

To implement a new model:
    1. Subclass Model
    2. Set `name` class attribute
    3. Implement predict(), param_names(), initial_params(), param_bounds()
    4. Optionally override required_features() if you don't need all three defaults
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CalibrationResult:
    """Result of calibrating a model."""

    model_name: str
    params: dict[str, float]
    log_loss: float
    log_loss_baseline: float
    improvement_pct: float
    n_samples: int
    n_markets: int
    metadata: dict[str, Any] = field(default_factory=dict)


class Model(ABC):
    """Abstract base class for binary option pricing models.

    All pricers predict P(Up) = P(S_T > K) for hourly binary options.

    The predict() method takes params as an explicit argument (not self)
    so the optimizer can evaluate at arbitrary parameter vectors without
    mutating the model instance.
    """

    name: str = "unnamed"

    @abstractmethod
    def predict(
        self,
        params: dict[str, float],
        S: np.ndarray,
        K: np.ndarray,
        tau: np.ndarray,
        features: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict P(Up) for a batch of observations.

        Args:
            params: Parameter values (keys match param_names()).
            S: Current Binance mid prices.
            K: Strike prices (opening price of the hour).
            tau: Time to expiry in seconds.
            features: Dict of feature arrays (keys from required_features()).

        Returns:
            Array of probabilities in [1e-9, 1-1e-9], same length as S.
        """
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Ordered list of parameter names."""
        ...

    @abstractmethod
    def initial_params(self) -> dict[str, float]:
        """Initial parameter values for optimization."""
        ...

    @abstractmethod
    def param_bounds(self) -> dict[str, tuple[float, float]]:
        """(lower, upper) bounds for each parameter."""
        ...

    def predict_variance(
        self,
        params: dict[str, float],
        S: np.ndarray,
        K: np.ndarray,
        tau: np.ndarray,
        features: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predicted variance of log(S_T/S) over [t, T].

        Returns sigma_eff^2 * tau (or sigma_total^2 * tau for jump models).
        Used by variance-targeted calibration (QLIKE objective).
        """
        raise NotImplementedError(f"{self.name} does not implement predict_variance()")

    def required_features(self) -> list[str]:
        """Feature columns this model needs from the dataset.

        Default: ["sigma_tod", "sigma_rv", "sigma_rel"].
        Override to request fewer or different features.
        """
        return ["sigma_tod", "sigma_rv", "sigma_rel"]

    def set_params(self, params: dict[str, float]) -> None:
        """Store fitted parameters on the model instance."""
        self._fitted_params = params.copy()

    @property
    def fitted_params(self) -> dict[str, float] | None:
        return getattr(self, "_fitted_params", None)
