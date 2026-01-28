"""Tests for 100ms grid construction."""

import numpy as np
import pandas as pd
import pytest

from pricer_calibration.data.grid import build_grid


def _make_bbo(ts_events, mids):
    return pd.DataFrame({"ts_event": ts_events, "mid": mids})


class TestBuildGrid:
    def test_constant_mid(self):
        """Constant mid should produce zero returns after first row."""
        ts = np.arange(0, 10000, 50, dtype=np.int64)  # 50ms updates
        bbo = _make_bbo(ts, np.full(len(ts), 100.0))
        grid = build_grid(bbo, delta_ms=100)

        assert len(grid) > 0
        assert (grid["S"] == 100.0).all()
        assert np.allclose(grid["r"].values[1:], 0.0)

    def test_grid_spacing_constant(self):
        """Grid should have exactly delta_ms spacing."""
        ts = np.arange(0, 5000, 30, dtype=np.int64)
        bbo = _make_bbo(ts, np.random.uniform(99, 101, len(ts)))
        grid = build_grid(bbo, delta_ms=100)

        diffs = np.diff(grid["t"].values)
        assert (diffs == 100).all()

    def test_no_nans_after_first_obs(self):
        """No NaNs should remain in the grid."""
        ts = np.arange(100, 3000, 40, dtype=np.int64)
        bbo = _make_bbo(ts, np.random.uniform(99, 101, len(ts)))
        grid = build_grid(bbo, delta_ms=100)

        assert grid["S"].isna().sum() == 0
        assert grid["logS"].isna().sum() == 0
        assert np.isfinite(grid["r"].values).all()

    def test_previous_tick_sampling(self):
        """Grid should use the latest observation at or before grid time."""
        bbo = _make_bbo(
            [50, 150, 250],
            [100.0, 200.0, 300.0],
        )
        grid = build_grid(bbo, delta_ms=100)

        # t=100 should use the observation at t=50 (mid=100)
        # t=200 should use the observation at t=150 (mid=200)
        assert grid.iloc[0]["S"] == 100.0
        assert grid.iloc[1]["S"] == 200.0

    def test_empty_input(self):
        """Empty BBO should return empty grid."""
        bbo = pd.DataFrame(columns=["ts_event", "mid"])
        grid = build_grid(bbo, delta_ms=100)
        assert len(grid) == 0

    def test_returns_are_log_diffs(self):
        """Returns should be log differences of S."""
        bbo = _make_bbo(
            [0, 50, 150, 250],
            [100.0, 100.0, 110.0, 105.0],
        )
        grid = build_grid(bbo, delta_ms=100)

        # t=0: S=100, t=100: S=100 (from ts=50), t=200: S=110 (from ts=150)
        expected_r2 = np.log(110.0) - np.log(100.0)
        assert abs(grid.iloc[2]["r"] - expected_r2) < 1e-10
