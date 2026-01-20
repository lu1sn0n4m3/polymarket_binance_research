"""Tests for alignment utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import (
    align_asof,
    align_bucketed,
    resample_to_grid,
    compute_derived_fields,
)


def test_align_asof_backward():
    """Test ASOF join with backward direction."""
    left = pd.DataFrame({
        "ts_recv": [100, 200, 300, 400],
        "left_val": [1, 2, 3, 4],
    })
    right = pd.DataFrame({
        "ts_recv": [50, 150, 350],
        "right_val": ["a", "b", "c"],
    })
    
    result = align_asof(left, right, direction="backward")
    
    assert len(result) == 4
    # At ts=100, latest right is ts=50 ("a")
    assert result.loc[0, "right_val"] == "a"
    # At ts=200, latest right is ts=150 ("b")
    assert result.loc[1, "right_val"] == "b"
    # At ts=300, latest right is still ts=150 ("b")
    assert result.loc[2, "right_val"] == "b"
    # At ts=400, latest right is ts=350 ("c")
    assert result.loc[3, "right_val"] == "c"


def test_align_bucketed():
    """Test bucketed alignment."""
    left = pd.DataFrame({
        "ts_recv": [1000, 1100, 2000, 2500],
        "left_val": [1, 2, 3, 4],
    })
    right = pd.DataFrame({
        "ts_recv": [1050, 1900, 2100],
        "right_val": [10, 20, 30],
    })
    
    result = align_bucketed(left, right, bucket_ms=1000, agg_method="last")
    
    # Should have 2 buckets: 1000 and 2000
    assert len(result) == 2


def test_resample_to_grid():
    """Test resampling to regular grid."""
    df = pd.DataFrame({
        "ts_recv": [100, 250, 400],
        "value": [1.0, 2.0, 3.0],
    })
    
    result = resample_to_grid(df, grid_ms=100, method="ffill")
    
    # Grid should cover 100, 200, 300, 400, 500
    assert len(result) == 5
    # Check forward fill
    assert result[result["ts_recv"] == 200]["value"].iloc[0] == 1.0
    assert result[result["ts_recv"] == 300]["value"].iloc[0] == 2.0


def test_compute_derived_fields():
    """Test derived field computation."""
    df = pd.DataFrame({
        "bid_px": [0.45, 0.46],
        "ask_px": [0.47, 0.48],
        "bid_sz": [100, 200],
        "ask_sz": [150, 100],
    })
    
    result = compute_derived_fields(df, prefix="")
    
    assert "mid" in result.columns
    assert "spread" in result.columns
    assert "microprice" in result.columns
    
    # Check mid
    assert np.isclose(result.loc[0, "mid"], 0.46)
    # Check spread
    assert np.isclose(result.loc[0, "spread"], 0.02)
