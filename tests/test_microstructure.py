"""Tests for microstructure features."""

import numpy as np
import pandas as pd
import pytest

from src.features.microstructure import (
    compute_microprice,
    compute_book_imbalance,
    compute_spread_bps,
    compute_vwap,
)


def test_compute_microprice():
    """Test microprice computation."""
    bid_px = pd.Series([0.45, 0.46])
    ask_px = pd.Series([0.47, 0.48])
    bid_sz = pd.Series([100, 200])
    ask_sz = pd.Series([150, 100])
    
    microprice = compute_microprice(bid_px, ask_px, bid_sz, ask_sz)
    
    # Microprice = (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)
    expected_0 = (0.45 * 150 + 0.47 * 100) / 250
    assert np.isclose(microprice.iloc[0], expected_0)


def test_compute_book_imbalance():
    """Test book imbalance computation."""
    bid_sz = pd.Series([100, 200, 150])
    ask_sz = pd.Series([100, 100, 250])
    
    imbalance = compute_book_imbalance(bid_sz, ask_sz)
    
    # First: (100-100)/200 = 0
    assert np.isclose(imbalance.iloc[0], 0)
    # Second: (200-100)/300 = 0.333...
    assert np.isclose(imbalance.iloc[1], 1/3)
    # Third: (150-250)/400 = -0.25
    assert np.isclose(imbalance.iloc[2], -0.25)


def test_compute_spread_bps():
    """Test spread in basis points."""
    bid_px = pd.Series([0.45, 100.0])
    ask_px = pd.Series([0.47, 100.10])
    
    spread_bps = compute_spread_bps(bid_px, ask_px)
    
    # First: (0.47-0.45) / 0.46 * 10000 = ~434.78 bps
    assert spread_bps.iloc[0] > 400
    assert spread_bps.iloc[0] < 450
    
    # Second: 0.10 / 100.05 * 10000 = ~10 bps
    assert spread_bps.iloc[1] > 9
    assert spread_bps.iloc[1] < 11


def test_compute_vwap_cumulative():
    """Test cumulative VWAP."""
    prices = pd.Series([100, 102, 101])
    sizes = pd.Series([10, 20, 30])
    
    vwap = compute_vwap(prices, sizes, window=None)
    
    # Final VWAP = (100*10 + 102*20 + 101*30) / 60
    expected = (100*10 + 102*20 + 101*30) / 60
    assert np.isclose(vwap.iloc[-1], expected)


def test_compute_vwap_rolling():
    """Test rolling VWAP."""
    prices = pd.Series([100, 102, 101, 103])
    sizes = pd.Series([10, 20, 30, 40])
    
    vwap = compute_vwap(prices, sizes, window=2)
    
    # Third value: (102*20 + 101*30) / 50
    expected = (102*20 + 101*30) / 50
    assert np.isclose(vwap.iloc[2], expected)
