"""Microstructure feature computation."""

import numpy as np
import pandas as pd


def compute_microprice(
    bid_px: pd.Series | np.ndarray,
    ask_px: pd.Series | np.ndarray,
    bid_sz: pd.Series | np.ndarray,
    ask_sz: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Compute microprice (size-weighted mid price).
    
    Microprice weights the mid towards the side with less size,
    reflecting where the next trade is more likely to occur.
    
    Formula: (bid * ask_sz + ask * bid_sz) / (bid_sz + ask_sz)
    
    Args:
        bid_px: Best bid prices
        ask_px: Best ask prices
        bid_sz: Best bid sizes
        ask_sz: Best ask sizes
        
    Returns:
        Microprice series/array
    """
    total_size = bid_sz + ask_sz
    
    microprice = np.where(
        total_size > 0,
        (bid_px * ask_sz + ask_px * bid_sz) / total_size,
        (bid_px + ask_px) / 2  # Fall back to mid if no size
    )
    
    if isinstance(bid_px, pd.Series):
        return pd.Series(microprice, index=bid_px.index)
    return microprice


def compute_weighted_mid(
    bid_px: pd.Series | np.ndarray,
    ask_px: pd.Series | np.ndarray,
    bid_sz: pd.Series | np.ndarray,
    ask_sz: pd.Series | np.ndarray,
    levels: int = 1,
) -> pd.Series | np.ndarray:
    """Compute weighted mid price from BBO.
    
    This is an alias for microprice when levels=1.
    For multi-level weighted mid, use book data directly.
    
    Args:
        bid_px: Best bid prices
        ask_px: Best ask prices
        bid_sz: Best bid sizes
        ask_sz: Best ask sizes
        levels: Number of levels to consider (only 1 supported for BBO)
        
    Returns:
        Weighted mid price
    """
    if levels != 1:
        raise ValueError("Only levels=1 supported for BBO data. Use book data for deeper levels.")
    
    return compute_microprice(bid_px, ask_px, bid_sz, ask_sz)


def compute_vwap(
    prices: pd.Series,
    sizes: pd.Series,
    window: int | None = None,
) -> pd.Series:
    """Compute Volume-Weighted Average Price.
    
    Args:
        prices: Trade prices
        sizes: Trade sizes
        window: Rolling window size (None = cumulative VWAP)
        
    Returns:
        VWAP series
    """
    if window is None:
        # Cumulative VWAP
        cumulative_value = (prices * sizes).cumsum()
        cumulative_volume = sizes.cumsum()
        return cumulative_value / cumulative_volume
    else:
        # Rolling VWAP
        rolling_value = (prices * sizes).rolling(window).sum()
        rolling_volume = sizes.rolling(window).sum()
        return rolling_value / rolling_volume


def compute_trade_imbalance(
    sides: pd.Series,
    sizes: pd.Series,
    window: int = 100,
) -> pd.Series:
    """Compute trade imbalance (buy volume - sell volume) / total volume.
    
    Args:
        sides: Trade sides ("buy" or "sell")
        sizes: Trade sizes
        window: Rolling window size
        
    Returns:
        Imbalance series in range [-1, 1]
    """
    buy_mask = sides == "buy"
    sell_mask = sides == "sell"
    
    buy_volume = (sizes * buy_mask.astype(float)).rolling(window).sum()
    sell_volume = (sizes * sell_mask.astype(float)).rolling(window).sum()
    total_volume = buy_volume + sell_volume
    
    imbalance = np.where(
        total_volume > 0,
        (buy_volume - sell_volume) / total_volume,
        0
    )
    
    return pd.Series(imbalance, index=sides.index)


def compute_book_imbalance(
    bid_sz: pd.Series | np.ndarray,
    ask_sz: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Compute book imbalance from BBO sizes.
    
    Formula: (bid_sz - ask_sz) / (bid_sz + ask_sz)
    
    Positive = more bid size (buying pressure)
    Negative = more ask size (selling pressure)
    
    Args:
        bid_sz: Best bid sizes
        ask_sz: Best ask sizes
        
    Returns:
        Imbalance in range [-1, 1]
    """
    total = bid_sz + ask_sz
    imbalance = np.where(
        total > 0,
        (bid_sz - ask_sz) / total,
        0
    )
    
    if isinstance(bid_sz, pd.Series):
        return pd.Series(imbalance, index=bid_sz.index)
    return imbalance


def compute_book_depth_imbalance(
    bid_prices: list[list[float]],
    bid_sizes: list[list[float]],
    ask_prices: list[list[float]],
    ask_sizes: list[list[float]],
    depth: int = 5,
) -> np.ndarray:
    """Compute book imbalance from L2 order book snapshots.
    
    Sums size up to `depth` levels on each side.
    
    Args:
        bid_prices: List of bid price arrays (per snapshot)
        bid_sizes: List of bid size arrays (per snapshot)
        ask_prices: List of ask price arrays (per snapshot)
        ask_sizes: List of ask size arrays (per snapshot)
        depth: Number of levels to consider
        
    Returns:
        Imbalance array in range [-1, 1]
    """
    imbalances = []
    
    for bids, asks in zip(bid_sizes, ask_sizes):
        bid_sum = sum(bids[:depth]) if len(bids) > 0 else 0
        ask_sum = sum(asks[:depth]) if len(asks) > 0 else 0
        total = bid_sum + ask_sum
        
        if total > 0:
            imbalances.append((bid_sum - ask_sum) / total)
        else:
            imbalances.append(0.0)
    
    return np.array(imbalances)


def compute_spread_bps(
    bid_px: pd.Series | np.ndarray,
    ask_px: pd.Series | np.ndarray,
) -> pd.Series | np.ndarray:
    """Compute spread in basis points.
    
    Args:
        bid_px: Best bid prices
        ask_px: Best ask prices
        
    Returns:
        Spread in basis points (100 bps = 1%)
    """
    mid = (bid_px + ask_px) / 2
    spread_bps = (ask_px - bid_px) / mid * 10000
    
    if isinstance(bid_px, pd.Series):
        return pd.Series(spread_bps, index=bid_px.index)
    return spread_bps
