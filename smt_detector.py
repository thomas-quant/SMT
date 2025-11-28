"""
SMT (Smart Money Technique) Divergence Detector Module

This module provides stateless detection functions for identifying SMT patterns
in correlated financial assets. It detects three types of SMT divergences:
1. Micro SMT: 2-candle pattern comparing current vs previous candle
2. Swing SMT: Divergence at shared swing highs/lows
3. FVG SMT: Divergence at shared Fair Value Gaps

All functions return a signal dictionary when a pattern is detected, or None otherwise.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Tuple


def check_micro_smt(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names: Tuple[str, str] = ("A1", "A2")
) -> Optional[Dict[str, Any]]:
    """
    Checks for a 2-candle "Micro SMT" on the most recent candle.
    
    This pattern compares the high/low of the current candle (index -1)
    to the high/low of the previous candle (index -2).
    
    Args:
        df_a1: DataFrame for asset 1 with OHLC columns
        df_a2: DataFrame for asset 2 with OHLC columns
        asset_names: Tuple of asset names for labeling (default: ("A1", "A2"))
    
    Returns:
        Dictionary with signal details if SMT detected, None otherwise.
        Signal dict contains: signal_type, timestamp, sweeping_asset,
        failing_asset, reference_price, invalidation_level
    """
    # Ensure we have at least 2 candles
    if len(df_a1) < 2 or len(df_a2) < 2:
        return None
    
    # Get current candle (C0) and previous candle (C-1) data
    # A1
    c0_high_a1 = df_a1['High'].iloc[-1]
    c0_low_a1 = df_a1['Low'].iloc[-1]
    c_minus1_high_a1 = df_a1['High'].iloc[-2]
    c_minus1_low_a1 = df_a1['Low'].iloc[-2]
    
    # A2
    c0_high_a2 = df_a2['High'].iloc[-1]
    c0_low_a2 = df_a2['Low'].iloc[-1]
    c_minus1_high_a2 = df_a2['High'].iloc[-2]
    c_minus1_low_a2 = df_a2['Low'].iloc[-2]
    
    # Reference prices (previous candle's high/low)
    r_high_a1 = c_minus1_high_a1
    r_low_a1 = c_minus1_low_a1
    r_high_a2 = c_minus1_high_a2
    r_low_a2 = c_minus1_low_a2
    
    timestamp = df_a1.index[-1]
    
    # Check for Bearish SMT (sweeping highs)
    # Case 1: A1 sweeps, A2 fails
    if c0_high_a1 > r_high_a1 and c0_high_a2 < r_high_a2:
        return {
            "signal_type": "Bearish Micro SMT",
            "timestamp": timestamp,
            "sweeping_asset": asset_names[0],
            "failing_asset": asset_names[1],
            "reference_price": r_high_a1,
            "invalidation_level": r_high_a2
        }
    
    # Case 2: A2 sweeps, A1 fails
    if c0_high_a2 > r_high_a2 and c0_high_a1 < r_high_a1:
        return {
            "signal_type": "Bearish Micro SMT",
            "timestamp": timestamp,
            "sweeping_asset": asset_names[1],
            "failing_asset": asset_names[0],
            "reference_price": r_high_a2,
            "invalidation_level": r_high_a1
        }
    
    # Check for Bullish SMT (sweeping lows)
    # Case 1: A1 sweeps, A2 fails
    if c0_low_a1 < r_low_a1 and c0_low_a2 > r_low_a2:
        return {
            "signal_type": "Bullish Micro SMT",
            "timestamp": timestamp,
            "sweeping_asset": asset_names[0],
            "failing_asset": asset_names[1],
            "reference_price": r_low_a1,
            "invalidation_level": r_low_a2
        }
    
    # Case 2: A2 sweeps, A1 fails
    if c0_low_a2 < r_low_a2 and c0_low_a1 > r_low_a1:
        return {
            "signal_type": "Bullish Micro SMT",
            "timestamp": timestamp,
            "sweeping_asset": asset_names[1],
            "failing_asset": asset_names[0],
            "reference_price": r_low_a2,
            "invalidation_level": r_low_a1
        }
    
    # No SMT detected
    return None


def check_swing_smt(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names: Tuple[str, str] = ("A1", "A2")
) -> Optional[Dict[str, Any]]:
    """
    Checks for a "Swing SMT" completed by the most recent candle.
    
    It looks back 'lookback_period' candles to find a shared swing high/low
    and checks if the current candle (-1) created a divergence by sweeping it.
    
    Args:
        df_a1: DataFrame for asset 1 with OHLC columns
        df_a2: DataFrame for asset 2 with OHLC columns
        lookback_period: Number of candles to look back for swing point
        asset_names: Tuple of asset names for labeling (default: ("A1", "A2"))
    
    Returns:
        Dictionary with signal details if SMT detected, None otherwise.
        Signal dict contains: signal_type, timestamp, reference_timestamp,
        sweeping_asset, failing_asset, reference_price, invalidation_level
    """
    # Ensure we have enough data
    if len(df_a1) < lookback_period + 1 or len(df_a2) < lookback_period + 1:
        return None
    
    # Get lookback slice (excluding current candle)
    data_a1 = df_a1.iloc[-lookback_period-1:-1]
    data_a2 = df_a2.iloc[-lookback_period-1:-1]
    
    # Get current candle data
    c0_high_a1 = df_a1['High'].iloc[-1]
    c0_low_a1 = df_a1['Low'].iloc[-1]
    c0_high_a2 = df_a2['High'].iloc[-1]
    c0_low_a2 = df_a2['Low'].iloc[-1]
    
    timestamp = df_a1.index[-1]
    
    # ===== Check for Bullish Swing SMT (shared swing low) =====
    r_low_a1 = data_a1['Low'].min()
    r_low_a2 = data_a2['Low'].min()
    
    # Find timestamps of swing lows
    t_low_a1 = data_a1['Low'].idxmin()
    t_low_a2 = data_a2['Low'].idxmin()
    
    # Check if timestamps match (shared swing low)
    if t_low_a1 == t_low_a2:
        # Check for divergence: A1 sweeps, A2 fails
        if c0_low_a1 < r_low_a1 and c0_low_a2 > r_low_a2:
            return {
                "signal_type": "Bullish Swing SMT",
                "timestamp": timestamp,
                "reference_timestamp": t_low_a1,
                "sweeping_asset": asset_names[0],
                "failing_asset": asset_names[1],
                "reference_price": r_low_a1,
                "invalidation_level": r_low_a2
            }
        
        # Check for divergence: A2 sweeps, A1 fails
        if c0_low_a2 < r_low_a2 and c0_low_a1 > r_low_a1:
            return {
                "signal_type": "Bullish Swing SMT",
                "timestamp": timestamp,
                "reference_timestamp": t_low_a2,
                "sweeping_asset": asset_names[1],
                "failing_asset": asset_names[0],
                "reference_price": r_low_a2,
                "invalidation_level": r_low_a1
            }
    
    # ===== Check for Bearish Swing SMT (shared swing high) =====
    r_high_a1 = data_a1['High'].max()
    r_high_a2 = data_a2['High'].max()
    
    # Find timestamps of swing highs
    t_high_a1 = data_a1['High'].idxmax()
    t_high_a2 = data_a2['High'].idxmax()
    
    # Check if timestamps match (shared swing high)
    if t_high_a1 == t_high_a2:
        # Check for divergence: A1 sweeps, A2 fails
        if c0_high_a1 > r_high_a1 and c0_high_a2 < r_high_a2:
            return {
                "signal_type": "Bearish Swing SMT",
                "timestamp": timestamp,
                "reference_timestamp": t_high_a1,
                "sweeping_asset": asset_names[0],
                "failing_asset": asset_names[1],
                "reference_price": r_high_a1,
                "invalidation_level": r_high_a2
            }
        
        # Check for divergence: A2 sweeps, A1 fails
        if c0_high_a2 > r_high_a2 and c0_high_a1 < r_high_a1:
            return {
                "signal_type": "Bearish Swing SMT",
                "timestamp": timestamp,
                "reference_timestamp": t_high_a2,
                "sweeping_asset": asset_names[1],
                "failing_asset": asset_names[0],
                "reference_price": r_high_a2,
                "invalidation_level": r_high_a1
            }
    
    # No swing SMT detected
    return None


def check_fvg_smt(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names: Tuple[str, str] = ("A1", "A2")
) -> Optional[Dict[str, Any]]:
    """
    Checks for a "Key Level SMT" based on a Fair Value Gap (FVG).
    
    It looks back 'lookback_period' to find a shared FVG, then checks
    if the current candle (-1) created a divergence by sweeping/filling
    its FVG while the other asset failed to.
    
    Args:
        df_a1: DataFrame for asset 1 with OHLC columns
        df_a2: DataFrame for asset 2 with OHLC columns
        lookback_period: Number of candles to look back for FVG
        asset_names: Tuple of asset names for labeling (default: ("A1", "A2"))
    
    Returns:
        Dictionary with signal details if SMT detected, None otherwise.
        Signal dict contains: signal_type, timestamp, reference_timestamp,
        sweeping_asset, failing_asset, reference_price, invalidation_level
    """
    # Find FVGs in both assets (excluding current candle from the search)
    fvg_a1 = _find_recent_valid_fvg(df_a1, lookback_period)
    fvg_a2 = _find_recent_valid_fvg(df_a2, lookback_period)
    
    # Both must have FVGs
    if fvg_a1 is None or fvg_a2 is None:
        return None
    
    # FVGs must be at the same timestamp and of the same type
    if fvg_a1['timestamp'] != fvg_a2['timestamp']:
        return None
    
    if fvg_a1['type'] != fvg_a2['type']:
        return None
    
    # Get current candle data
    c0_high_a1 = df_a1['High'].iloc[-1]
    c0_low_a1 = df_a1['Low'].iloc[-1]
    c0_high_a2 = df_a2['High'].iloc[-1]
    c0_low_a2 = df_a2['Low'].iloc[-1]
    
    timestamp = df_a1.index[-1]
    fvg_type = fvg_a1['type']
    
    # ===== Check for Bullish FVG SMT =====
    if fvg_type == 'bullish':
        # Use the current fill level (top) from the FVG info
        ref_price_a1 = fvg_a1['top']
        ref_price_a2 = fvg_a2['top']
        
        # A1 breaks FVG top (low goes below top), A2 fails to break (low stays above top)
        if c0_low_a1 < ref_price_a1 and c0_low_a2 >= ref_price_a2:
            return {
                "signal_type": "Bullish FVG SMT",
                "timestamp": timestamp,
                "reference_timestamp": fvg_a1['timestamp'],
                "sweeping_asset": asset_names[0],
                "failing_asset": asset_names[1],
                "reference_price": ref_price_a1,
                "invalidation_level": ref_price_a2
            }
        
        # A2 breaks FVG top (low goes below top), A1 fails to break (low stays above top)
        if c0_low_a2 < ref_price_a2 and c0_low_a1 >= ref_price_a1:
            return {
                "signal_type": "Bullish FVG SMT",
                "timestamp": timestamp,
                "reference_timestamp": fvg_a2['timestamp'],
                "sweeping_asset": asset_names[1],
                "failing_asset": asset_names[0],
                "reference_price": ref_price_a2,
                "invalidation_level": ref_price_a1
            }
    
    # ===== Check for Bearish FVG SMT =====
    elif fvg_type == 'bearish':
        # Use the current fill level (bottom) from the FVG info
        ref_price_a1 = fvg_a1['bottom']
        ref_price_a2 = fvg_a2['bottom']
        
        # A1 breaks FVG bottom (high goes above bottom), A2 fails to break (high stays below bottom)
        if c0_high_a1 > ref_price_a1 and c0_high_a2 <= ref_price_a2:
            return {
                "signal_type": "Bearish FVG SMT",
                "timestamp": timestamp,
                "reference_timestamp": fvg_a1['timestamp'],
                "sweeping_asset": asset_names[0],
                "failing_asset": asset_names[1],
                "reference_price": ref_price_a1,
                "invalidation_level": ref_price_a2
            }
        
        # A2 breaks FVG bottom (high goes above bottom), A1 fails to break (high stays below bottom)
        if c0_high_a2 > ref_price_a2 and c0_high_a1 <= ref_price_a1:
            return {
                "signal_type": "Bearish FVG SMT",
                "timestamp": timestamp,
                "reference_timestamp": fvg_a2['timestamp'],
                "sweeping_asset": asset_names[1],
                "failing_asset": asset_names[0],
                "reference_price": ref_price_a2,
                "invalidation_level": ref_price_a1
            }
    
    # No FVG SMT detected
    return None


def _find_recent_valid_fvg(df: pd.DataFrame, lookback: int) -> Optional[Dict[str, Any]]:
    """
    Helper function to find the most recent VALID FVG (Bullish or Bearish)
    within the lookback period.
    
    A valid FVG:
    1. Must not have been fully filled by both sides
    2. Tracks partial fills to use as the reference price
    
    An FVG is a 3-candle pattern:
    - Bullish FVG: High(C1) < Low(C3), gap from High(C1) to Low(C3)
    - Bearish FVG: Low(C1) > High(C3), gap from High(C3) to Low(C1)
    
    Args:
        df: DataFrame with OHLC columns
        lookback: Number of candles to look back
    
    Returns:
        Dictionary with FVG info: {'type': 'bullish'/'bearish', 'top': float,
        'bottom': float, 'timestamp': timestamp} or None if no valid FVG found.
    """
    # Need at least 3 candles for FVG pattern
    if len(df) < 3:
        return None
    
    # Determine how far back we can look
    # Each pattern needs 3 candles: C1, C2, C3
    max_lookback = min(lookback, len(df) - 3)
    
    # Scan backwards from most recent (excluding current candle) to find the first VALID FVG
    for i in range(1, max_lookback + 1):
        # For a pattern where C3 is at position -(i+1):
        # C1 is at -(i+3), C2 is at -(i+2), C3 is at -(i+1)
        # We skip index -1 (current candle)
        if i + 3 > len(df):
            break
        
        c1_idx = -(i+3)
        c3_idx = -(i+1)
        
        c1_high = df['High'].iloc[c1_idx]
        c1_low = df['Low'].iloc[c1_idx]
        c3_high = df['High'].iloc[c3_idx]
        c3_low = df['Low'].iloc[c3_idx]
        
        # The timestamp for the FVG is the timestamp of C1 (the first candle)
        c1_timestamp = df.index[c1_idx]
        
        # Check for Bullish FVG: High(C1) < Low(C3)
        if c1_high < c3_low:
            original_bottom = c1_high
            original_top = c3_low
            
            # Check if FVG has been fully filled (invalidated)
            # Get all candles after C3 up to (but not including) current candle
            candles_after = df.iloc[c3_idx+1:-1]
            
            if len(candles_after) > 0:
                # Check if any candle's low went below the bottom (fully filled)
                if (candles_after['Low'] < original_bottom).any():
                    # FVG fully filled, skip it
                    continue
                
                # Find the lowest low that touched the FVG from above
                # (partial fill tracking)
                lows_in_gap = candles_after[candles_after['Low'] < original_top]['Low']
                if len(lows_in_gap) > 0:
                    # FVG partially filled - use the deepest penetration as new top
                    current_top = lows_in_gap.min()
                else:
                    # FVG untouched
                    current_top = original_top
            else:
                # No candles after, FVG is pristine
                current_top = original_top
            
            return {
                'type': 'bullish',
                'bottom': original_bottom,
                'top': current_top,
                'timestamp': c1_timestamp
            }
        
        # Check for Bearish FVG: Low(C1) > High(C3)
        if c1_low > c3_high:
            original_bottom = c3_high
            original_top = c1_low
            
            # Check if FVG has been fully filled (invalidated)
            # Get all candles after C3 up to (but not including) current candle
            candles_after = df.iloc[c3_idx+1:-1]
            
            if len(candles_after) > 0:
                # Check if any candle's high went above the top (fully filled)
                if (candles_after['High'] > original_top).any():
                    # FVG fully filled, skip it
                    continue
                
                # Find the highest high that touched the FVG from below
                # (partial fill tracking)
                highs_in_gap = candles_after[candles_after['High'] > original_bottom]['High']
                if len(highs_in_gap) > 0:
                    # FVG partially filled - use the highest penetration as new bottom
                    current_bottom = highs_in_gap.max()
                else:
                    # FVG untouched
                    current_bottom = original_bottom
            else:
                # No candles after, FVG is pristine
                current_bottom = original_bottom
            
            return {
                'type': 'bearish',
                'bottom': current_bottom,
                'top': original_top,
                'timestamp': c1_timestamp
            }
    
    # No valid FVG found
    return None
