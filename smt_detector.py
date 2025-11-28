"""
SMT (Smart Money Technique) Divergence Detector Module - REFACTORED

Improvements:
1. Eliminated code duplication through helper functions
2. Consistent sweep/fail logic across all functions
3. More robust extreme validation with <= instead of ==
4. Clearer structure and better comments
"""

import pandas as pd
from typing import Optional, Dict, Any, Tuple


def _check_divergence(
    c0_val_a1: float, 
    c0_val_a2: float, 
    ref_val_a1: float, 
    ref_val_a2: float,
    is_bullish: bool,
    allow_equal_sweep: bool = False
) -> Optional[int]:
    """
    Check for divergence between two assets.
    
    Sweep logic depends on allow_equal_sweep:
    - If False (default): Sweep = breaking through (>, <), Fail = at or beyond (>=, <=)
    - If True (micro SMT): Sweep = touching or breaking (>=, <=), Fail = staying beyond (>, <)
    
    Returns:
        0 if A1 sweeps and A2 fails
        1 if A2 sweeps and A1 fails
        None if no divergence detected
    """
    if allow_equal_sweep:
        # Micro SMT logic: equal counts as sweep (order fill logic)
        if is_bullish:
            # Sweep = at or below, Fail = strictly above
            a1_sweeps = c0_val_a1 <= ref_val_a1 and c0_val_a2 > ref_val_a2
            a2_sweeps = c0_val_a2 <= ref_val_a2 and c0_val_a1 > ref_val_a1
        else:
            # Sweep = at or above, Fail = strictly below
            a1_sweeps = c0_val_a1 >= ref_val_a1 and c0_val_a2 < ref_val_a2
            a2_sweeps = c0_val_a2 >= ref_val_a2 and c0_val_a1 < ref_val_a1
    else:
        # Swing/FVG SMT logic: must break through
        if is_bullish:
            # Sweep = strictly below, Fail = at or above
            a1_sweeps = c0_val_a1 < ref_val_a1 and c0_val_a2 >= ref_val_a2
            a2_sweeps = c0_val_a2 < ref_val_a2 and c0_val_a1 >= ref_val_a1
        else:
            # Sweep = strictly above, Fail = at or below
            a1_sweeps = c0_val_a1 > ref_val_a1 and c0_val_a2 <= ref_val_a2
            a2_sweeps = c0_val_a2 > ref_val_a2 and c0_val_a1 <= ref_val_a1
    
    if a1_sweeps:
        return 0
    elif a2_sweeps:
        return 1
    return None


def _get_timestamp_index(df: pd.DataFrame, timestamp) -> Optional[int]:
    """Safely get the integer index for a timestamp."""
    try:
        idx = df.index.get_loc(timestamp)
        if isinstance(idx, slice):
            return idx.start
        return idx
    except (KeyError, TypeError):
        return None


def _validate_current_is_extreme(
    df: pd.DataFrame, 
    swing_idx: int, 
    is_bullish: bool
) -> bool:
    """
    Validate that current candle is the extreme point between swing and now.
    Uses <= for robustness against floating point precision issues.
    """
    if swing_idx is None or swing_idx >= len(df) - 1:
        return False
    
    between_candles = df.iloc[swing_idx + 1:]
    if len(between_candles) == 0:
        return False
    
    current_val = df.iloc[-1]['Low' if is_bullish else 'High']
    
    if is_bullish:
        # Current must be <= all lows between swing and now
        return current_val <= between_candles['Low'].min()
    else:
        # Current must be >= all highs between swing and now
        return current_val >= between_candles['High'].max()


def check_micro_smt(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names: Tuple[str, str] = ("A1", "A2")
) -> Optional[Dict[str, Any]]:
    """
    Checks for a 2-candle "Micro SMT" on the most recent candle.
    
    This pattern compares the high/low of the current candle (index -1)
    to the high/low of the previous candle (index -2).
    """
    if len(df_a1) < 2 or len(df_a2) < 2:
        return None
    
    # Current candle values
    c0_high_a1, c0_low_a1 = df_a1['High'].iloc[-1], df_a1['Low'].iloc[-1]
    c0_high_a2, c0_low_a2 = df_a2['High'].iloc[-1], df_a2['Low'].iloc[-1]
    
    # Reference values (previous candle)
    r_high_a1, r_low_a1 = df_a1['High'].iloc[-2], df_a1['Low'].iloc[-2]
    r_high_a2, r_low_a2 = df_a2['High'].iloc[-2], df_a2['Low'].iloc[-2]
    
    timestamp = df_a1.index[-1]
    
    # Check for Bearish SMT (sweeping highs)
    sweeper = _check_divergence(c0_high_a1, c0_high_a2, r_high_a1, r_high_a2, 
                                 is_bullish=False, allow_equal_sweep=True)
    if sweeper is not None:
        return {
            "signal_type": "Bearish Micro SMT",
            "timestamp": timestamp,
            "sweeping_asset": asset_names[sweeper],
            "failing_asset": asset_names[1 - sweeper],
            "reference_price": r_high_a1 if sweeper == 0 else r_high_a2,
            "invalidation_level": r_high_a2 if sweeper == 0 else r_high_a1
        }
    
    # Check for Bullish SMT (sweeping lows)
    sweeper = _check_divergence(c0_low_a1, c0_low_a2, r_low_a1, r_low_a2, 
                                 is_bullish=True, allow_equal_sweep=True)
    if sweeper is not None:
        return {
            "signal_type": "Bullish Micro SMT",
            "timestamp": timestamp,
            "sweeping_asset": asset_names[sweeper],
            "failing_asset": asset_names[1 - sweeper],
            "reference_price": r_low_a1 if sweeper == 0 else r_low_a2,
            "invalidation_level": r_low_a2 if sweeper == 0 else r_low_a1
        }
    
    return None


def _find_valid_swing_low(df: pd.DataFrame, lookback: int) -> Optional[Dict[str, Any]]:
    """
    Find the most recent valid swing low within the lookback period.
    A valid swing low must be lower than both adjacent candles.
    """
    if len(df) < 3:
        return None
    
    # Scan backwards, need at least one candle on each side
    for i in range(len(df) - 2, 0, -1):
        if (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
            df['Low'].iloc[i] < df['Low'].iloc[i+1]):
            return {
                'timestamp': df.index[i],
                'price': df['Low'].iloc[i]
            }
    
    return None


def _find_valid_swing_high(df: pd.DataFrame, lookback: int) -> Optional[Dict[str, Any]]:
    """
    Find the most recent valid swing high within the lookback period.
    A valid swing high must be higher than both adjacent candles.
    """
    if len(df) < 3:
        return None
    
    # Scan backwards, need at least one candle on each side
    for i in range(len(df) - 2, 0, -1):
        if (df['High'].iloc[i] > df['High'].iloc[i-1] and 
            df['High'].iloc[i] > df['High'].iloc[i+1]):
            return {
                'timestamp': df.index[i],
                'price': df['High'].iloc[i]
            }
    
    return None


def check_swing_smt(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names: Tuple[str, str] = ("A1", "A2"),
    timestamp_tolerance: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Checks for a "Swing SMT" completed by the most recent candle.
    
    Looks back to find shared swing highs/lows and validates that the current
    candle creates a divergence by sweeping the swing while the other asset fails.
    
    IMPORTANT: Validates that the sweeping asset hasn't previously broken the swing
    between the swing point and current candle. This prevents testing invalidated swings.
    
    Args:
        timestamp_tolerance: Allow swing timestamps to differ by this many candles.
                           Default 1 allows for micro SMTs between correlated assets.
    """
    if len(df_a1) < lookback_period + 1 or len(df_a2) < lookback_period + 1:
        return None
    
    # Get lookback slice (excluding current candle)
    data_a1 = df_a1.iloc[-lookback_period-1:-1]
    data_a2 = df_a2.iloc[-lookback_period-1:-1]
    
    # Current candle values
    c0_high_a1, c0_low_a1 = df_a1['High'].iloc[-1], df_a1['Low'].iloc[-1]
    c0_high_a2, c0_low_a2 = df_a2['High'].iloc[-1], df_a2['Low'].iloc[-1]
    
    timestamp = df_a1.index[-1]
    
    # Helper function to check swing SMT for a given direction
    def _check_swing_direction(
        swing_finder,
        dfs: Tuple[pd.DataFrame, pd.DataFrame],
        c0_vals: Tuple[float, float],
        is_bullish: bool
    ):
        swing_a1 = swing_finder(data_a1, lookback_period)
        swing_a2 = swing_finder(data_a2, lookback_period)
        
        # Both must have valid swings
        if swing_a1 is None or swing_a2 is None:
            return None
        
        # Check if timestamps are within tolerance window
        ts_a1 = swing_a1['timestamp']
        ts_a2 = swing_a2['timestamp']
        
        # Get indices to calculate time difference
        idx_a1 = _get_timestamp_index(df_a1, ts_a1)
        idx_a2 = _get_timestamp_index(df_a2, ts_a2)
        
        if idx_a1 is None or idx_a2 is None:
            return None
        
        # Check if swings are within tolerance (measured in candle count)
        if abs(idx_a1 - idx_a2) > timestamp_tolerance:
            return None
        
        ref_price_a1 = swing_a1['price']
        ref_price_a2 = swing_a2['price']
        
        # Use the earlier timestamp as reference (more conservative)
        swing_timestamp = ts_a1 if idx_a1 <= idx_a2 else ts_a2
        
        # Check for divergence
        sweeper = _check_divergence(
            c0_vals[0], c0_vals[1], 
            ref_price_a1, ref_price_a2, 
            is_bullish
        )
        
        if sweeper is None:
            return None
        
        # Critical validation: current candle must be the extreme point
        # between swing and now for the sweeping asset
        # This ensures the swing hasn't been previously invalidated
        swing_idx = idx_a1 if sweeper == 0 else idx_a2
        if not _validate_current_is_extreme(dfs[sweeper], swing_idx, is_bullish):
            return None
        
        signal_type = "Bullish Swing SMT" if is_bullish else "Bearish Swing SMT"
        return {
            "signal_type": signal_type,
            "timestamp": timestamp,
            "reference_timestamp": swing_timestamp,
            "sweeping_asset": asset_names[sweeper],
            "failing_asset": asset_names[1 - sweeper],
            "reference_price": ref_price_a1 if sweeper == 0 else ref_price_a2,
            "invalidation_level": ref_price_a2 if sweeper == 0 else ref_price_a1
        }
    
    # Check for Bullish Swing SMT (shared swing low)
    result = _check_swing_direction(
        _find_valid_swing_low,
        (df_a1, df_a2),
        (c0_low_a1, c0_low_a2),
        is_bullish=True
    )
    if result:
        return result
    
    # Check for Bearish Swing SMT (shared swing high)
    result = _check_swing_direction(
        _find_valid_swing_high,
        (df_a1, df_a2),
        (c0_high_a1, c0_high_a2),
        is_bullish=False
    )
    if result:
        return result
    
    return None


def check_fvg_smt(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names: Tuple[str, str] = ("A1", "A2")
) -> Optional[Dict[str, Any]]:
    """
    Checks for a "Key Level SMT" based on a Fair Value Gap (FVG).
    """
    fvg_a1 = _find_recent_valid_fvg(df_a1, lookback_period)
    fvg_a2 = _find_recent_valid_fvg(df_a2, lookback_period)
    
    # Both must have FVGs at same timestamp and of same type
    if (fvg_a1 is None or fvg_a2 is None or
        fvg_a1['timestamp'] != fvg_a2['timestamp'] or
        fvg_a1['type'] != fvg_a2['type']):
        return None
    
    # Current candle values
    c0_high_a1, c0_low_a1 = df_a1['High'].iloc[-1], df_a1['Low'].iloc[-1]
    c0_high_a2, c0_low_a2 = df_a2['High'].iloc[-1], df_a2['Low'].iloc[-1]
    
    timestamp = df_a1.index[-1]
    fvg_type = fvg_a1['type']
    is_bullish = (fvg_type == 'bullish')
    
    # Get reference prices based on FVG type
    if is_bullish:
        ref_price_a1, ref_price_a2 = fvg_a1['top'], fvg_a2['top']
        c0_vals = (c0_low_a1, c0_low_a2)
    else:
        ref_price_a1, ref_price_a2 = fvg_a1['bottom'], fvg_a2['bottom']
        c0_vals = (c0_high_a1, c0_high_a2)
    
    # Check for divergence
    sweeper = _check_divergence(
        c0_vals[0], c0_vals[1],
        ref_price_a1, ref_price_a2,
        is_bullish
    )
    
    if sweeper is None:
        return None
    
    signal_type = f"{'Bullish' if is_bullish else 'Bearish'} FVG SMT"
    return {
        "signal_type": signal_type,
        "timestamp": timestamp,
        "reference_timestamp": fvg_a1['timestamp'],
        "sweeping_asset": asset_names[sweeper],
        "failing_asset": asset_names[1 - sweeper],
        "reference_price": ref_price_a1 if sweeper == 0 else ref_price_a2,
        "invalidation_level": ref_price_a2 if sweeper == 0 else ref_price_a1
    }


def _find_recent_valid_fvg(df: pd.DataFrame, lookback: int) -> Optional[Dict[str, Any]]:
    """
    Find the most recent VALID FVG within the lookback period.
    
    An FVG is a 3-candle pattern:
    - Bullish FVG: High(C1) < Low(C3)
    - Bearish FVG: Low(C1) > High(C3)
    
    Tracks partial fills and excludes fully filled FVGs.
    """
    if len(df) < 3:
        return None
    
    max_lookback = min(lookback, len(df) - 3)
    
    # Scan backwards from most recent (excluding current candle)
    for i in range(1, max_lookback + 1):
        if i + 3 > len(df):
            break
        
        c1_idx = -(i + 3)
        c3_idx = -(i + 1)
        
        c1_high, c1_low = df['High'].iloc[c1_idx], df['Low'].iloc[c1_idx]
        c3_high, c3_low = df['High'].iloc[c3_idx], df['Low'].iloc[c3_idx]
        c1_timestamp = df.index[c1_idx]
        
        # Get candles after C3 (up to but not including current)
        candles_after = df.iloc[c3_idx+1:-1]
        
        # Check for Bullish FVG: High(C1) < Low(C3)
        if c1_high < c3_low:
            original_bottom, original_top = c1_high, c3_low
            
            if len(candles_after) > 0:
                # Skip if fully filled (low went below bottom)
                if (candles_after['Low'] < original_bottom).any():
                    continue
                
                # Track partial fills
                lows_in_gap = candles_after[candles_after['Low'] < original_top]['Low']
                current_top = lows_in_gap.min() if len(lows_in_gap) > 0 else original_top
            else:
                current_top = original_top
            
            return {
                'type': 'bullish',
                'bottom': original_bottom,
                'top': current_top,
                'timestamp': c1_timestamp
            }
        
        # Check for Bearish FVG: Low(C1) > High(C3)
        if c1_low > c3_high:
            original_bottom, original_top = c3_high, c1_low
            
            if len(candles_after) > 0:
                # Skip if fully filled (high went above top)
                if (candles_after['High'] > original_top).any():
                    continue
                
                # Track partial fills
                highs_in_gap = candles_after[candles_after['High'] > original_bottom]['High']
                current_bottom = highs_in_gap.max() if len(highs_in_gap) > 0 else original_bottom
            else:
                current_bottom = original_bottom
            
            return {
                'type': 'bearish',
                'bottom': current_bottom,
                'top': original_top,
                'timestamp': c1_timestamp
            }
    
    return None
