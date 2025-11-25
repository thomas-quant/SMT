"""
SMT Detector Testing Script

This script loads CSV data for two correlated assets and tests the SMT detector
by running it across each candle in the dataset. It displays all detected signals
in a readable format.

Usage:
    python test_smt_detector.py

Expected CSV format:
    - Two CSV files: one for each asset (e.g., NQ.csv, ES.csv)
    - Columns: timestamp (or datetime index), Open, High, Low, Close
    - Same timestamps in both files (aligned data)
"""

import pandas as pd
import smt_detector as smt
from datetime import datetime


def load_data(file_a1: str, file_a2: str):
    """
    Load CSV data for both assets and prepare DataFrames.
    
    Args:
        file_a1: Path to CSV file for asset 1
        file_a2: Path to CSV file for asset 2
    
    Returns:
        Tuple of (df_a1, df_a2) with time-indexed DataFrames
    """
    # Load CSVs
    df_a1 = pd.read_csv(file_a1)
    df_a2 = pd.read_csv(file_a2)
    
    # Try to find the timestamp column
    time_cols = ['timestamp', 'Timestamp', 'datetime', 'Datetime', 'Date', 'date', 'time', 'Time']
    time_col = None
    for col in time_cols:
        if col in df_a1.columns:
            time_col = col
            break
    
    # Set index to timestamp if found
    if time_col:
        df_a1[time_col] = pd.to_datetime(df_a1[time_col])
        df_a2[time_col] = pd.to_datetime(df_a2[time_col])
        df_a1.set_index(time_col, inplace=True)
        df_a2.set_index(time_col, inplace=True)
    else:
        # Assume first column or use integer index
        if df_a1.columns[0] not in ['Open', 'High', 'Low', 'Close']:
            df_a1.set_index(df_a1.columns[0], inplace=True)
            df_a2.set_index(df_a2.columns[0], inplace=True)
    
    # Ensure we have OHLC columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    for col in required_cols:
        if col not in df_a1.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df_a1, df_a2


def print_signal(signal: dict, candle_idx: int, total_candles: int):
    """
    Pretty print a detected signal.
    
    Args:
        signal: Signal dictionary from SMT detector
        candle_idx: Current candle index (0-based)
        total_candles: Total number of candles
    """
    print("\n" + "="*70)
    print(f"🚨 SIGNAL DETECTED at Candle #{candle_idx + 1}/{total_candles}")
    print("="*70)
    print(f"Type:              {signal['signal_type']}")
    print(f"Timestamp:         {signal['timestamp']}")
    if 'reference_timestamp' in signal:
        print(f"Reference Time:    {signal['reference_timestamp']}")
    print(f"Sweeping Asset:    {signal['sweeping_asset']}")
    print(f"Failing Asset:     {signal['failing_asset']}")
    print(f"Reference Price:   {signal['reference_price']:.2f}")
    print(f"Invalidation:      {signal['invalidation_level']:.2f}")
    print("="*70)


def test_smt_detector(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names: tuple = ("NQ", "ES"),
    lookback_swing: int = 20,
    lookback_fvg: int = 20,
    start_idx: int = None
):
    """
    Test SMT detector by iterating through the data candle by candle.
    
    Args:
        df_a1: DataFrame for asset 1
        df_a2: DataFrame for asset 2
        asset_names: Tuple of asset names
        lookback_swing: Lookback period for swing SMT
        lookback_fvg: Lookback period for FVG SMT
        start_idx: Starting index (default: max lookback needed)
    """
    # Determine starting point (need enough history)
    min_start = max(lookback_swing, lookback_fvg) + 3  # +3 for FVG pattern
    if start_idx is None:
        start_idx = min_start
    else:
        start_idx = max(start_idx, min_start)
    
    total_candles = len(df_a1)
    
    print(f"\n{'='*70}")
    print(f"SMT DETECTOR TEST")
    print(f"{'='*70}")
    print(f"Asset 1:           {asset_names[0]}")
    print(f"Asset 2:           {asset_names[1]}")
    print(f"Total Candles:     {total_candles}")
    print(f"Testing from:      Candle #{start_idx + 1} to #{total_candles}")
    print(f"Swing Lookback:    {lookback_swing}")
    print(f"FVG Lookback:      {lookback_fvg}")
    print(f"{'='*70}\n")
    
    # Counters for statistics
    micro_count = 0
    swing_count = 0
    fvg_count = 0
    
    # Iterate through data, testing at each candle
    for i in range(start_idx, total_candles):
        # Get data up to current candle (inclusive)
        current_a1 = df_a1.iloc[:i+1]
        current_a2 = df_a2.iloc[:i+1]
        
        # Test Micro SMT
        signal = smt.check_micro_smt(current_a1, current_a2, asset_names)
        if signal:
            print_signal(signal, i, total_candles)
            micro_count += 1
        
        # Test Swing SMT
        signal = smt.check_swing_smt(
            current_a1, current_a2, lookback_swing, asset_names
        )
        if signal:
            print_signal(signal, i, total_candles)
            swing_count += 1
        
        # Test FVG SMT
        signal = smt.check_fvg_smt(
            current_a1, current_a2, lookback_fvg, asset_names
        )
        if signal:
            print_signal(signal, i, total_candles)
            fvg_count += 1
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Micro SMT Signals: {micro_count}")
    print(f"Swing SMT Signals: {swing_count}")
    print(f"FVG SMT Signals:   {fvg_count}")
    print(f"Total Signals:     {micro_count + swing_count + fvg_count}")
    print("="*70 + "\n")


def inspect_candle(df_a1: pd.DataFrame, df_a2: pd.DataFrame, idx: int, asset_names: tuple):
    """
    Inspect a specific candle and show OHLC data for debugging.
    
    Args:
        df_a1: DataFrame for asset 1
        df_a2: DataFrame for asset 2
        idx: Candle index to inspect
        asset_names: Tuple of asset names
    """
    print(f"\n{'='*70}")
    print(f"CANDLE INSPECTION - Index {idx}")
    print(f"{'='*70}")
    
    if idx >= len(df_a1):
        print("ERROR: Index out of range")
        return
    
    # Show current and previous candle
    for offset, label in [(0, "Current"), (-1, "Previous")]:
        actual_idx = idx + offset
        if actual_idx < 0:
            continue
            
        print(f"\n{label} Candle (idx {actual_idx}):")
        print(f"Timestamp: {df_a1.index[actual_idx]}")
        
        print(f"\n{asset_names[0]}:")
        print(f"  Open:  {df_a1['Open'].iloc[actual_idx]:.2f}")
        print(f"  High:  {df_a1['High'].iloc[actual_idx]:.2f}")
        print(f"  Low:   {df_a1['Low'].iloc[actual_idx]:.2f}")
        print(f"  Close: {df_a1['Close'].iloc[actual_idx]:.2f}")
        
        print(f"\n{asset_names[1]}:")
        print(f"  Open:  {df_a2['Open'].iloc[actual_idx]:.2f}")
        print(f"  High:  {df_a2['High'].iloc[actual_idx]:.2f}")
        print(f"  Low:   {df_a2['Low'].iloc[actual_idx]:.2f}")
        print(f"  Close: {df_a2['Close'].iloc[actual_idx]:.2f}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # ===== CONFIGURATION =====
    # Update these paths to your CSV files
    FILE_A1 = "NQ.csv"  # Asset 1 (e.g., Nasdaq futures)
    FILE_A2 = "ES.csv"  # Asset 2 (e.g., S&P 500 futures)
    ASSET_NAMES = ("NQ", "ES")
    
    # Detection parameters
    LOOKBACK_SWING = 20  # Candles to look back for swing highs/lows
    LOOKBACK_FVG = 20    # Candles to look back for FVGs
    
    # ===== LOAD DATA =====
    print("Loading data...")
    try:
        df_a1, df_a2 = load_data(FILE_A1, FILE_A2)
        print(f"✓ Loaded {len(df_a1)} candles for {ASSET_NAMES[0]}")
        print(f"✓ Loaded {len(df_a2)} candles for {ASSET_NAMES[1]}")
        
        # Verify data is aligned
        if len(df_a1) != len(df_a2):
            print("WARNING: DataFrames have different lengths!")
            print(f"{ASSET_NAMES[0]}: {len(df_a1)} candles")
            print(f"{ASSET_NAMES[1]}: {len(df_a2)} candles")
            
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Could not find CSV file")
        print(f"Please update FILE_A1 and FILE_A2 in the script to point to your CSV files")
        print(f"\nExpected files:")
        print(f"  - {FILE_A1}")
        print(f"  - {FILE_A2}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR loading data: {e}")
        exit(1)
    
    # ===== RUN TESTS =====
    test_smt_detector(
        df_a1, df_a2,
        asset_names=ASSET_NAMES,
        lookback_swing=LOOKBACK_SWING,
        lookback_fvg=LOOKBACK_FVG
    )
    
    # ===== OPTIONAL: INSPECT SPECIFIC CANDLES =====
    # Uncomment to inspect specific candles for debugging
    # inspect_candle(df_a1, df_a2, idx=50, asset_names=ASSET_NAMES)
    # inspect_candle(df_a1, df_a2, idx=100, asset_names=ASSET_NAMES)
