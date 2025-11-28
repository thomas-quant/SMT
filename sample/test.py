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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np


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
    start_idx: int = None,
    visualize_signals: bool = True
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
        
        """# Test Micro SMT
        signal = smt.check_micro_smt(current_a1, current_a2, asset_names)
        if signal:
            print_signal(signal, i, total_candles)
            micro_count += 1"""
        
        """# Test Swing SMT
        signal = smt.check_swing_smt(
            current_a1, current_a2, lookback_swing, asset_names
        )
        if signal:
            print_signal(signal, i, total_candles)
            swing_count += 1"""
        
        # Test FVG SMT
        signal = smt.check_fvg_smt(
            current_a1, current_a2, lookback_fvg, asset_names
        )
        if signal:
            print_signal(signal, i, total_candles)
            fvg_count += 1
            # Visualize the signal if enabled
            if visualize_signals:
                try:
                    visualize_fvg_smt(
                        signal, df_a1, df_a2, asset_names, lookback_fvg,
                        candles_before=15, candles_after=5,
                        save_path=f"fvg_smt_signal_{fvg_count}.png"
                    )
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Micro SMT Signals: {micro_count}")
    print(f"Swing SMT Signals: {swing_count}")
    print(f"FVG SMT Signals:   {fvg_count}")
    print(f"Total Signals:     {micro_count + swing_count + fvg_count}")
    print("="*70 + "\n")


def visualize_fvg_smt(
    signal: dict,
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names: tuple,
    lookback_fvg: int,
    candles_before: int = 10,
    candles_after: int = 5,
    save_path: str = None
):
    """
    Create a visual chart showing the FVG SMT signal.
    
    Args:
        signal: Signal dictionary from FVG SMT detector
        df_a1: Full DataFrame for asset 1
        df_a2: Full DataFrame for asset 2
        asset_names: Tuple of asset names
        lookback_fvg: Lookback period used for FVG detection
        candles_before: Number of candles before FVG to show
        candles_after: Number of candles after signal to show
        save_path: Optional path to save the image (if None, displays interactively)
    """
    if signal['signal_type'] not in ['Bullish FVG SMT', 'Bearish FVG SMT']:
        print("Warning: This function only visualizes FVG SMT signals")
        return
    
    # Find the signal timestamp index
    signal_timestamp = signal['timestamp']
    try:
        signal_idx = df_a1.index.get_loc(signal_timestamp)
        if isinstance(signal_idx, slice):
            signal_idx = signal_idx.start
    except KeyError:
        print(f"Warning: Signal timestamp {signal_timestamp} not found in data")
        return
    
    # Find the FVG reference timestamp index
    ref_timestamp = signal['reference_timestamp']
    try:
        ref_idx = df_a1.index.get_loc(ref_timestamp)
        if isinstance(ref_idx, slice):
            ref_idx = ref_idx.start
    except KeyError:
        print(f"Warning: Reference timestamp {ref_timestamp} not found in data")
        return
    
    # Get the data slice that was used for detection (up to signal candle)
    detection_data_a1 = df_a1.iloc[:signal_idx+1]
    detection_data_a2 = df_a2.iloc[:signal_idx+1]
    
    # Find the FVG details
    fvg_a1 = smt._find_recent_fvg(detection_data_a1, lookback_fvg)
    fvg_a2 = smt._find_recent_fvg(detection_data_a2, lookback_fvg)
    
    if fvg_a1 is None or fvg_a2 is None:
        print("Warning: Could not find FVG details for visualization")
        return
    
    # Determine the window to show
    # Show from a few candles before FVG formation to a few after signal
    fvg_start_idx = max(0, ref_idx - candles_before)
    window_end_idx = min(len(df_a1), signal_idx + candles_after + 1)
    
    # Get the data window
    window_a1 = df_a1.iloc[fvg_start_idx:window_end_idx]
    window_a2 = df_a2.iloc[fvg_start_idx:window_end_idx]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{signal['signal_type']} - {signal_timestamp}", 
                 fontsize=16, fontweight='bold')
    
    # Plot both assets
    for ax, window_df, asset_name, fvg_info in [
        (ax1, window_a1, asset_names[0], fvg_a1),
        (ax2, window_a2, asset_names[1], fvg_a2)
    ]:
        # Plot candlesticks
        _plot_candlesticks(ax, window_df, asset_name)
        
        # Highlight FVG zone
        if ref_timestamp in window_df.index:
            try:
                fvg_start_window = window_df.index.get_loc(ref_timestamp)
                if isinstance(fvg_start_window, slice):
                    fvg_start_window = fvg_start_window.start
                _highlight_fvg(ax, window_df, fvg_info, fvg_start_window)
            except (KeyError, TypeError):
                pass
        
        # Mark signal candle
        if signal_timestamp in window_df.index:
            try:
                signal_window_idx = window_df.index.get_loc(signal_timestamp)
                if isinstance(signal_window_idx, slice):
                    signal_window_idx = signal_window_idx.start
                _mark_signal_candle(ax, window_df, signal, signal_window_idx, asset_name)
            except (KeyError, TypeError):
                pass
        
        # Draw reference and invalidation levels
        if asset_name == signal['sweeping_asset']:
            ax.axhline(y=signal['reference_price'], color='red', linestyle='--', 
                      linewidth=2, label=f"Reference: {signal['reference_price']:.2f}")
        else:
            ax.axhline(y=signal['invalidation_level'], color='orange', linestyle='--', 
                      linewidth=2, label=f"Invalidation: {signal['invalidation_level']:.2f}")
        
        ax.set_ylabel(f'{asset_name} Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time', fontsize=12)
    
    # Add annotation
    annotation_text = (
        f"Sweeping Asset: {signal['sweeping_asset']}\n"
        f"Failing Asset: {signal['failing_asset']}\n"
        f"FVG Type: {fvg_a1['type'].title()}"
    )
    fig.text(0.02, 0.02, annotation_text, fontsize=10, 
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
        plt.close(fig)  # Close to free memory
    else:
        plt.show()


def _plot_candlesticks(ax, df: pd.DataFrame, asset_name: str):
    """Helper function to plot candlesticks."""
    for idx, (timestamp, row) in enumerate(df.iterrows()):
        open_price = row['Open']
        high_price = row['High']
        low_price = row['Low']
        close_price = row['Close']
        
        # Determine color
        color = 'green' if close_price >= open_price else 'red'
        
        # Draw the wick
        ax.plot([idx, idx], [low_price, high_price], color='black', linewidth=1)
        
        # Draw the body
        body_height = abs(close_price - open_price)
        body_bottom = min(open_price, close_price)
        
        rect = Rectangle((idx - 0.3, body_bottom), 0.6, body_height,
                        facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
    
    # Set x-axis labels
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([ts.strftime('%m/%d %H:%M') if hasattr(ts, 'strftime') else str(ts) 
                        for ts in df.index], rotation=45, ha='right')


def _highlight_fvg(ax, df: pd.DataFrame, fvg_info: dict, fvg_start_idx: int):
    """Helper function to highlight the FVG zone."""
    # FVG spans 3 candles (C1, C2, C3)
    fvg_end_idx = min(fvg_start_idx + 2, len(df) - 1)
    
    # Draw FVG zone rectangle - extend from FVG start to end of window
    fvg_bottom = fvg_info['bottom']
    fvg_top = fvg_info['top']
    fvg_height = fvg_top - fvg_bottom
    
    # Draw the FVG zone extending to the right (showing it's still active)
    rect = Rectangle((fvg_start_idx - 0.5, fvg_bottom), 
                    len(df) - fvg_start_idx + 0.5, fvg_height,
                    facecolor='yellow', edgecolor='orange', linewidth=2, 
                    alpha=0.15, linestyle='--', zorder=0)
    ax.add_patch(rect)
    
    # Draw horizontal lines for FVG boundaries
    ax.axhline(y=fvg_bottom, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(y=fvg_top, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
    
    # Label the FVG
    mid_x = fvg_start_idx + 1
    mid_y = (fvg_bottom + fvg_top) / 2
    ax.text(mid_x, mid_y, 
           f"FVG {fvg_info['type'].upper()}\n[{fvg_bottom:.2f} - {fvg_top:.2f}]",
           fontsize=9, ha='left', va='center',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='orange'),
           zorder=2)


def _mark_signal_candle(ax, df: pd.DataFrame, signal: dict, signal_idx: int, asset_name: str):
    """Helper function to mark the signal candle."""
    signal_row = df.iloc[signal_idx]
    
    # Draw a circle around the signal candle
    if asset_name == signal['sweeping_asset']:
        # Sweeping asset - mark with red circle
        circle = plt.Circle((signal_idx, signal_row['Close']), 0.5, 
                           color='red', fill=False, linewidth=3)
        ax.add_patch(circle)
        ax.text(signal_idx, signal_row['High'] + (signal_row['High'] - signal_row['Low']) * 0.1,
               'SIGNAL\n(Sweeps)', fontsize=9, ha='center', color='red', fontweight='bold')
    else:
        # Failing asset - mark with orange circle
        circle = plt.Circle((signal_idx, signal_row['Close']), 0.5, 
                           color='orange', fill=False, linewidth=3)
        ax.add_patch(circle)
        ax.text(signal_idx, signal_row['High'] + (signal_row['High'] - signal_row['Low']) * 0.1,
               'SIGNAL\n(Fails)', fontsize=9, ha='center', color='orange', fontweight='bold')


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
    FILE_A1 = "NQ_5min.csv"  # Asset 1 (e.g., Nasdaq futures)
    FILE_A2 = "ES_5min.csv"  # Asset 2 (e.g., S&P 500 futures)
    ASSET_NAMES = ("NQ", "ES")
    
    # Detection parameters
    LOOKBACK_SWING = 30  # Candles to look back for swing highs/lows
    LOOKBACK_FVG = 30    # Candles to look back for FVGs
    
    # Visualization settings
    VISUALIZE_SIGNALS = True  # Set to False to disable chart generation
    
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
        lookback_fvg=LOOKBACK_FVG,
        visualize_signals=VISUALIZE_SIGNALS
    )
    
    # ===== OPTIONAL: INSPECT SPECIFIC CANDLES =====
    # Uncomment to inspect specific candles for debugging
    # inspect_candle(df_a1, df_a2, idx=50, asset_names=ASSET_NAMES)
    # inspect_candle(df_a1, df_a2, idx=100, asset_names=ASSET_NAMES)
