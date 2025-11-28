"""SMT Detector Testing Script - Loads CSV data and tests SMT detector across candles."""

import pandas as pd
import smt_detector as smt
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


def load_data(file_a1: str, file_a2: str):
    """Load CSV data for both assets and return time-indexed DataFrames."""
    df_a1 = pd.read_csv(file_a1)
    df_a2 = pd.read_csv(file_a2)
    
    time_cols = ['timestamp', 'Timestamp', 'datetime', 'Datetime', 'Date', 'date', 'time', 'Time']
    time_col = next((col for col in time_cols if col in df_a1.columns), None)
    
    if time_col:
        df_a1[time_col] = pd.to_datetime(df_a1[time_col])
        df_a2[time_col] = pd.to_datetime(df_a2[time_col])
        df_a1.set_index(time_col, inplace=True)
        df_a2.set_index(time_col, inplace=True)
    elif df_a1.columns[0] not in ['Open', 'High', 'Low', 'Close']:
        df_a1.set_index(df_a1.columns[0], inplace=True)
        df_a2.set_index(df_a2.columns[0], inplace=True)
    
    for col in ['Open', 'High', 'Low', 'Close']:
        if col not in df_a1.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df_a1, df_a2


def print_signal(signal: dict, candle_idx: int, total_candles: int):
    """Pretty print a detected signal."""
    print(f"\n{'='*60}")
    print(f"SIGNAL at Candle #{candle_idx + 1}/{total_candles}: {signal['signal_type']}")
    print(f"Time: {signal['timestamp']}" + (f" | Ref: {signal['reference_timestamp']}" if 'reference_timestamp' in signal else ""))
    print(f"Sweep: {signal['sweeping_asset']} | Fail: {signal['failing_asset']}")
    print(f"Ref Price: {signal['reference_price']:.2f} | Invalidation: {signal['invalidation_level']:.2f}")


def test_smt_detector(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names: tuple = ("NQ", "ES"),
    lookback_swing: int = 3,
    lookback_fvg: int = 10,
    start_idx: int = None,
    visualize_signals: bool = True
):
    """Test SMT detector by iterating through data candle by candle."""
    min_start = max(lookback_swing, lookback_fvg) + 3
    start_idx = max(start_idx, min_start) if start_idx else min_start
    total_candles = len(df_a1)
    
    print(f"\nSMT TEST: {asset_names[0]}/{asset_names[1]} | Candles: {start_idx+1}-{total_candles} | Lookback: swing={lookback_swing}, fvg={lookback_fvg}")
    
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
            if visualize_signals:
                try:
                    visualize_micro_smt(
                        signal, df_a1, df_a2, asset_names,
                        candles_before=15, candles_after=5,
                        save_path=f"output/micro_smt/micro_smt_signal_{micro_count}.png"
                    )
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")
        # Test Swing SMT
        signal = smt.check_swing_smt(
            current_a1, current_a2, lookback_swing, asset_names
        )
        if signal:
            print_signal(signal, i, total_candles)
            swing_count += 1

            if visualize_signals:
                try:
                    visualize_swing_smt(
                        signal, df_a1, df_a2, asset_names, lookback_swing,
                        candles_before=15, candles_after=5,
                        save_path=f"output/swing_smt/swing_smt_signal_{swing_count}.png"
                    )
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")
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
                        save_path=f"output/fvg_smt/fvg_smt_signal_{fvg_count}.png"
                    )
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}")
    
    print(f"\nSUMMARY: Micro={micro_count}, Swing={swing_count}, FVG={fvg_count}, Total={micro_count + swing_count + fvg_count}")


def visualize_fvg_smt(
    signal: dict, df_a1: pd.DataFrame, df_a2: pd.DataFrame, asset_names: tuple,
    lookback_fvg: int, candles_before: int = 10, candles_after: int = 5, save_path: str = None
):
    """Create a visual chart showing the FVG SMT signal."""
    if signal['signal_type'] not in ['Bullish FVG SMT', 'Bearish FVG SMT']:
        return
    
    signal_timestamp, ref_timestamp = signal['timestamp'], signal['reference_timestamp']
    try:
        signal_idx = df_a1.index.get_loc(signal_timestamp)
        ref_idx = df_a1.index.get_loc(ref_timestamp)
        signal_idx = signal_idx.start if isinstance(signal_idx, slice) else signal_idx
        ref_idx = ref_idx.start if isinstance(ref_idx, slice) else ref_idx
    except KeyError:
        return
    
    fvg_a1 = smt._find_recent_valid_fvg(df_a1.iloc[:signal_idx+1], lookback_fvg)
    fvg_a2 = smt._find_recent_valid_fvg(df_a2.iloc[:signal_idx+1], lookback_fvg)
    if fvg_a1 is None or fvg_a2 is None:
        return
    
    start_idx = max(0, ref_idx - candles_before)
    end_idx = min(len(df_a1), signal_idx + candles_after + 1)
    window_a1, window_a2 = df_a1.iloc[start_idx:end_idx], df_a2.iloc[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{signal['signal_type']} - {signal_timestamp}", fontsize=16, fontweight='bold')
    
    for ax, window_df, asset_name, fvg_info in [(ax1, window_a1, asset_names[0], fvg_a1), (ax2, window_a2, asset_names[1], fvg_a2)]:
        _plot_candlesticks(ax, window_df)
        if ref_timestamp in window_df.index:
            try:
                fvg_idx = window_df.index.get_loc(ref_timestamp)
                _highlight_fvg(ax, window_df, fvg_info, fvg_idx.start if isinstance(fvg_idx, slice) else fvg_idx)
            except (KeyError, TypeError):
                pass
        if signal_timestamp in window_df.index:
            try:
                sig_idx = window_df.index.get_loc(signal_timestamp)
                _mark_signal_candle(ax, window_df, signal, sig_idx.start if isinstance(sig_idx, slice) else sig_idx, asset_name)
            except (KeyError, TypeError):
                pass
        
        if asset_name == signal['sweeping_asset']:
            ax.axhline(y=signal['reference_price'], color='red', linestyle='--', linewidth=2, label=f"Ref: {signal['reference_price']:.2f}")
        else:
            ax.axhline(y=signal['invalidation_level'], color='orange', linestyle='--', linewidth=2, label=f"Inv: {signal['invalidation_level']:.2f}")
        ax.set_ylabel(f'{asset_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time')
    fig.text(0.02, 0.02, f"Sweep: {signal['sweeping_asset']} | Fail: {signal['failing_asset']} | FVG: {fvg_a1['type'].title()}", 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def visualize_swing_smt(
    signal: dict, df_a1: pd.DataFrame, df_a2: pd.DataFrame, asset_names: tuple,
    lookback_swing: int, candles_before: int = 10, candles_after: int = 5, save_path: str = None
):
    """Create a visual chart showing the Swing SMT signal."""
    if signal['signal_type'] not in ['Bullish Swing SMT', 'Bearish Swing SMT']:
        return
    
    signal_timestamp, ref_timestamp = signal['timestamp'], signal['reference_timestamp']
    try:
        signal_idx = df_a1.index.get_loc(signal_timestamp)
        ref_idx = df_a1.index.get_loc(ref_timestamp)
        signal_idx = signal_idx.start if isinstance(signal_idx, slice) else signal_idx
        ref_idx = ref_idx.start if isinstance(ref_idx, slice) else ref_idx
    except KeyError:
        return
    
    start_idx = max(0, ref_idx - candles_before)
    end_idx = min(len(df_a1), signal_idx + candles_after + 1)
    window_a1, window_a2 = df_a1.iloc[start_idx:end_idx], df_a2.iloc[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{signal['signal_type']} - {signal_timestamp}", fontsize=16, fontweight='bold')
    is_bullish = 'Bullish' in signal['signal_type']
    
    for ax, window_df, asset_name in [(ax1, window_a1, asset_names[0]), (ax2, window_a2, asset_names[1])]:
        _plot_candlesticks(ax, window_df)
        if ref_timestamp in window_df.index:
            try:
                sw_idx = window_df.index.get_loc(ref_timestamp)
                _highlight_swing_point(ax, window_df, signal, sw_idx.start if isinstance(sw_idx, slice) else sw_idx, asset_name, is_bullish)
            except (KeyError, TypeError):
                pass
        if signal_timestamp in window_df.index:
            try:
                sig_idx = window_df.index.get_loc(signal_timestamp)
                _mark_signal_candle(ax, window_df, signal, sig_idx.start if isinstance(sig_idx, slice) else sig_idx, asset_name)
            except (KeyError, TypeError):
                pass
        
        if asset_name == signal['sweeping_asset']:
            ax.axhline(y=signal['reference_price'], color='red', linestyle='--', linewidth=2, label=f"Ref: {signal['reference_price']:.2f}")
        else:
            ax.axhline(y=signal['invalidation_level'], color='orange', linestyle='--', linewidth=2, label=f"Inv: {signal['invalidation_level']:.2f}")
        ax.set_ylabel(f'{asset_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time')
    swing_type = "Swing Low" if is_bullish else "Swing High"
    fig.text(0.02, 0.02, f"Sweep: {signal['sweeping_asset']} | Fail: {signal['failing_asset']} | {swing_type}",
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def visualize_micro_smt(
    signal: dict, df_a1: pd.DataFrame, df_a2: pd.DataFrame, asset_names: tuple,
    candles_before: int = 10, candles_after: int = 5, save_path: str = None
):
    """Create a visual chart showing the Micro SMT signal."""
    if signal['signal_type'] not in ['Bullish Micro SMT', 'Bearish Micro SMT']:
        return
    
    signal_timestamp = signal['timestamp']
    try:
        signal_idx = df_a1.index.get_loc(signal_timestamp)
        signal_idx = signal_idx.start if isinstance(signal_idx, slice) else signal_idx
    except KeyError:
        return
    
    if signal_idx < 1:
        return
    
    prev_idx = signal_idx - 1
    prev_timestamp = df_a1.index[prev_idx]
    start_idx = max(0, prev_idx - candles_before)
    end_idx = min(len(df_a1), signal_idx + candles_after + 1)
    window_a1, window_a2 = df_a1.iloc[start_idx:end_idx], df_a2.iloc[start_idx:end_idx]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{signal['signal_type']} - {signal_timestamp}", fontsize=16, fontweight='bold')
    is_bullish = 'Bullish' in signal['signal_type']
    
    for ax, window_df, asset_name in [(ax1, window_a1, asset_names[0]), (ax2, window_a2, asset_names[1])]:
        _plot_candlesticks(ax, window_df)
        if prev_timestamp in window_df.index:
            try:
                prev_win_idx = window_df.index.get_loc(prev_timestamp)
                _highlight_reference_candle(ax, window_df, prev_win_idx.start if isinstance(prev_win_idx, slice) else prev_win_idx, asset_name, is_bullish)
            except (KeyError, TypeError):
                pass
        if signal_timestamp in window_df.index:
            try:
                sig_idx = window_df.index.get_loc(signal_timestamp)
                _mark_signal_candle(ax, window_df, signal, sig_idx.start if isinstance(sig_idx, slice) else sig_idx, asset_name)
            except (KeyError, TypeError):
                pass
        
        if asset_name == signal['sweeping_asset']:
            ax.axhline(y=signal['reference_price'], color='red', linestyle='--', linewidth=2, label=f"Ref: {signal['reference_price']:.2f}")
        else:
            ax.axhline(y=signal['invalidation_level'], color='orange', linestyle='--', linewidth=2, label=f"Inv: {signal['invalidation_level']:.2f}")
        ax.set_ylabel(f'{asset_name}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time')
    fig.text(0.02, 0.02, f"Sweep: {signal['sweeping_asset']} | Fail: {signal['failing_asset']} | 2-Candle Micro",
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def _save_or_show(fig, save_path):
    """Save figure to path or display interactively."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _plot_candlesticks(ax, df: pd.DataFrame):
    """Plot candlesticks on axis."""
    for idx, (_, row) in enumerate(df.iterrows()):
        color = 'green' if row['Close'] >= row['Open'] else 'red'
        ax.plot([idx, idx], [row['Low'], row['High']], color='black', linewidth=1)
        body_bottom = min(row['Open'], row['Close'])
        rect = Rectangle((idx - 0.3, body_bottom), 0.6, abs(row['Close'] - row['Open']),
                         facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax.add_patch(rect)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([ts.strftime('%m/%d %H:%M') if hasattr(ts, 'strftime') else str(ts) 
                        for ts in df.index], rotation=45, ha='right')


def _highlight_fvg(ax, df: pd.DataFrame, fvg_info: dict, fvg_start_idx: int):
    """Highlight the FVG zone."""
    fvg_bottom, fvg_top = fvg_info['bottom'], fvg_info['top']
    rect = Rectangle((fvg_start_idx - 0.5, fvg_bottom), len(df) - fvg_start_idx + 0.5, fvg_top - fvg_bottom,
                     facecolor='yellow', edgecolor='orange', linewidth=2, alpha=0.15, linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.axhline(y=fvg_bottom, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
    ax.axhline(y=fvg_top, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, zorder=1)
    ax.text(fvg_start_idx + 1, (fvg_bottom + fvg_top) / 2, f"FVG {fvg_info['type'].upper()}\n[{fvg_bottom:.2f} - {fvg_top:.2f}]",
            fontsize=9, ha='left', va='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8, edgecolor='orange'), zorder=2)


def _mark_signal_candle(ax, df: pd.DataFrame, signal: dict, signal_idx: int, asset_name: str):
    """Mark the signal candle with a circle."""
    row = df.iloc[signal_idx]
    is_sweep = asset_name == signal['sweeping_asset']
    color = 'red' if is_sweep else 'orange'
    label = 'SIGNAL\n(Sweeps)' if is_sweep else 'SIGNAL\n(Fails)'
    ax.add_patch(plt.Circle((signal_idx, row['Close']), 0.5, color=color, fill=False, linewidth=3))
    ax.text(signal_idx, row['High'] + (row['High'] - row['Low']) * 0.1, label, fontsize=9, ha='center', color=color, fontweight='bold')


def _highlight_swing_point(ax, df: pd.DataFrame, signal: dict, swing_idx: int, asset_name: str, is_bullish: bool):
    """Highlight the swing point with a diamond marker."""
    row = df.iloc[swing_idx]
    swing_price = row['Low'] if is_bullish else row['High']
    label, color = ("SWING LOW", 'blue') if is_bullish else ("SWING HIGH", 'purple')
    ax.plot(swing_idx, swing_price, marker='D', markersize=12, color=color, markeredgecolor='black', markeredgewidth=2, zorder=5, label=label)
    offset = (row['High'] - row['Low']) * 0.15
    text_y, va = (swing_price - offset, 'top') if is_bullish else (swing_price + offset, 'bottom')
    ax.text(swing_idx, text_y, f"{label}\n{swing_price:.2f}", fontsize=9, ha='center', va=va, color=color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color), zorder=6)


def _highlight_reference_candle(ax, df: pd.DataFrame, ref_idx: int, asset_name: str, is_bullish: bool):
    """Highlight the reference candle (previous candle in Micro SMT)."""
    row = df.iloc[ref_idx]
    highlight_price = row['Low'] if is_bullish else row['High']
    color = 'blue' if is_bullish else 'purple'
    label = "REF (Low)" if is_bullish else "REF (High)"
    
    rect = Rectangle((ref_idx - 0.4, row['Low']), 0.8, row['High'] - row['Low'],
                     facecolor='none', edgecolor=color, linewidth=2.5, linestyle='--', alpha=0.8, zorder=4)
    ax.add_patch(rect)
    ax.plot(ref_idx, highlight_price, marker='o', markersize=10, color=color, markeredgecolor='black', markeredgewidth=2, zorder=5)
    offset = (row['High'] - row['Low']) * 0.15
    text_y, va = (highlight_price - offset, 'top') if is_bullish else (highlight_price + offset, 'bottom')
    ax.text(ref_idx, text_y, f"{label}\n{highlight_price:.2f}", fontsize=9, ha='center', va=va, color=color, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color), zorder=6)


if __name__ == "__main__":
    # Configuration
    FILE_A1, FILE_A2 = "NQ_5min.csv", "ES_5min.csv"
    ASSET_NAMES = ("NQ", "ES")
    LOOKBACK_SWING, LOOKBACK_FVG = 30, 30
    VISUALIZE_SIGNALS = True
    
    print("Loading data...")
    try:
        df_a1, df_a2 = load_data(FILE_A1, FILE_A2)
        print(f"Loaded {len(df_a1)} candles for {ASSET_NAMES[0]}, {len(df_a2)} for {ASSET_NAMES[1]}")
        if len(df_a1) != len(df_a2):
            print(f"WARNING: Length mismatch!")
    except FileNotFoundError:
        print(f"ERROR: CSV files not found ({FILE_A1}, {FILE_A2})")
        exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        exit(1)
    
    test_smt_detector(df_a1, df_a2, ASSET_NAMES, LOOKBACK_SWING, LOOKBACK_FVG, visualize_signals=VISUALIZE_SIGNALS)
