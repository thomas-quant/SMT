# SMT Helper Module

A Python toolkit for detecting and tracking **Smart Money Technique (SMT) Divergences** between correlated assets. SMT divergences occur when two correlated instruments (e.g., ES/NQ futures) show conflicting price action at key levels—one sweeps a level while the other fails to do so.

## Overview

This module provides a complete lifecycle management system for SMT signals:

1. **Detection** — Identify SMT divergence patterns in real-time
2. **Registration** — Track signals with unique IDs and metadata
3. **Invalidation** — Monitor when SMT signals are broken by price action

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Detector   │ ──▶ │  Registry   │ ──▶ │   Break     │ ──▶ │  Registry   │
│  (detect)   │     │  (store)    │     │  (monitor)  │     │  (update)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
     Micro/Swing/FVG    status="active"     check levels     status="broken"
```

## Features

### SMT Detection Types

| Type | Description | Pattern |
|------|-------------|---------|
| **Micro SMT** | 2-candle divergence | Current candle sweeps previous candle's high/low on one asset, other asset fails |
| **Swing SMT** | Swing point divergence | Price sweeps a swing high/low on one asset while the other asset holds |
| **FVG SMT** | Fair Value Gap divergence | Price enters an FVG zone on one asset while the other asset fails to reach it |

### Core Capabilities

- ✅ Stateless detection functions (no side effects)
- ✅ UUID-based signal tracking
- ✅ Automatic invalidation detection
- ✅ Timeframe-aware registry
- ✅ Real-time candle-by-candle processing
- ✅ Visualization support for detected signals

## Installation

```bash
# Clone or copy the module to your project
git clone <your-repo-url>

# Install dependencies
pip install pandas numpy matplotlib
```

### Dependencies

- `pandas` — DataFrame operations for OHLC data
- `numpy` — Numerical operations (optional, for testing)
- `matplotlib` — Signal visualization (optional)

## Project Structure

```
SMT Helper Module/
├── smt_detector.py    # Stateless detection functions
├── smt_break.py       # Invalidation level tracker
├── smt_registry.py    # SMT lifecycle management
├── smt_manager.py     # Integration orchestrator
├── README.md
└── Sample/
    ├── test.py        # Testing script with visualizations
    ├── NQ_5min.csv    # Sample NQ data
    ├── ES_5min.csv    # Sample ES data
    └── output/        # Generated visualization images
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from smt_manager import SMTManager

# Initialize manager
manager = SMTManager(
    timeframe="5m",
    asset_names=("ES", "NQ"),
    lookback_period=20,
    enable_micro=True,
    enable_swing=True,
    enable_fvg=True
)

# Process each candle
for i in range(start_idx, len(df_es)):
    df_es_slice = df_es.iloc[:i+1]
    df_nq_slice = df_nq.iloc[:i+1]
    candle = df_es.iloc[i]
    
    result = manager.update(
        df_es_slice, df_nq_slice,
        high=candle['High'],
        low=candle['Low'],
        close=candle['Close'],
        ts=df_es.index[i]
    )
    
    # Handle new SMTs
    for smt in result['new_smts']:
        print(f"New SMT: {smt['signal']['signal_type']}")
        print(f"  Invalidation level: {smt['signal']['invalidation_level']}")
    
    # Handle broken SMTs
    for smt in result['broken_smts']:
        print(f"SMT Invalidated: {smt['id']}")

# Query registry
active = manager.get_active_smts()
broken = manager.get_broken_smts()
```

### Using Individual Components

```python
from smt_detector import check_micro_smt, check_swing_smt, check_fvg_smt
from smt_break import SMTBreak
from smt_registry import SMTRegistry

# Detection only
signal = check_micro_smt(df_es, df_nq, asset_names=("ES", "NQ"))
if signal:
    print(signal)
    # {
    #     "signal_type": "Bullish Micro SMT",
    #     "timestamp": Timestamp('2024-01-01 09:35:00'),
    #     "sweeping_asset": "ES",
    #     "failing_asset": "NQ",
    #     "reference_price": 4500.25,
    #     "invalidation_level": 4501.50
    # }

# Manual registry management
registry = SMTRegistry()
smt_id = registry.add_smt(signal, timeframe="5m")
registry.mark_broken(smt_id, broken_ts=current_timestamp)

# Manual break tracking
tracker = SMTBreak()
tracker.add(signal, id=smt_id)
broken_list = tracker.update_candle(high, low, close, ts)
```

## API Reference

### SMTManager

The main orchestrator for the complete lifecycle.

```python
SMTManager(
    timeframe: str,              # e.g., "1m", "5m", "1h"
    asset_names: Tuple[str, str] = ("A1", "A2"),
    lookback_period: int = 20,
    enable_micro: bool = True,
    enable_swing: bool = True,
    enable_fvg: bool = True
)
```

| Method | Description |
|--------|-------------|
| `update(df_a1, df_a2, high, low, close, ts)` | Process a candle, returns `{"new_smts": [...], "broken_smts": [...]}` |
| `get_active_smts()` | Get all active SMTs |
| `get_broken_smts()` | Get all invalidated SMTs |
| `get_all_smts()` | Get all SMTs regardless of status |
| `clear()` | Reset all state |

### SMTRegistry

Stores and manages SMT lifecycle state.

```python
registry = SMTRegistry()
```

| Method | Description |
|--------|-------------|
| `add_smt(signal, timeframe)` | Register new SMT, returns UUID |
| `mark_broken(smt_id, broken_ts)` | Mark SMT as invalidated |
| `get_smt(smt_id)` | Get specific SMT by ID |
| `get_active_smts()` | Get all active SMTs |
| `get_broken_smts()` | Get all broken SMTs |
| `get_smts_by_timeframe(tf)` | Filter by timeframe |

### SMT Entry Shape

```python
{
    "id": "uuid-string",
    "status": "active" | "broken",
    "timeframe": "5m",
    "created_ts": Timestamp,
    "broken_ts": Timestamp | None,
    "signal": {
        "signal_type": "Bullish Micro SMT",
        "timestamp": Timestamp,
        "sweeping_asset": "ES",
        "failing_asset": "NQ",
        "reference_price": 4500.25,
        "invalidation_level": 4501.50
    }
}
```

### Detection Functions

All detection functions are stateless and return `None` if no signal is detected.

```python
# Micro SMT (2-candle pattern)
check_micro_smt(df_a1, df_a2, asset_names=("A1", "A2")) -> Optional[Dict]

# Swing SMT (swing point divergence)
check_swing_smt(df_a1, df_a2, lookback_period, asset_names, timestamp_tolerance=1) -> Optional[Dict]

# FVG SMT (fair value gap divergence)
check_fvg_smt(df_a1, df_a2, lookback_period, asset_names) -> Optional[Dict]
```

### SMTBreak

Tracks invalidation levels and detects when price crosses them.

```python
tracker = SMTBreak()
tracker.add(signal, id=optional_id)  # Register level
tracker.update_candle(high, low, close, ts)  # Returns list of broken levels
tracker.remove(id)  # Remove specific level
tracker.clear()  # Clear all levels
```

## Data Requirements

DataFrames must have:
- **Index**: Timestamps (datetime)
- **Columns**: `Open`, `High`, `Low`, `Close`

Both asset DataFrames must be time-aligned (same timestamps at each index position).

```python
# Example DataFrame structure
                          Open     High      Low    Close
2024-01-01 09:30:00   4500.00  4502.50  4498.25  4501.75
2024-01-01 09:35:00   4501.75  4505.00  4500.50  4504.25
...
```

## Running Tests

```bash
cd "SMT Helper Module/Sample"
python test.py
```

The test script will:
1. Load sample NQ/ES 5-minute data
2. Process candles through the SMTManager
3. Generate visualizations in `output/` folder
4. Print summary statistics

### Sample Output

```
======================================================================
SMT TEST using SMTManager
======================================================================
Assets: NQ/ES | Timeframe: 5m
Candles: 34-500
...

[NEW SMT] Candle #42/500: Bullish Micro SMT
  ID: a1b2c3d4...
  Status: active
  Invalidation: 15025.50

[BROKEN SMT] Candle #48/500: Bullish Micro SMT
  ID: a1b2c3d4...
  Status: broken
  Created: 2024-01-01 09:30:00
  Broken: 2024-01-01 10:00:00

======================================================================
FINAL REGISTRY STATE
======================================================================
Total SMTs detected: 40
  Active: 12
  Broken: 28

Invalidation Rate: 28/40 = 70.0%
```

## Visualization

The test script generates charts for each detected signal:

| Folder | Contents |
|--------|----------|
| `output/micro_smt/` | Micro SMT signal charts |
| `output/swing_smt/` | Swing SMT signal charts |
| `output/fvg_smt/` | FVG SMT signal charts |
| `output/smt_break/` | Invalidation event charts |

## Design Principles

1. **Stateless Detection** — Detector functions have no side effects
2. **Single Source of Truth** — Registry owns all SMT state
3. **Explicit Lifecycle** — Clear transitions: `active` → `broken`
4. **Separation of Concerns** — Detection, storage, and tracking are independent
5. **Easy to Extend** — Ready for strength scoring, decay logic, confluence analysis

## Future Roadmap

This MVP focuses on correctness and clean state management. Potential extensions:

- [ ] Signal strength scoring
- [ ] Time decay weighting
- [ ] Multi-timeframe confluence
- [ ] Custom detector plugins
- [ ] Event callbacks/hooks
- [ ] Persistence layer
- [ ] Real-time streaming support

## License

MIT License — Free to use, modify, and distribute.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing patterns
- No premature abstractions
- Tests pass before submitting

---

**Note**: This module is designed for educational and research purposes. Always validate signals with your own analysis before making trading decisions.

