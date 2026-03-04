# SMT Divergence Toolkit

A Python package for detecting and tracking **Smart Money Technique (SMT) Divergences** between correlated assets. SMT divergences occur when two correlated instruments (e.g. ES/NQ futures) show conflicting price action at a key level — one asset sweeps the level while the other fails to reach it.

## Overview

Complete lifecycle management for SMT signals:

1. **Detection** — Identify divergence patterns in real-time
2. **Registration** — Track signals with unique IDs and metadata
3. **Invalidation** — Monitor when signals are broken by price action

```
detector.py  →  SMTManager.update()  →  registry.py  (status: active)
                                     →  break_tracker.py
                                              │ level crossed?
                                              ↓
                                         registry.py  (status: broken)
```

## Installation

```bash
git clone https://github.com/thomas-quant/SMT.git
cd SMT

python -m venv .venv && source .venv/bin/activate
pip install -e .          # core (numpy, pandas)
pip install -e ".[dev]"   # adds matplotlib
```

### Requirements

- Python >= 3.9
- `pandas >= 1.5`
- `numpy >= 1.23`
- `matplotlib` (optional, for visualization)

## Project Structure

```
SMT/
├── smt/
│   ├── __init__.py        # Public API
│   ├── manager.py         # SMTManager — integration orchestrator
│   ├── detector.py        # Stateless detection functions
│   ├── break_tracker.py   # Invalidation level tracker
│   └── registry.py        # SMT lifecycle / in-memory store
├── pyproject.toml
└── README.md
```

## Quick Start

```python
import pandas as pd
from smt import SMTManager

manager = SMTManager(
    timeframe="5m",
    asset_names=("ES", "NQ"),
    lookback_period=20,
    enable_micro=True,
    enable_swing=True,
    enable_fvg=True,
)

# Call on every candle close — DataFrames must share a datetime index
for i in range(start_idx, len(df_es)):
    result = manager.update(
        df_es.iloc[:i+1],
        df_nq.iloc[:i+1],
        high=df_es.iloc[i]["High"],
        low=df_es.iloc[i]["Low"],
        close=df_es.iloc[i]["Close"],
        ts=df_es.index[i],
    )

    for smt in result["new_smts"]:
        print(f"New SMT: {smt['signal']['signal_type']}")
        print(f"  Invalidation: {smt['signal']['invalidation_level']}")

    for smt in result["broken_smts"]:
        print(f"SMT Invalidated: {smt['id']}")
```

## Detection Types

| Type | Function | Pattern |
|------|----------|---------|
| **Micro** | `check_micro_smt` | Current candle sweeps previous candle's H/L; other asset doesn't |
| **Swing** | `check_swing_smt` | Sweeps a recent swing high/low; other asset holds its swing |
| **FVG** | `check_fvg_smt` | Enters an FVG zone on one asset; other asset fails to reach its FVG |

All three functions are **stateless** — they take DataFrames and return a signal dict or `None`.

## Using Individual Components

```python
from smt import SMTManager, SMTBreak, SMTRegistry
from smt.detector import check_micro_smt, check_swing_smt, check_fvg_smt

# Detection only
signal = check_micro_smt(df_es, df_nq, asset_names=("ES", "NQ"))

# Manual registry
registry = SMTRegistry()
smt_id = registry.add_smt(signal, timeframe="5m")
registry.mark_broken(smt_id, broken_ts=ts)

# Manual break tracking
tracker = SMTBreak()
tracker.add(signal, id=smt_id)
broken_list = tracker.update_candle(high, low, close, ts)
```

## API Reference

### SMTManager

```python
SMTManager(
    timeframe: str,                    # e.g. "1m", "5m", "1h"
    asset_names: Tuple[str, str] = ("A1", "A2"),
    lookback_period: int = 20,
    enable_micro: bool = True,
    enable_swing: bool = True,
    enable_fvg: bool = True,
)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `update(df_a1, df_a2, high, low, close, ts)` | `{"new_smts": [...], "broken_smts": [...]}` | Process one candle |
| `get_active_smts()` | list | All active SMTs |
| `get_broken_smts()` | list | All invalidated SMTs |
| `get_all_smts()` | list | All SMTs |
| `clear()` | — | Reset all state |

### SMTRegistry

| Method | Description |
|--------|-------------|
| `add_smt(signal, timeframe)` | Register a new SMT, returns UUID |
| `mark_broken(smt_id, broken_ts)` | Mark as invalidated |
| `get_smt(smt_id)` | Fetch by ID |
| `get_active_smts()` | Filter by status |
| `get_broken_smts()` | Filter by status |
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
        "invalidation_level": 4501.50,
    }
}
```

### Detection Functions

```python
check_micro_smt(df_a1, df_a2, asset_names=("A1", "A2")) -> Optional[Dict]
check_swing_smt(df_a1, df_a2, lookback_period, asset_names, timestamp_tolerance=1) -> Optional[Dict]
check_fvg_smt(df_a1, df_a2, lookback_period, asset_names) -> Optional[Dict]
```

### SMTBreak

```python
tracker = SMTBreak()
tracker.add(signal, id=optional_id)          # Register invalidation level
tracker.update_candle(high, low, close, ts)  # Returns list of broken IDs
tracker.remove(id)
tracker.clear()
```

## Data Requirements

Both DataFrames must have:
- **Index**: datetime timestamps (aligned — `df_a1.index[-1] == df_a2.index[-1]` on every call)
- **Columns**: `Open`, `High`, `Low`, `Close`

## Design Principles

- **Stateless detection** — no side effects in detector functions
- **Single source of truth** — registry owns all SMT state
- **Explicit lifecycle** — clear `active` → `broken` transition
- **Separation of concerns** — detection, storage, and tracking are independent

## License

MIT License — free to use, modify, and distribute.
