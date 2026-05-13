# SMT Divergence Toolkit

Python toolkit for detecting **Smart Money Technique (SMT) divergence**
between two correlated assets, such as ES/NQ futures. It supports
candle-by-candle signal tracking plus vectorized historical scans for research
and backtesting.

SMT divergence occurs when one asset sweeps a reference high/low or
fair-value-gap level while the paired asset fails to confirm the move.

## Features

- Detects micro, swing, and fair-value-gap (FVG) SMT setups.
- Tracks live signals from creation to invalidation with `SMTManager`.
- Scans aligned historical OHLC data into a backtest-ready event table.
- Keeps detector functions stateless for direct research use.
- Validates OHLC columns, datetime indexes, sorted unique indexes, and paired-asset alignment.

## Installation

```bash
git clone https://github.com/thomas-quant/SMT.git
cd SMT

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
pip install -e ".[dev]"  # pytest + matplotlib
```

Requirements: Python 3.9+, pandas 1.5+, numpy 1.23+.

## Data Requirements

Both input DataFrames must:

- use a sorted, unique `DatetimeIndex`;
- share the same index;
- include `Open`, `High`, `Low`, and `Close` columns.

## Quick Start

### Live / Sequential Processing

Use `SMTManager` when candles arrive incrementally.

```python
from smt import SMTManager

manager = SMTManager(
    timeframe="5m",
    asset_names=("ES", "NQ"),
    lookback_period=20,
    enable_micro=True,
    enable_swing=True,
    enable_fvg=True,
)

for end in range(2, len(df_es) + 1):
    result = manager.update(df_es.iloc[:end], df_nq.iloc[:end])

    for smt in result["new_smts"]:
        signal = smt["signal"]
        print(f"New {signal['signal_type']} on {signal['timestamp']}")
        print(
            f"Invalidation: {signal['invalidation_asset']} "
            f"{signal['invalidation_direction']} {signal['invalidation_level']}"
        )

    for smt in result["broken_smts"]:
        print(f"Broken SMT: {smt['id']} at {smt['broken_ts']}")
```

### Historical / Backtesting

Use `scan_smts_historical` when full aligned history is already available.

```python
from smt import scan_smts_historical

events = scan_smts_historical(
    df_es,
    df_nq,
    asset_names=("ES", "NQ"),
    lookback_period=20,
    enable_micro=True,
    enable_swing=True,
    enable_fvg=True,
)

active = events[events["status"] == "active"]
broken = events[events["status"] == "broken"]
```

## Detection Types

| Type | Function | Pattern |
| --- | --- | --- |
| Micro | `check_micro_smt` | Sweeps previous candle high/low. |
| Swing | `check_swing_smt` | Sweeps recent swing high/low. |
| FVG | `check_fvg_smt` | Enters one FVG while pair fails. |

Detector functions return a signal `dict` or `None`.

## Event Output

`scan_smts_historical` returns one row per event:

```text
signal_type
created_ts
reference_timestamp
sweeping_asset
failing_asset
reference_price
invalidation_asset
invalidation_direction
invalidation_level
broken_ts
status
```

`status` is `broken` when a later candle crosses the invalidation level on
`invalidation_asset`; otherwise it remains `active`.

## Public API

```python
from smt import (
    SMTBreak,
    SMTManager,
    SMTRegistry,
    check_fvg_smt,
    check_micro_smt,
    check_swing_smt,
    scan_smts_historical,
)
```

### `SMTManager`

```python
SMTManager(
    timeframe: str,
    asset_names: tuple[str, str] = ("A1", "A2"),
    lookback_period: int = 20,
    enable_micro: bool = True,
    enable_swing: bool = True,
    enable_fvg: bool = True,
)
```

| Method | Description |
| --- | --- |
| `update(df_a1, df_a2)` | Detect new SMTs and mark invalidated active SMTs. |
| `get_active_smts()` | Return active SMTs keyed by UUID. |
| `get_broken_smts()` | Return invalidated SMTs keyed by UUID. |
| `get_all_smts()` | Return all tracked SMTs keyed by UUID. |
| `clear()` | Reset registry and invalidation tracker. |

## Project Layout

```text
SMT/
├── smt/
│   ├── __init__.py        # Public API
│   ├── break_tracker.py   # Invalidation-level tracking
│   ├── detector.py        # Stateless SMT detectors
│   ├── historical.py      # Historical event scanner
│   ├── manager.py         # Live lifecycle orchestrator
│   └── registry.py        # In-memory signal registry
├── tests/
├── pyproject.toml
└── README.md
```

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python3 -m pytest -q
```

Do not install dependencies into system Python.
