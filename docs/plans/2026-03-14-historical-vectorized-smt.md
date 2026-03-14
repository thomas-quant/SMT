# Historical Vectorized SMT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a historical SMT scanning API that returns a full event table for backtesting, while leaving the existing live/stateful manager workflow unchanged.

**Architecture:** Add a new `smt/historical.py` batch scanner that validates aligned OHLC history, detects micro/swing/FVG SMT creation events across the full dataset, resolves each event's first invalidation timestamp, and returns a canonical event table. Keep `SMTManager`, `SMTRegistry`, and `SMTBreak` unchanged for live use, and document the new historical path in `README.md`.

**Tech Stack:** Python 3.9+, pandas, numpy, pytest

---

### Task 1: Add historical API scaffolding and empty-result contract

**Files:**
- Create: `tests/test_historical.py`
- Create: `smt/historical.py`
- Modify: `smt/__init__.py`

**Step 1: Write the failing test**

```python
import pandas as pd
import pytest

from smt import scan_smts_historical


def _build_df(rows, index=None):
    if index is None:
        index = pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 09:35:00",
                "2024-01-01 09:40:00",
                "2024-01-01 09:45:00",
            ]
        )[: len(rows)]
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])


def test_scan_smts_historical_returns_empty_dataframe_with_expected_columns():
    df_a1 = _build_df(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.5, 101.5, 99.5, 101.0),
        ]
    )
    df_a2 = _build_df(
        [
            (200.0, 201.0, 199.0, 200.0),
            (200.5, 201.5, 199.5, 201.0),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        enable_micro=False,
        enable_swing=False,
        enable_fvg=False,
    )

    assert list(result.columns) == [
        "signal_type",
        "created_ts",
        "reference_timestamp",
        "sweeping_asset",
        "failing_asset",
        "reference_price",
        "invalidation_asset",
        "invalidation_direction",
        "invalidation_level",
        "broken_ts",
        "status",
    ]
    assert result.empty


def test_scan_smts_historical_reuses_detector_validation_rules():
    valid = _build_df(
        [
            (100.0, 101.0, 99.0, 100.0),
            (100.5, 101.5, 99.5, 101.0),
        ]
    )
    invalid = valid.drop(columns=["Low"])

    with pytest.raises(ValueError, match="Open, High, Low, Close"):
        scan_smts_historical(invalid, valid)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_returns_empty_dataframe_with_expected_columns tests/test_historical.py::test_scan_smts_historical_reuses_detector_validation_rules -v`

Expected: FAIL with import error or missing `scan_smts_historical`

**Step 3: Write minimal implementation**

```python
# smt/historical.py
import pandas as pd

from .detector import _validate_dataframes


EVENT_COLUMNS = [
    "signal_type",
    "created_ts",
    "reference_timestamp",
    "sweeping_asset",
    "failing_asset",
    "reference_price",
    "invalidation_asset",
    "invalidation_direction",
    "invalidation_level",
    "broken_ts",
    "status",
]


def scan_smts_historical(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names=("A1", "A2"),
    lookback_period: int = 20,
    enable_micro: bool = True,
    enable_swing: bool = True,
    enable_fvg: bool = True,
) -> pd.DataFrame:
    _validate_dataframes(df_a1, df_a2, min_len=2)
    return pd.DataFrame(columns=EVENT_COLUMNS)
```

```python
# smt/__init__.py
from .historical import scan_smts_historical
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_returns_empty_dataframe_with_expected_columns tests/test_historical.py::test_scan_smts_historical_reuses_detector_validation_rules -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_historical.py smt/historical.py smt/__init__.py
git commit -m "feat: add historical SMT scanner scaffold"
```

### Task 2: Add historical micro SMT detection and sorting

**Files:**
- Modify: `tests/test_historical.py`
- Modify: `smt/historical.py`

**Step 1: Write the failing test**

```python
def test_scan_smts_historical_emits_micro_events_with_metadata():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
            (100.0, 101.0, 90.5, 100.5),
        ]
    )
    df_a2 = _build_df(
        [
            (100.0, 105.0, 95.0, 100.0),
            (99.0, 104.0, 96.0, 102.0),
            (101.0, 103.5, 96.5, 102.0),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        enable_micro=True,
        enable_swing=False,
        enable_fvg=False,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "Bearish Micro SMT"
    assert row["created_ts"] == df_a1.index[1]
    assert row["reference_timestamp"] is pd.NaT or pd.isna(row["reference_timestamp"])
    assert row["sweeping_asset"] == "ES"
    assert row["failing_asset"] == "NQ"
    assert row["invalidation_asset"] == "NQ"
    assert row["invalidation_direction"] == "above"
    assert row["status"] == "active"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_emits_micro_events_with_metadata -v`

Expected: FAIL because the historical scanner returns no events

**Step 3: Write minimal implementation**

```python
def _event_row(signal):
    return {
        "signal_type": signal["signal_type"],
        "created_ts": signal["timestamp"],
        "reference_timestamp": signal.get("reference_timestamp", pd.NaT),
        "sweeping_asset": signal["sweeping_asset"],
        "failing_asset": signal["failing_asset"],
        "reference_price": signal["reference_price"],
        "invalidation_asset": signal["invalidation_asset"],
        "invalidation_direction": signal["invalidation_direction"],
        "invalidation_level": signal["invalidation_level"],
        "broken_ts": pd.NaT,
        "status": "active",
    }


def _scan_micro_events(df_a1, df_a2, asset_names):
    rows = []
    for end in range(2, len(df_a1) + 1):
        signal = check_micro_smt(df_a1.iloc[:end], df_a2.iloc[:end], asset_names=asset_names)
        if signal is not None:
            rows.append(_event_row(signal))
    return rows
```

Use `_scan_micro_events(...)` inside `scan_smts_historical(...)`, build a DataFrame from the rows, and sort by `created_ts` then `signal_type`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_emits_micro_events_with_metadata -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_historical.py smt/historical.py
git commit -m "feat: add historical micro SMT detection"
```

### Task 3: Add historical invalidation resolution for active vs broken status

**Files:**
- Modify: `tests/test_historical.py`
- Modify: `smt/historical.py`

**Step 1: Write the failing test**

```python
def test_scan_smts_historical_resolves_broken_ts_from_invalidation_asset():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
            (100.0, 106.0, 99.0, 101.0),
            (101.0, 104.8, 100.0, 102.0),
        ]
    )
    df_a2 = _build_df(
        [
            (100.0, 105.0, 95.0, 100.0),
            (99.0, 104.0, 96.0, 102.0),
            (102.0, 104.5, 101.0, 103.0),
            (103.0, 105.5, 102.0, 104.0),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        enable_micro=True,
        enable_swing=False,
        enable_fvg=False,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["created_ts"] == df_a1.index[1]
    assert row["broken_ts"] == df_a1.index[3]
    assert row["status"] == "broken"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_resolves_broken_ts_from_invalidation_asset -v`

Expected: FAIL because `broken_ts` remains missing and `status` stays `"active"`

**Step 3: Write minimal implementation**

```python
def _resolve_broken_ts(events: pd.DataFrame, df_a1: pd.DataFrame, df_a2: pd.DataFrame, asset_names):
    if events.empty:
        return events

    asset_map = {
        asset_names[0]: df_a1,
        asset_names[1]: df_a2,
    }

    resolved = events.copy()
    broken_ts = []
    status = []

    for row in resolved.itertuples(index=False):
        future = asset_map[row.invalidation_asset].loc[
            asset_map[row.invalidation_asset].index > row.created_ts
        ]

        if row.invalidation_direction == "above":
            hits = future.index[future["High"] >= row.invalidation_level]
        else:
            hits = future.index[future["Low"] <= row.invalidation_level]

        if len(hits) == 0:
            broken_ts.append(pd.NaT)
            status.append("active")
        else:
            broken_ts.append(hits[0])
            status.append("broken")

    resolved["broken_ts"] = broken_ts
    resolved["status"] = status
    return resolved
```

Call `_resolve_broken_ts(...)` before returning from `scan_smts_historical(...)`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_resolves_broken_ts_from_invalidation_asset -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_historical.py smt/historical.py
git commit -m "feat: resolve historical SMT invalidations"
```

### Task 4: Add historical swing SMT detection

**Files:**
- Modify: `tests/test_historical.py`
- Modify: `smt/historical.py`

**Step 1: Write the failing test**

```python
def test_scan_smts_historical_emits_swing_events():
    df_a1 = _build_df(
        [
            (10.0, 11.0, 9.0, 10.0),
            (10.0, 10.5, 8.0, 8.5),
            (8.5, 10.0, 8.7, 9.5),
            (9.5, 10.0, 8.9, 9.8),
            (9.8, 10.1, 7.5, 7.9),
        ]
    )
    df_a2 = _build_df(
        [
            (20.0, 21.0, 19.0, 20.0),
            (20.0, 20.5, 18.0, 18.5),
            (18.5, 20.0, 18.7, 19.5),
            (19.5, 20.0, 18.9, 19.8),
            (19.8, 20.1, 18.2, 19.0),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        lookback_period=4,
        enable_micro=False,
        enable_swing=True,
        enable_fvg=False,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "Bullish Swing SMT"
    assert row["created_ts"] == df_a1.index[4]
    assert row["reference_timestamp"] == df_a1.index[1]
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_emits_swing_events -v`

Expected: FAIL because swing events are not yet emitted

**Step 3: Write minimal implementation**

```python
def _scan_swing_events(df_a1, df_a2, lookback_period, asset_names):
    rows = []
    for end in range(lookback_period + 1, len(df_a1) + 1):
        signal = check_swing_smt(
            df_a1.iloc[:end],
            df_a2.iloc[:end],
            lookback_period=lookback_period,
            asset_names=asset_names,
        )
        if signal is not None:
            rows.append(_event_row(signal))
    return rows
```

Wire `_scan_swing_events(...)` into `scan_smts_historical(...)` when `enable_swing` is true.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_emits_swing_events -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_historical.py smt/historical.py
git commit -m "feat: add historical swing SMT detection"
```

### Task 5: Add historical FVG SMT detection and detector toggles

**Files:**
- Modify: `tests/test_historical.py`
- Modify: `smt/historical.py`

**Step 1: Write the failing test**

```python
def test_scan_smts_historical_emits_fvg_events():
    df_a1 = _build_df(
        [
            (8.0, 9.0, 7.0, 8.0),
            (9.0, 10.0, 9.0, 9.5),
            (14.0, 15.0, 13.0, 14.0),
            (12.5, 13.0, 12.0, 12.8),
            (12.7, 13.0, 12.5, 12.6),
            (12.0, 12.5, 11.5, 12.0),
        ]
    )
    df_a2 = _build_df(
        [
            (19.0, 20.0, 19.0, 19.5),
            (22.5, 23.0, 22.0, 22.6),
            (22.8, 23.0, 22.5, 22.9),
            (23.5, 24.0, 23.0, 23.5),
            (22.5, 23.0, 22.0, 22.4),
            (22.6, 23.0, 22.5, 22.8),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        lookback_period=4,
        asset_names=("ES", "NQ"),
        enable_micro=False,
        enable_swing=False,
        enable_fvg=True,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "Bullish FVG SMT"
    assert row["reference_timestamp"] == df_a2.index[0]


def test_scan_smts_historical_honors_detector_toggles():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
        ]
    )
    df_a2 = _build_df(
        [
            (100.0, 105.0, 95.0, 100.0),
            (99.0, 104.0, 96.0, 102.0),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        enable_micro=False,
        enable_swing=False,
        enable_fvg=False,
    )

    assert result.empty
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_emits_fvg_events tests/test_historical.py::test_scan_smts_historical_honors_detector_toggles -v`

Expected: FAIL because FVG events are not yet emitted

**Step 3: Write minimal implementation**

```python
def _scan_fvg_events(df_a1, df_a2, lookback_period, asset_names):
    rows = []
    for end in range(lookback_period + 1, len(df_a1) + 1):
        signal = check_fvg_smt(
            df_a1.iloc[:end],
            df_a2.iloc[:end],
            lookback_period=lookback_period,
            asset_names=asset_names,
        )
        if signal is not None:
            rows.append(_event_row(signal))
    return rows
```

Wire `_scan_fvg_events(...)` into `scan_smts_historical(...)`, concatenate all enabled rows, resolve invalidations, and return the sorted result with `EVENT_COLUMNS`.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_emits_fvg_events tests/test_historical.py::test_scan_smts_historical_honors_detector_toggles -v`

Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_historical.py smt/historical.py
git commit -m "feat: add historical FVG SMT detection"
```

### Task 6: Update README and package docs for the historical scanner

**Files:**
- Modify: `README.md`
- Modify: `tests/test_historical.py`

**Step 1: Write the failing test**

```python
def test_scan_smts_historical_returns_columns_used_in_documentation():
    df_a1 = _build_df([(100.0, 101.0, 99.0, 100.0), (100.5, 101.5, 99.5, 101.0)])
    df_a2 = _build_df([(200.0, 201.0, 199.0, 200.0), (200.5, 201.5, 199.5, 201.0)])

    result = scan_smts_historical(
        df_a1,
        df_a2,
        enable_micro=False,
        enable_swing=False,
        enable_fvg=False,
    )

    assert "created_ts" in result.columns
    assert "broken_ts" in result.columns
    assert "status" in result.columns
```

This test is intentionally lightweight: README behavior is anchored by the public API columns rather than brittle string assertions on documentation text.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_historical.py::test_scan_smts_historical_returns_columns_used_in_documentation -v`

Expected: PASS if the API is already correct. If it passes, proceed without code changes to the API and update only `README.md`.

**Step 3: Write minimal implementation**

Update `README.md` to:

- keep the existing live `SMTManager` quick-start
- add a new `Historical / Backtesting` section using `scan_smts_historical`
- explain `SMTManager` vs `scan_smts_historical`
- show a short example filtering the returned event table by `status` or `signal_type`

Example snippet to include:

```python
from smt import scan_smts_historical

events = scan_smts_historical(
    df_es,
    df_nq,
    asset_names=("ES", "NQ"),
    lookback_period=20,
)

active = events[events["status"] == "active"]
bearish_micro = events[events["signal_type"] == "Bearish Micro SMT"]
```

**Step 4: Run focused tests and full suite**

Run: `python -m pytest tests/test_historical.py -v`

Expected: PASS

Run: `python -m pytest -v`

Expected: PASS all tests, including existing manager and detector tests

**Step 5: Commit**

```bash
git add README.md tests/test_historical.py
git commit -m "docs: add historical SMT backtesting usage"
```
