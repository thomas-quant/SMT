# Historical Low-Level Swing/FVG Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the historical swing and FVG scanners with low-level array-based implementations that preserve the current non-vectorized detector behavior.

**Architecture:** Keep the public historical API unchanged and confine the rewrite to private helpers in `smt/historical.py`. Use the existing detector functions as the oracle by adding parity tests first, then replace the expanding-window swing and FVG scanners one path at a time.

**Tech Stack:** Python 3.12, pandas, numpy, pytest

---

### Task 1: Add oracle parity tests for swing and FVG historical scanning

**Files:**
- Modify: `tests/test_historical.py`

**Step 1: Write the failing test**

```python
from smt.detector import check_fvg_smt, check_swing_smt


def _expected_historical_events_from_detector(df_a1, df_a2, detector_fn, *, lookback_period, asset_names):
    rows = []
    min_end = lookback_period + 1
    for end in range(min_end, len(df_a1) + 1):
        signal = detector_fn(
            df_a1.iloc[:end],
            df_a2.iloc[:end],
            lookback_period=lookback_period,
            asset_names=asset_names,
        )
        if signal is None:
            continue
        rows.append(
            {
                "signal_type": signal["signal_type"],
                "created_ts": signal["timestamp"],
                "reference_timestamp": signal.get("reference_timestamp", pd.NaT),
                "sweeping_asset": signal["sweeping_asset"],
                "failing_asset": signal["failing_asset"],
                "reference_price": signal["reference_price"],
                "invalidation_asset": signal["invalidation_asset"],
                "invalidation_direction": signal["invalidation_direction"],
                "invalidation_level": signal["invalidation_level"],
            }
        )
    return pd.DataFrame(rows)


def test_historical_swing_matches_detector_oracle():
    ...


def test_historical_fvg_matches_detector_oracle():
    ...
```

Each test should:

- build multi-event fixtures
- call `scan_smts_historical(...)` with only one detector enabled
- build expected rows by replaying the detector
- compare exact event fields after selecting the same columns

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_historical.py::test_historical_swing_matches_detector_oracle tests/test_historical.py::test_historical_fvg_matches_detector_oracle -v`

Expected: FAIL because current historical output may still pass, but the tests should initially assert exact row equality including a sufficiently rich fixture that exposes differences once implementation changes begin

**Step 3: Write minimal implementation**

No production code yet. Refine the tests until they lock in:

- exact event ordering
- exact metadata parity
- exact post-resolution `broken_ts` and `status`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_historical.py::test_historical_swing_matches_detector_oracle tests/test_historical.py::test_historical_fvg_matches_detector_oracle -v`

Expected: PASS against the current expanding-window implementation

**Step 5: Commit**

```bash
git add tests/test_historical.py
git commit -m "test: add oracle parity coverage for historical swing and FVG"
```

### Task 2: Replace historical swing scanning with a low-level array path

**Files:**
- Modify: `smt/historical.py`
- Modify: `tests/test_historical.py`

**Step 1: Write the failing test**

Use the parity test from Task 1 plus a focused regression case if needed:

```python
def test_historical_swing_matches_detector_oracle():
    ...
```

If the oracle test is already green, add a second test that stresses:

- multiple swing candidates in the lookback window
- timestamp tolerance edge cases
- extreme validation edge cases

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_historical.py::test_historical_swing_matches_detector_oracle -v`

Expected: FAIL after the first low-level implementation draft if parity is incomplete

**Step 3: Write minimal implementation**

Add private helpers in `smt/historical.py`, for example:

```python
def _swing_low_mask(low: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(low), dtype=bool)
    mask[1:-1] = (low[1:-1] < low[:-2]) & (low[1:-1] < low[2:])
    return mask


def _swing_high_mask(high: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(high), dtype=bool)
    mask[1:-1] = (high[1:-1] > high[:-2]) & (high[1:-1] > high[2:])
    return mask
```

Then implement a low-level `_scan_swing_events(...)` that:

- works from NumPy arrays
- inspects the same trailing lookback region
- picks the same most recent qualifying swing
- enforces the same divergence and extreme-validation rules
- emits rows via `_event_row(...)`

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_historical.py::test_historical_swing_matches_detector_oracle -v`

Expected: PASS

Then run:

Run: `python3 -m pytest tests/test_historical.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add smt/historical.py tests/test_historical.py
git commit -m "feat: add low-level historical swing scanner"
```

### Task 3: Replace historical FVG scanning with a low-level array path

**Files:**
- Modify: `smt/historical.py`
- Modify: `tests/test_historical.py`

**Step 1: Write the failing test**

Use the parity test from Task 1 plus a focused regression case if needed:

```python
def test_historical_fvg_matches_detector_oracle():
    ...
```

If needed, add a second fixture that stresses:

- multiple recent FVG candidates
- partial fills
- fully invalidated gaps

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_historical.py::test_historical_fvg_matches_detector_oracle -v`

Expected: FAIL after the first low-level implementation draft if parity is incomplete

**Step 3: Write minimal implementation**

Add private helpers in `smt/historical.py` to precompute candidate FVG starts and then resolve the most recent valid FVG per completion bar.

Skeleton:

```python
def _scan_fvg_events(...):
    rows = []
    # precompute bullish/bearish candidate positions
    # for each completion bar, inspect only the same historical window
    # reproduce partial-fill and full-fill checks from detector.py
    return rows
```

The implementation must preserve:

- candidate type matching across assets
- earlier `reference_timestamp`
- same partial-fill adjustment to top/bottom
- same divergence behavior

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_historical.py::test_historical_fvg_matches_detector_oracle -v`

Expected: PASS

Then run:

Run: `python3 -m pytest tests/test_historical.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add smt/historical.py tests/test_historical.py
git commit -m "feat: add low-level historical FVG scanner"
```

### Task 4: Verify parity and performance across the full historical scanner

**Files:**
- Modify: `tests/test_historical.py`
- Modify: `README.md` (only if wording needs correction; otherwise no change)

**Step 1: Write the failing test**

Add one integration test that uses both `swing` and `fvg` enabled on a richer fixture and compares the historical scanner output against concatenated detector-oracle rows.

```python
def test_historical_scanner_matches_combined_detector_oracle():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_historical.py::test_historical_scanner_matches_combined_detector_oracle -v`

Expected: FAIL until ordering and combined parity are correct

**Step 3: Write minimal implementation**

Adjust event ordering or helper behavior in `smt/historical.py` only if needed to keep exact parity.

**Step 4: Run full verification**

Run: `python3 -m pytest tests/test_historical.py -v`

Expected: PASS

Run: `python3 -m pytest -v`

Expected: PASS all tests

Run: `python3 - <<'PY'
import time
import numpy as np
import pandas as pd
from smt.historical import scan_smts_historical
...
PY`

Expected: observe lower swing/FVG runtime than the current expanding-window implementation

**Step 5: Commit**

```bash
git add smt/historical.py tests/test_historical.py README.md
git commit -m "test: verify historical low-level scanner parity"
```
