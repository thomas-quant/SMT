# Historical Low-Level Swing/FVG Rewrite Design

## Goal

Replace the historical `swing` and `fvg` scanners with low-level array-based implementations that preserve the current non-vectorized detector behavior as closely as possible.

## Scope

This design changes only the historical/backtesting path in `smt/historical.py`.

In scope:

- rewrite historical swing detection
- rewrite historical FVG detection
- preserve parity with `check_swing_smt()` and `check_fvg_smt()`
- add oracle parity tests that compare the low-level path to the current non-vectorized behavior

Out of scope:

- changing `SMTManager`
- changing `check_swing_smt()` or `check_fvg_smt()`
- changing historical `micro` detection
- changing historical `broken_ts` resolution
- changing the public event table schema

## Constraints

- Historical output must match the existing expanding-window implementation as closely as possible.
- The detector module remains the source of truth for behavior.
- The public API stays `scan_smts_historical(...) -> pd.DataFrame`.
- Any performance gain must not come from changing event selection rules.

## Approaches Considered

### 1. Low-level NumPy rewrite inside `historical.py`

Replace the expanding-window swing/FVG scanners with private array-based helpers while leaving the detector module unchanged.

Pros:

- Best performance upside
- Keeps live path untouched
- Safest path for parity testing

Cons:

- Most implementation detail

### 2. Hybrid cached scanner

Keep more of the current detector calling pattern, but precompute some metadata.

Pros:

- Smaller change

Cons:

- Likely not enough speedup for swing-heavy backtests

### 3. Behavior rewrite

Simplify the historical rules and optimize around the simpler behavior.

Pros:

- Easier to implement

Cons:

- Violates the parity requirement

## Chosen Design

Use approach 1.

## Architecture

Keep all new logic private to `smt/historical.py`.

- `scan_smts_historical()` remains unchanged at the API level
- `_scan_micro_events()` remains unchanged
- `_resolve_broken_ts()` remains unchanged
- replace `_scan_swing_events()` with a low-level swing scanner
- replace `_scan_fvg_events()` with a low-level FVG scanner

`smt/detector.py` remains unchanged and acts as the behavioral oracle for parity tests.

## Swing Data Flow

The low-level swing scanner will:

1. Materialize per-asset arrays for `High`, `Low`, and timestamps.
2. Precompute swing-low and swing-high masks using the same adjacent-candle rules as the detector.
3. For each completion bar, inspect only the same trailing lookback region used by `check_swing_smt()`.
4. Select the most recent qualifying swing per asset inside that region.
5. Enforce the same `timestamp_tolerance`.
6. Apply the same divergence rule.
7. Apply the same “current bar is the extreme since the swing” validation.
8. Emit the same event metadata fields.

## FVG Data Flow

The low-level FVG scanner will:

1. Materialize per-asset arrays for `High`, `Low`, and timestamps.
2. Precompute bullish and bearish FVG candidates from 3-candle relationships.
3. For each completion bar, inspect the same lookback region used by `_find_recent_valid_fvg()`.
4. Select the most recent valid FVG in that region.
5. Reproduce the current partial-fill and full-invalidation logic.
6. Apply the same divergence rule and reference timestamp selection.
7. Emit the same event metadata fields.

## Testing Strategy

Keep the existing fixture tests and add oracle parity tests.

### Swing parity

Build expected swing events in the test by replaying `check_swing_smt()` over expanding windows, then compare them to the historical scanner with only `swing` enabled.

### FVG parity

Build expected FVG events in the test by replaying `check_fvg_smt()` over expanding windows, then compare them to the historical scanner with only `fvg` enabled.

### Coverage

- same columns
- same row count
- same timestamps
- same reference timestamps
- same asset metadata
- same invalidation metadata
- same final `broken_ts` and `status` after historical resolution

## Risks

- FVG parity is easier to get wrong because the current logic tracks partial fills inside the lookback window.
- Swing parity is easier to get wrong around “most recent valid swing” selection and extreme validation.
- Low-level code can become unreadable if optimization is prioritized over correctness too early.

## Non-Goals

- Rewriting the live detector path
- Changing SMT definitions
- Optimizing historical `micro`
- Optimizing `broken_ts` resolution in this pass
