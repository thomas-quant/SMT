# Historical Vectorized SMT Design

## Goal

Add a historical/backtesting API that scans full aligned OHLC history and returns an SMT event table, while keeping the existing live/stateful manager workflow unchanged.

## Scope

This design adds a separate batch API for historical analysis:

- `scan_smts_historical(df_a1, df_a2, ...) -> pd.DataFrame`

It covers:

- batch detection of micro, swing, and FVG SMTs across full history
- batch invalidation resolution into `broken_ts` and `status`
- public API export for the historical scanner
- test coverage for the historical path
- README updates documenting the new historical workflow alongside the existing live manager workflow

It does not change the existing `SMTManager`, `SMTRegistry`, or `SMTBreak` live path beyond any shared validation reuse needed by the historical implementation.

## Constraints

- Keep the current candle-by-candle live API intact.
- Use the same input requirements as the current detectors: aligned `DatetimeIndex`, unique/sorted indexes, and `Open`/`High`/`Low`/`Close` columns.
- Return a pandas DataFrame as the historical result, not custom state objects.
- Keep the historical path simple and explicit rather than trying to unify it with the live manager abstraction.

## Approaches Considered

### 1. Retrofit `SMTManager` for batch history

Teach `SMTManager` to run over full DataFrames and emit a final historical result.

Pros:

- Reuses an existing public entry point.

Cons:

- Forces one class to serve both live and historical use cases.
- Preserves stateful abstractions that are unnecessary for backtesting.
- Makes the manager API more complex and less clear.

### 2. Add a separate historical batch API

Create a dedicated historical scanning function that accepts complete DataFrames and returns an event table.

Pros:

- Simplest user-facing shape for backtesting.
- Leaves the live path untouched.
- Avoids registry and break-tracker state for the historical workflow.

Cons:

- Adds a second public entry point.
- Requires some historical-specific implementation rather than reusing the manager directly.

### 3. Expose detector-only batch outputs and require callers to resolve invalidations separately

Return creation events only and make backtest users run a second helper for lifecycle completion.

Pros:

- Maximum internal separation of concerns.

Cons:

- Worse ergonomics for the main use case.
- Easy for callers to misuse.
- Produces an incomplete backtesting result by default.

## Chosen Design

Use approach 2.

The package will expose a dedicated historical API that returns one canonical event table for the whole history. Internally, it will still separate detection from invalidation resolution, but callers will receive a fully enriched result in one function call.

## Architecture

Add a new module:

- `smt/historical.py`

Public function:

```python
scan_smts_historical(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names: tuple[str, str] = ("A1", "A2"),
    lookback_period: int = 20,
    enable_micro: bool = True,
    enable_swing: bool = True,
    enable_fvg: bool = True,
) -> pd.DataFrame
```

This function will:

1. Validate inputs using the same rules as the detector path.
2. Detect all historical SMT creation events.
3. Resolve the first future invalidation crossing for each event.
4. Return a single event table sorted by `created_ts`.

The existing live architecture remains unchanged:

- `SMTManager` stays stateful and candle-by-candle.
- `SMTRegistry` remains the live lifecycle store.
- `SMTBreak` remains the live invalidation tracker.

## Historical Data Flow

The historical pipeline has two internal stages.

### 1. Creation event detection

Each detector mode produces rows with a shared schema:

- `signal_type`
- `created_ts`
- `reference_timestamp`
- `sweeping_asset`
- `failing_asset`
- `reference_price`
- `invalidation_asset`
- `invalidation_direction`
- `invalidation_level`

Detection strategy:

- Micro SMT: vectorized with shifted comparisons over the full series.
- Swing SMT: batch scan across history using the current swing logic, but producing rows for every qualifying completion bar.
- FVG SMT: batch scan across history using the current FVG logic, but producing rows for every qualifying completion bar.

All enabled detector outputs are concatenated and sorted.

### 2. Invalidation resolution

For each creation event:

- choose the future candles for `invalidation_asset` after `created_ts`
- if `invalidation_direction == "above"`, find the first future bar where `High >= invalidation_level`
- if `invalidation_direction == "below"`, find the first future bar where `Low <= invalidation_level`

If a crossing exists:

- `broken_ts` is the first crossing timestamp
- `status` is `"broken"`

Otherwise:

- `broken_ts` is `NaT`
- `status` is `"active"`

This historical path does not create UUIDs or runtime tracker state.

## Event Table Contract

The returned DataFrame will contain exactly these columns:

- `signal_type`
- `created_ts`
- `reference_timestamp`
- `sweeping_asset`
- `failing_asset`
- `reference_price`
- `invalidation_asset`
- `invalidation_direction`
- `invalidation_level`
- `broken_ts`
- `status`

Behavioral rules:

- empty results return an empty DataFrame with the full schema
- `reference_timestamp` uses `NaT` where not applicable
- `status` is derived only from historical invalidation resolution
- rows are sorted by `created_ts` with a stable secondary ordering

## Testing Strategy

Add a new test module:

- `tests/test_historical.py`

Cover:

1. micro SMT creation over full history
2. swing SMT creation over full history
3. FVG SMT creation over full history
4. invalidation resolution for both `"above"` and `"below"`
5. empty/no-signal output shape
6. malformed dataframe validation
7. detector toggle behavior
8. metadata consistency with the current detector signal fields

Tests should use small synthetic OHLC datasets with explicit timestamps so both creation and invalidation timing are deterministic.

## README Updates

Update `README.md` to document both modes clearly.

Keep:

- the current `SMTManager` example for live sequential use

Add:

- a new historical/backtesting section using `scan_smts_historical(...)`
- a short explanation of when to use `SMTManager` versus the batch historical scanner
- an example showing the returned event table and simple filtering by `signal_type` or `status`

## Risks

- Swing and FVG historical scanning are more complex than micro and may tempt over-abstraction.
- Reusing current detector internals too literally may produce a slow or awkward batch implementation.
- If event ordering is not pinned down, tests may become brittle around same-timestamp multi-signal cases.

## Non-Goals

- Replacing `SMTManager` with the historical scanner
- persistence or serialization
- portfolio logic or PnL calculation
- bar-aligned feature engineering outputs
