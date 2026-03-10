# SMT Tracking And API Fixes Design

## Goal

Fix the five review findings in the SMT package by making invalidation tracking asset-aware, tightening dataframe validation, preventing registry state leakage, correcting FVG reference timestamps, and bringing the public documentation back in sync with the code.

## Scope

This design covers these five issues:

1. Cross-asset invalidation is tracked against a single undifferentiated price stream.
2. Registry read methods leak mutable internal state.
3. FVG signals report `reference_timestamp` inconsistently.
4. Detector validation is too weak for a library API.
5. README examples and API docs no longer match the implementation.

## Constraints

- The public API can change to support correct asset-aware break tracking.
- Detection functions should remain stateless with no lifecycle side effects.
- Registry remains in-memory only.
- Break tracking should use information already present in the signal instead of re-inferring intent from price history.

## Approaches Considered

### 1. Rewrite the manager API and make tracking asset-aware

Change `SMTManager.update()` to accept only `df_a1` and `df_a2`. The manager reads the current candle for each asset from the latest row and updates break tracking per asset. Detector signals explicitly declare which asset owns the invalidation level and which direction breaks it.

Pros:

- Fixes the root cause directly.
- Removes duplicated current-candle inputs.
- Prevents inconsistent caller-supplied break data.

Cons:

- Breaking API change for existing callers.

### 2. Keep explicit current-candle inputs, but require a per-asset candle map

Preserve a manager-level break tracking input, but change it from one price stream to two named price streams.

Pros:

- Also correct.

Cons:

- Keeps duplicate inputs that can drift from the DataFrames.
- More surface area with no clear gain.

### 3. Infer asset identity from existing IDs or signal fields

Keep the single-stream manager API and try to determine which asset to use from the UUID or surrounding metadata.

Pros:

- Smaller API diff.

Cons:

- Does not work. The current tracker still evaluates only one price stream.
- UUIDs are registry identifiers, not market identity.

## Chosen Design

Use approach 1.

## Architecture Changes

### Signals

Every detector output will include:

- `invalidation_asset`
- `invalidation_direction`

`invalidation_asset` names the asset whose price should be watched for invalidation. This is the failing asset, because its key level is the invalidation threshold.

`invalidation_direction` is explicit:

- `"above"` for bearish SMTs
- `"below"` for bullish SMTs

This removes hidden inference from the break tracker.

### Break Tracking

`SMTBreak` will store entries keyed by registry ID with:

- invalidation asset
- invalidation direction
- invalidation level
- original signal payload

It will also store last close per asset rather than a single global `last_price`.

The break tracker will expose an asset-aware update method that receives the latest candle for each asset and checks only the entries tied to that asset.

### Manager

`SMTManager.update()` will become:

```python
update(df_a1: pd.DataFrame, df_a2: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]
```

The manager will:

1. Validate/run detectors against the two DataFrames.
2. Register new signals in the registry.
3. Register each signal with `SMTBreak`.
4. Read the latest OHLC values for both assets from the DataFrames.
5. Ask the break tracker to evaluate invalidation per asset.
6. Mark broken SMTs in the registry and return registry snapshots.

No separate `high`, `low`, `close`, or `ts` arguments will remain.

### Registry

Registry query methods will return deep copies so callers cannot mutate stored lifecycle state. The registry remains the only owner of mutable SMT entries.

### Detector Validation

Validation will become strict enough for library use:

- require `Open`, `High`, `Low`, `Close`
- require both DataFrames to have the same full index
- require monotonic increasing and unique indexes
- raise `ValueError` for malformed inputs
- still return `None` for valid inputs that simply do not produce a signal

This keeps bad caller input distinct from “no pattern found”.

### FVG Reference Timestamp

`check_fvg_smt()` will mirror `check_swing_smt()` and emit the earlier of the two matched reference timestamps.

## Testing Strategy

Add regression tests before implementation for each finding:

1. Manager/break tracking only breaks an SMT when the invalidation asset crosses its level.
2. Registry read APIs return isolated copies.
3. FVG reference timestamp uses the earlier aligned reference.
4. Invalid DataFrames raise `ValueError`.
5. README examples/API descriptions match the implemented signatures and return shapes where feasible through focused assertions on public APIs.

Tests should use small synthetic OHLC DataFrames with explicit timestamps so the lifecycle is deterministic.

## Risks

- Tight validation may break permissive existing callers. That is acceptable because silent misuse is worse.
- Rewriting the manager API requires README and downstream caller updates.
- Asset-aware tracking introduces more explicit signal fields; tests need to pin those fields to avoid regressions.

## Non-Goals

- Persistence or serialization.
- Ranking or scoring SMTs.
- Expanding detector pattern definitions beyond the existing three modes.
