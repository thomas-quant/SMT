import pandas as pd

from .detector import _validate_dataframes, check_micro_smt, check_swing_smt


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


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)


def _event_row(signal) -> dict:
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


def _scan_micro_events(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names,
) -> list[dict]:
    rows = []
    for end in range(2, len(df_a1) + 1):
        signal = check_micro_smt(df_a1.iloc[:end], df_a2.iloc[:end], asset_names=asset_names)
        if signal is not None:
            rows.append(_event_row(signal))
    return rows


def _resolve_broken_ts(
    events: pd.DataFrame,
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    asset_names,
) -> pd.DataFrame:
    if events.empty:
        return events

    asset_map = {
        asset_names[0]: df_a1,
        asset_names[1]: df_a2,
    }

    resolved = events.copy()
    broken_timestamps = []
    statuses = []

    for row in resolved.itertuples(index=False):
        asset_df = asset_map[row.invalidation_asset]
        future = asset_df.loc[asset_df.index > row.created_ts]

        if row.invalidation_direction == "above":
            hits = future.index[future["High"] >= row.invalidation_level]
        else:
            hits = future.index[future["Low"] <= row.invalidation_level]

        if len(hits) == 0:
            broken_timestamps.append(pd.NaT)
            statuses.append("active")
        else:
            broken_timestamps.append(hits[0])
            statuses.append("broken")

    resolved["broken_ts"] = broken_timestamps
    resolved["status"] = statuses
    return resolved


def _scan_swing_events(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names,
) -> list[dict]:
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

    rows = []
    if enable_micro:
        rows.extend(_scan_micro_events(df_a1, df_a2, asset_names))
    if enable_swing:
        rows.extend(_scan_swing_events(df_a1, df_a2, lookback_period, asset_names))

    if not rows:
        return _empty_events()

    events = pd.DataFrame(rows).sort_values(
        by=["created_ts", "signal_type"],
        kind="stable",
    ).reset_index(drop=True)

    return _resolve_broken_ts(events, df_a1, df_a2, asset_names)[EVENT_COLUMNS]
