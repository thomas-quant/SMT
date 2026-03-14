import pandas as pd

from .detector import _validate_dataframes, check_micro_smt


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

    if not rows:
        return _empty_events()

    return pd.DataFrame(rows).sort_values(
        by=["created_ts", "signal_type"],
        kind="stable",
    )[EVENT_COLUMNS].reset_index(drop=True)
