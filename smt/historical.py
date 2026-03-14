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
