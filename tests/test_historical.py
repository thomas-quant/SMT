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
