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


def test_scan_smts_historical_emits_micro_events_with_metadata():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
            (100.0, 100.5, 91.2, 100.5),
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
    assert pd.isna(row["reference_timestamp"])
    assert row["sweeping_asset"] == "ES"
    assert row["failing_asset"] == "NQ"
    assert row["invalidation_asset"] == "NQ"
    assert row["invalidation_direction"] == "above"
    assert row["status"] == "active"


def test_scan_smts_historical_resolves_broken_ts_from_invalidation_asset():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
            (100.0, 106.0, 99.0, 101.0),
            (101.0, 106.2, 100.0, 102.0),
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
