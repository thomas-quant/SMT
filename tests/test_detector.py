import pandas as pd
import pytest

from smt.detector import check_fvg_smt, check_micro_smt


def _build_df(rows, index=None):
    if index is None:
        index = pd.DatetimeIndex(
            [
                "2024-01-01 09:30:00",
                "2024-01-01 09:35:00",
                "2024-01-01 09:40:00",
                "2024-01-01 09:45:00",
                "2024-01-01 09:50:00",
                "2024-01-01 09:55:00",
            ]
        )[: len(rows)]
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])


def test_micro_smt_emits_invalidation_metadata():
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

    signal = check_micro_smt(df_a1, df_a2, asset_names=("ES", "NQ"))

    assert signal is not None
    assert signal["failing_asset"] == "NQ"
    assert signal["invalidation_asset"] == "NQ"
    assert signal["invalidation_direction"] == "above"


def test_fvg_smt_uses_earlier_reference_timestamp():
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

    signal = check_fvg_smt(
        df_a1,
        df_a2,
        lookback_period=4,
        asset_names=("ES", "NQ"),
    )

    assert signal is not None
    assert signal["signal_type"] == "Bullish FVG SMT"
    assert signal["reference_timestamp"] == df_a2.index[0]
    assert signal["invalidation_asset"] == signal["failing_asset"]
    assert signal["invalidation_direction"] == "below"


def test_detector_raises_for_missing_ohlc_columns():
    df_valid = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
        ]
    )
    df_missing = df_valid.drop(columns=["Low"])

    with pytest.raises(ValueError, match="Open, High, Low, Close"):
        check_micro_smt(df_missing, df_valid)


def test_detector_raises_for_misaligned_indexes():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
        ],
        index=pd.DatetimeIndex(
            ["2024-01-01 09:30:00", "2024-01-01 09:40:00"]
        ),
    )
    df_a2 = _build_df(
        [
            (100.0, 105.0, 95.0, 100.0),
            (99.0, 104.0, 96.0, 102.0),
        ],
        index=pd.DatetimeIndex(
            ["2024-01-01 09:35:00", "2024-01-01 09:40:00"]
        ),
    )

    with pytest.raises(ValueError, match="same index"):
        check_micro_smt(df_a1, df_a2)


def test_detector_raises_for_non_monotonic_or_duplicate_index():
    duplicate_index = pd.DatetimeIndex(
        ["2024-01-01 09:30:00", "2024-01-01 09:30:00"]
    )
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
        ],
        index=duplicate_index,
    )
    df_a2 = _build_df(
        [
            (100.0, 105.0, 95.0, 100.0),
            (99.0, 104.0, 96.0, 102.0),
        ],
        index=duplicate_index,
    )

    with pytest.raises(ValueError, match="unique and sorted"):
        check_micro_smt(df_a1, df_a2)


def test_detector_returns_none_for_valid_but_insufficient_history():
    df_a1 = _build_df([(95.0, 100.0, 90.0, 95.0)])
    df_a2 = _build_df([(100.0, 105.0, 95.0, 100.0)])

    assert check_micro_smt(df_a1, df_a2) is None
