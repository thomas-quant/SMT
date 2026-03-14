import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from smt import scan_smts_historical
from smt.detector import check_fvg_smt, check_swing_smt


HISTORICAL_EVENT_COLUMNS = [
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


def _build_walk_df(seed, n=60, jump_every=None, jump_size=3.0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01 09:30:00", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0.0, 0.6, n))

    if jump_every is not None:
        for i in range(jump_every, n, jump_every):
            close[i:] += jump_size if (i // jump_every) % 2 else -jump_size

    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.15, 0.75, n)
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
        },
        index=index,
    )


def _resolve_expected_events(events, df_a1, df_a2, asset_names):
    if events.empty:
        return events

    asset_map = {
        asset_names[0]: df_a1,
        asset_names[1]: df_a2,
    }
    resolved = events.copy()
    broken_ts = []
    status = []

    for row in resolved.itertuples(index=False):
        asset_df = asset_map[row.invalidation_asset]
        future = asset_df.loc[asset_df.index > row.created_ts]

        if row.invalidation_direction == "above":
            hits = future.index[future["High"] >= row.invalidation_level]
        else:
            hits = future.index[future["Low"] <= row.invalidation_level]

        if len(hits) == 0:
            broken_ts.append(pd.NaT)
            status.append("active")
        else:
            broken_ts.append(hits[0])
            status.append("broken")

    resolved["broken_ts"] = broken_ts
    resolved["status"] = status
    return resolved[HISTORICAL_EVENT_COLUMNS]


def _expected_historical_events_from_detector(
    df_a1,
    df_a2,
    detector_fn,
    *,
    lookback_period,
    asset_names,
):
    rows = []
    for end in range(lookback_period + 1, len(df_a1) + 1):
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
                "broken_ts": pd.NaT,
                "status": "active",
            }
        )

    if not rows:
        return pd.DataFrame(columns=HISTORICAL_EVENT_COLUMNS)

    events = pd.DataFrame(rows).sort_values(
        by=["created_ts", "signal_type"],
        kind="stable",
    ).reset_index(drop=True)
    return _resolve_expected_events(events, df_a1, df_a2, asset_names)


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


def test_scan_smts_historical_emits_swing_events():
    df_a1 = _build_df(
        [
            (10.0, 11.0, 9.0, 10.0),
            (10.0, 10.5, 8.0, 8.5),
            (8.5, 10.0, 8.7, 9.5),
            (9.5, 10.0, 8.9, 9.8),
            (9.8, 10.1, 7.5, 7.9),
        ]
    )
    df_a2 = _build_df(
        [
            (20.0, 21.0, 19.0, 20.0),
            (20.0, 20.5, 18.0, 18.5),
            (18.5, 20.0, 18.7, 19.5),
            (19.5, 20.0, 18.9, 19.8),
            (19.8, 20.1, 18.2, 19.0),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        lookback_period=4,
        enable_micro=False,
        enable_swing=True,
        enable_fvg=False,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "Bullish Swing SMT"
    assert row["created_ts"] == df_a1.index[4]
    assert row["reference_timestamp"] == df_a1.index[1]


def test_scan_smts_historical_emits_fvg_events():
    df_a1 = _build_df(
        [
            (8.0, 9.0, 7.0, 8.0),
            (9.0, 10.0, 9.0, 9.5),
            (14.0, 15.0, 13.0, 14.0),
            (12.5, 13.0, 12.0, 12.8),
            (12.7, 13.0, 12.5, 12.6),
        ]
    )
    df_a2 = _build_df(
        [
            (19.0, 20.0, 19.0, 19.5),
            (22.5, 23.0, 22.0, 22.6),
            (22.8, 23.0, 22.5, 22.9),
            (23.5, 24.0, 23.0, 23.5),
            (22.5, 23.0, 22.0, 22.4),
        ]
    )

    result = scan_smts_historical(
        df_a1,
        df_a2,
        lookback_period=4,
        asset_names=("ES", "NQ"),
        enable_micro=False,
        enable_swing=False,
        enable_fvg=True,
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["signal_type"] == "Bullish FVG SMT"
    assert row["reference_timestamp"] == df_a2.index[0]


def test_scan_smts_historical_honors_detector_toggles():
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

    result = scan_smts_historical(
        df_a1,
        df_a2,
        enable_micro=False,
        enable_swing=False,
        enable_fvg=False,
    )

    assert result.empty


def test_historical_swing_matches_detector_oracle():
    df_a1 = _build_walk_df(seed=1, n=60)
    df_a2 = _build_walk_df(seed=2, n=60)

    expected = _expected_historical_events_from_detector(
        df_a1,
        df_a2,
        check_swing_smt,
        lookback_period=20,
        asset_names=("ES", "NQ"),
    )
    actual = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        lookback_period=20,
        enable_micro=False,
        enable_swing=True,
        enable_fvg=False,
    )

    assert not expected.empty
    assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_historical_fvg_matches_detector_oracle():
    df_a1 = _build_walk_df(seed=3, n=60, jump_every=7, jump_size=2.5)
    df_a2 = _build_walk_df(seed=4, n=60, jump_every=7, jump_size=2.5)

    expected = _expected_historical_events_from_detector(
        df_a1,
        df_a2,
        check_fvg_smt,
        lookback_period=20,
        asset_names=("ES", "NQ"),
    )
    actual = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        lookback_period=20,
        enable_micro=False,
        enable_swing=False,
        enable_fvg=True,
    )

    assert not expected.empty
    assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_historical_scanner_matches_combined_detector_oracle():
    df_a1 = _build_walk_df(seed=5, n=60, jump_every=7, jump_size=2.5)
    df_a2 = _build_walk_df(seed=6, n=60, jump_every=7, jump_size=2.5)

    expected_swing = _expected_historical_events_from_detector(
        df_a1,
        df_a2,
        check_swing_smt,
        lookback_period=20,
        asset_names=("ES", "NQ"),
    )
    expected_fvg = _expected_historical_events_from_detector(
        df_a1,
        df_a2,
        check_fvg_smt,
        lookback_period=20,
        asset_names=("ES", "NQ"),
    )
    expected = pd.concat([expected_swing, expected_fvg], ignore_index=True).sort_values(
        by=["created_ts", "signal_type"],
        kind="stable",
    ).reset_index(drop=True)
    actual = scan_smts_historical(
        df_a1,
        df_a2,
        asset_names=("ES", "NQ"),
        lookback_period=20,
        enable_micro=False,
        enable_swing=True,
        enable_fvg=True,
    )

    assert not expected.empty
    assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
