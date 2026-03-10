import pandas as pd

from smt import SMTManager


def _build_df(rows):
    index = pd.DatetimeIndex(
        [
            "2024-01-01 09:30:00",
            "2024-01-01 09:35:00",
            "2024-01-01 09:40:00",
            "2024-01-01 09:45:00",
        ]
    )[: len(rows)]
    return pd.DataFrame(rows, index=index, columns=["Open", "High", "Low", "Close"])


def test_manager_breaks_signal_only_when_invalidation_asset_crosses():
    df_a1 = _build_df(
        [
            (95.0, 100.0, 90.0, 95.0),
            (96.0, 101.0, 91.0, 100.0),
            (100.0, 106.0, 99.0, 101.0),
            (101.0, 104.8, 100.0, 102.0),
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

    manager = SMTManager(
        timeframe="5m",
        asset_names=("ES", "NQ"),
        enable_micro=True,
        enable_swing=False,
        enable_fvg=False,
    )

    signal_result = manager.update(df_a1.iloc[:2], df_a2.iloc[:2])
    assert len(signal_result["new_smts"]) == 1

    smt = signal_result["new_smts"][0]
    assert smt["signal"]["signal_type"] == "Bearish Micro SMT"
    assert smt["signal"]["sweeping_asset"] == "ES"
    assert smt["signal"]["failing_asset"] == "NQ"
    assert smt["signal"]["invalidation_asset"] == "NQ"
    assert smt["signal"]["invalidation_direction"] == "above"

    wrong_asset_result = manager.update(df_a1.iloc[:3], df_a2.iloc[:3])
    assert wrong_asset_result["broken_smts"] == []
    assert len(manager.get_active_smts()) == 1

    right_asset_result = manager.update(df_a1, df_a2)
    assert len(right_asset_result["broken_smts"]) == 1
    assert right_asset_result["broken_smts"][0]["id"] == smt["id"]
    assert len(manager.get_broken_smts()) == 1
