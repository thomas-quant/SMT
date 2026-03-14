import numpy as np
import pandas as pd

from .detector import (
    _build_signal,
    _check_divergence,
    _validate_dataframes,
    check_micro_smt,
)


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


def _latest_position_in_range(
    positions: np.ndarray,
    lower: int,
    upper: int,
):
    if lower > upper:
        return None

    pos_idx = np.searchsorted(positions, upper, side="right") - 1
    if pos_idx < 0:
        return None

    position = int(positions[pos_idx])
    return position if position >= lower else None


def _swing_low_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    if len(values) >= 3:
        mask[1:-1] = (values[1:-1] < values[:-2]) & (values[1:-1] < values[2:])
    return mask


def _swing_high_mask(values: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(values), dtype=bool)
    if len(values) >= 3:
        mask[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] > values[2:])
    return mask


def _current_is_extreme(
    values: np.ndarray,
    swing_idx: int,
    current_idx: int,
    is_bullish: bool,
) -> bool:
    if swing_idx is None or swing_idx >= current_idx:
        return False

    between = values[swing_idx + 1 : current_idx + 1]
    if len(between) == 0:
        return False

    current_value = values[current_idx]
    if is_bullish:
        return current_value <= between.min()
    return current_value >= between.max()


def _scan_swing_events(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names,
) -> list[dict]:
    index = df_a1.index
    high_a1 = df_a1["High"].to_numpy()
    low_a1 = df_a1["Low"].to_numpy()
    high_a2 = df_a2["High"].to_numpy()
    low_a2 = df_a2["Low"].to_numpy()

    swing_low_positions_a1 = np.flatnonzero(_swing_low_mask(low_a1))
    swing_low_positions_a2 = np.flatnonzero(_swing_low_mask(low_a2))
    swing_high_positions_a1 = np.flatnonzero(_swing_high_mask(high_a1))
    swing_high_positions_a2 = np.flatnonzero(_swing_high_mask(high_a2))

    rows = []
    for current_idx in range(lookback_period, len(df_a1)):
        lower = current_idx - lookback_period + 1
        upper = current_idx - 2

        swing_low_a1 = _latest_position_in_range(swing_low_positions_a1, lower, upper)
        swing_low_a2 = _latest_position_in_range(swing_low_positions_a2, lower, upper)
        if (
            swing_low_a1 is not None
            and swing_low_a2 is not None
            and abs(swing_low_a1 - swing_low_a2) <= 1
        ):
            sweeper = _check_divergence(
                low_a1[current_idx],
                low_a2[current_idx],
                low_a1[swing_low_a1],
                low_a2[swing_low_a2],
                is_bullish=True,
            )
            if sweeper is not None:
                swing_idx = swing_low_a1 if sweeper == 0 else swing_low_a2
                sweep_values = low_a1 if sweeper == 0 else low_a2
                if _current_is_extreme(sweep_values, swing_idx, current_idx, is_bullish=True):
                    reference_timestamp = (
                        index[swing_low_a1] if swing_low_a1 <= swing_low_a2 else index[swing_low_a2]
                    )
                    signal = _build_signal(
                        signal_type="Bullish Swing SMT",
                        timestamp=index[current_idx],
                        sweeping_asset=asset_names[sweeper],
                        failing_asset=asset_names[1 - sweeper],
                        reference_price=low_a1[swing_low_a1] if sweeper == 0 else low_a2[swing_low_a2],
                        invalidation_level=low_a2[swing_low_a2] if sweeper == 0 else low_a1[swing_low_a1],
                        reference_timestamp=reference_timestamp,
                    )
                    rows.append(_event_row(signal))
                    continue

        swing_high_a1 = _latest_position_in_range(swing_high_positions_a1, lower, upper)
        swing_high_a2 = _latest_position_in_range(swing_high_positions_a2, lower, upper)
        if (
            swing_high_a1 is None
            or swing_high_a2 is None
            or abs(swing_high_a1 - swing_high_a2) > 1
        ):
            continue

        sweeper = _check_divergence(
            high_a1[current_idx],
            high_a2[current_idx],
            high_a1[swing_high_a1],
            high_a2[swing_high_a2],
            is_bullish=False,
        )
        if sweeper is None:
            continue

        swing_idx = swing_high_a1 if sweeper == 0 else swing_high_a2
        sweep_values = high_a1 if sweeper == 0 else high_a2
        if not _current_is_extreme(sweep_values, swing_idx, current_idx, is_bullish=False):
            continue

        reference_timestamp = (
            index[swing_high_a1] if swing_high_a1 <= swing_high_a2 else index[swing_high_a2]
        )
        signal = _build_signal(
            signal_type="Bearish Swing SMT",
            timestamp=index[current_idx],
            sweeping_asset=asset_names[sweeper],
            failing_asset=asset_names[1 - sweeper],
            reference_price=high_a1[swing_high_a1] if sweeper == 0 else high_a2[swing_high_a2],
            invalidation_level=high_a2[swing_high_a2] if sweeper == 0 else high_a1[swing_high_a1],
            reference_timestamp=reference_timestamp,
        )
        rows.append(_event_row(signal))

    return rows


def _scan_fvg_events(
    df_a1: pd.DataFrame,
    df_a2: pd.DataFrame,
    lookback_period: int,
    asset_names,
) -> list[dict]:
    index = df_a1.index
    high_a1 = df_a1["High"].to_numpy()
    low_a1 = df_a1["Low"].to_numpy()
    high_a2 = df_a2["High"].to_numpy()
    low_a2 = df_a2["Low"].to_numpy()

    def find_recent_valid_fvg(high: np.ndarray, low: np.ndarray, current_idx: int):
        max_lookback = min(lookback_period, current_idx - 2)
        for i in range(1, max_lookback + 1):
            c1_idx = current_idx - i - 2
            c3_idx = current_idx - i
            candles_after_low = low[c3_idx + 1 : current_idx]
            candles_after_high = high[c3_idx + 1 : current_idx]

            c1_high = high[c1_idx]
            c1_low = low[c1_idx]
            c3_high = high[c3_idx]
            c3_low = low[c3_idx]

            if c1_high < c3_low:
                original_bottom = c1_high
                original_top = c3_low

                if len(candles_after_low) > 0:
                    if (candles_after_low < original_bottom).any():
                        continue

                    lows_in_gap = candles_after_low[candles_after_low < original_top]
                    current_top = lows_in_gap.min() if len(lows_in_gap) > 0 else original_top
                else:
                    current_top = original_top

                return {
                    "type": "bullish",
                    "bottom": original_bottom,
                    "top": current_top,
                    "idx": c1_idx,
                }

            if c1_low > c3_high:
                original_bottom = c3_high
                original_top = c1_low

                if len(candles_after_high) > 0:
                    if (candles_after_high > original_top).any():
                        continue

                    highs_in_gap = candles_after_high[candles_after_high > original_bottom]
                    current_bottom = highs_in_gap.max() if len(highs_in_gap) > 0 else original_bottom
                else:
                    current_bottom = original_bottom

                return {
                    "type": "bearish",
                    "bottom": current_bottom,
                    "top": original_top,
                    "idx": c1_idx,
                }

        return None

    rows = []
    for current_idx in range(lookback_period, len(df_a1)):
        fvg_a1 = find_recent_valid_fvg(high_a1, low_a1, current_idx)
        fvg_a2 = find_recent_valid_fvg(high_a2, low_a2, current_idx)

        if fvg_a1 is None or fvg_a2 is None or fvg_a1["type"] != fvg_a2["type"]:
            continue

        if abs(fvg_a1["idx"] - fvg_a2["idx"]) > 1:
            continue

        is_bullish = fvg_a1["type"] == "bullish"
        if is_bullish:
            ref_price_a1 = fvg_a1["top"]
            ref_price_a2 = fvg_a2["top"]
            current_value_a1 = low_a1[current_idx]
            current_value_a2 = low_a2[current_idx]
        else:
            ref_price_a1 = fvg_a1["bottom"]
            ref_price_a2 = fvg_a2["bottom"]
            current_value_a1 = high_a1[current_idx]
            current_value_a2 = high_a2[current_idx]

        sweeper = _check_divergence(
            current_value_a1,
            current_value_a2,
            ref_price_a1,
            ref_price_a2,
            is_bullish,
        )
        if sweeper is None:
            continue

        reference_timestamp = (
            index[fvg_a1["idx"]] if fvg_a1["idx"] <= fvg_a2["idx"] else index[fvg_a2["idx"]]
        )
        signal = _build_signal(
            signal_type=f"{'Bullish' if is_bullish else 'Bearish'} FVG SMT",
            timestamp=index[current_idx],
            sweeping_asset=asset_names[sweeper],
            failing_asset=asset_names[1 - sweeper],
            reference_price=ref_price_a1 if sweeper == 0 else ref_price_a2,
            invalidation_level=ref_price_a2 if sweeper == 0 else ref_price_a1,
            reference_timestamp=reference_timestamp,
        )
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
    if enable_fvg:
        rows.extend(_scan_fvg_events(df_a1, df_a2, lookback_period, asset_names))

    if not rows:
        return _empty_events()

    events = pd.DataFrame(rows).sort_values(
        by=["created_ts", "signal_type"],
        kind="stable",
    ).reset_index(drop=True)

    return _resolve_broken_ts(events, df_a1, df_a2, asset_names)[EVENT_COLUMNS]
