"""Live SMT lifecycle orchestration."""

from typing import Any, Dict, List, Tuple

import pandas as pd

from .break_tracker import SMTBreak
from .detector import check_micro_smt, check_swing_smt, check_fvg_smt
from .registry import SMTRegistry


class SMTManager:
    """
    Orchestrates the SMT detection → tracking → invalidation lifecycle.

    This class coordinates:
    - SMTRegistry: stores all SMTs and their state
    - SMTBreak: monitors invalidation levels for active SMTs
    - Detectors: functions that identify SMT patterns

    The manager does NOT contain price logic; it delegates to:
    - Detectors for pattern recognition
    - SMTBreak for level monitoring
    """

    def __init__(
        self,
        timeframe: str,
        asset_names: Tuple[str, str] = ("A1", "A2"),
        lookback_period: int = 20,
        enable_micro: bool = True,
        enable_swing: bool = True,
        enable_fvg: bool = True,
    ):
        self.timeframe = timeframe
        self.asset_names = asset_names
        self.lookback_period = lookback_period

        self.enable_micro = enable_micro
        self.enable_swing = enable_swing
        self.enable_fvg = enable_fvg

        self.registry = SMTRegistry()
        self.break_tracker = SMTBreak()

    def update(
        self,
        df_a1: pd.DataFrame,
        df_a2: pd.DataFrame,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a candle update through the complete SMT lifecycle.

        This is the main entry point called on each candle close.

        Args:
            df_a1: DataFrame for asset 1 (with current candle included)
            df_a2: DataFrame for asset 2 (with current candle included)

        Returns:
            Dictionary with:
            - "new_smts": List of newly detected SMT entries
            - "broken_smts": List of SMTs invalidated this candle
        """
        result = {"new_smts": [], "broken_smts": []}

        new_signals = self._run_detectors(df_a1, df_a2)

        for signal in new_signals:
            smt_id = self.registry.add_smt(signal, self.timeframe)
            self.break_tracker.add(signal, entry_id=smt_id)

            smt_entry = self.registry.get_smt(smt_id)
            result["new_smts"].append(smt_entry)

        broken_list = []
        for asset, df in zip(self.asset_names, (df_a1, df_a2)):
            latest = df.iloc[-1]
            broken_list.extend(
                self.break_tracker.update_asset(
                    asset=asset,
                    high=float(latest["High"]),
                    low=float(latest["Low"]),
                    close=float(latest["Close"]),
                    ts=df.index[-1],
                )
            )

        for broken in broken_list:
            smt_id = broken["id"]
            self.registry.mark_broken(smt_id, broken_ts=broken["ts"])

            smt_entry = self.registry.get_smt(smt_id)
            if smt_entry:
                result["broken_smts"].append(smt_entry)

        return result

    def _run_detectors(
        self,
        df_a1: pd.DataFrame,
        df_a2: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Run enabled detectors and collect signals."""
        signals = []

        if self.enable_micro:
            signal = check_micro_smt(df_a1, df_a2, self.asset_names)
            if signal:
                signals.append(signal)

        if self.enable_swing:
            signal = check_swing_smt(
                df_a1,
                df_a2,
                self.lookback_period,
                self.asset_names,
            )
            if signal:
                signals.append(signal)

        if self.enable_fvg:
            signal = check_fvg_smt(
                df_a1,
                df_a2,
                self.lookback_period,
                self.asset_names,
            )
            if signal:
                signals.append(signal)

        return signals

    def get_active_smts(self) -> Dict[str, Dict[str, Any]]:
        """Get all active SMTs from registry."""
        return self.registry.get_active_smts()

    def get_all_smts(self) -> Dict[str, Dict[str, Any]]:
        """Get all SMTs from registry."""
        return self.registry.get_all_smts()

    def get_broken_smts(self) -> Dict[str, Dict[str, Any]]:
        """Get all broken SMTs from registry."""
        return self.registry.get_broken_smts()

    def clear(self) -> None:
        """Clear all state (registry and break tracker)."""
        self.registry.clear()
        self.break_tracker.clear()

    @property
    def active_count(self) -> int:
        """Count of active SMTs."""
        return self.registry.active_count

    @property
    def broken_count(self) -> int:
        """Count of broken SMTs."""
        return self.registry.broken_count
