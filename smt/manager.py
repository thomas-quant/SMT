# manager.py
"""
SMT Manager - Glue code orchestrating detector, registry, and break tracker.

This module demonstrates the complete SMT lifecycle:
1. Detection: detector emits signals
2. Registration: signals are stored in registry
3. Tracking: invalidation levels are monitored by break_tracker
4. Invalidation: broken SMTs are marked in registry

Integration Flow (per candle update):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Call detector(s) with latest candle data                    │
│    └─> May return zero or more SMT signals                     │
│                                                                 │
│ 2. For each new SMT signal:                                    │
│    ├─> Add to registry (assigns UUID, status="active")         │
│    └─> Register invalidation_level with SMTBreak               │
│                                                                 │
│ 3. Call SMTBreak.update_candle(high, low, close, ts)           │
│    └─> Returns list of broken SMTs (level was crossed)         │
│                                                                 │
│ 4. For each broken SMT:                                        │
│    └─> Update registry status to "broken"                      │
└─────────────────────────────────────────────────────────────────┘

Usage:
    manager = SMTManager(timeframe="5m")

    # On each candle close:
    result = manager.update(df_asset1, df_asset2, high, low, close, ts)

    # result contains:
    # - new_smts: list of newly detected SMTs
    # - broken_smts: list of SMTs that were invalidated this candle
"""

import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from .registry import SMTRegistry
from .break_tracker import SMTBreak
from .detector import check_micro_smt, check_swing_smt, check_fvg_smt


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
        enable_fvg: bool = True
    ):
        """
        Initialize the SMT Manager.

        Args:
            timeframe: Chart timeframe string (e.g., "1m", "5m", "1h")
            asset_names: Tuple of asset names for detector output
            lookback_period: Lookback period for swing/FVG detection
            enable_micro: Enable micro SMT detection
            enable_swing: Enable swing SMT detection
            enable_fvg: Enable FVG SMT detection
        """
        self.timeframe = timeframe
        self.asset_names = asset_names
        self.lookback_period = lookback_period

        # Detection flags
        self.enable_micro = enable_micro
        self.enable_swing = enable_swing
        self.enable_fvg = enable_fvg

        # Core components
        self.registry = SMTRegistry()
        self.break_tracker = SMTBreak()

    def update(
        self,
        df_a1: pd.DataFrame,
        df_a2: pd.DataFrame,
        high: float,
        low: float,
        close: float,
        ts: Any = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a candle update through the complete SMT lifecycle.

        This is the main entry point called on each candle close.

        Args:
            df_a1: DataFrame for asset 1 (with current candle included)
            df_a2: DataFrame for asset 2 (with current candle included)
            high: High price of current candle (for break detection)
            low: Low price of current candle (for break detection)
            close: Close price of current candle (for break detection)
            ts: Current timestamp

        Returns:
            Dictionary with:
            - "new_smts": List of newly detected SMT entries
            - "broken_smts": List of SMTs invalidated this candle
        """
        result = {
            "new_smts": [],
            "broken_smts": []
        }

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Run detectors to find new SMT signals
        # ═══════════════════════════════════════════════════════════════
        new_signals = self._run_detectors(df_a1, df_a2)

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Register new SMTs and set up break tracking
        # ═══════════════════════════════════════════════════════════════
        for signal in new_signals:
            # Add to registry (assigns UUID, status="active")
            smt_id = self.registry.add_smt(signal, self.timeframe)

            # Register invalidation level with break tracker using the registry ID
            self.break_tracker.add(signal, entry_id=smt_id)

            # Get the full SMT entry for output
            smt_entry = self.registry.get_smt(smt_id)
            result["new_smts"].append(smt_entry)

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Check for broken invalidation levels
        # ═══════════════════════════════════════════════════════════════
        broken_list = self.break_tracker.update_candle(high, low, close, ts)

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Update registry for broken SMTs
        # ═══════════════════════════════════════════════════════════════
        for broken in broken_list:
            smt_id = broken["id"]

            # Mark as broken in registry
            self.registry.mark_broken(smt_id, broken_ts=ts)

            # Get the updated SMT entry for output
            smt_entry = self.registry.get_smt(smt_id)
            if smt_entry:
                result["broken_smts"].append(smt_entry)

        return result

    def _run_detectors(
        self,
        df_a1: pd.DataFrame,
        df_a2: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Run enabled detectors and collect signals."""
        signals = []

        # Micro SMT (2-candle pattern)
        if self.enable_micro:
            signal = check_micro_smt(df_a1, df_a2, self.asset_names)
            if signal:
                signals.append(signal)

        # Swing SMT (swing point divergence)
        if self.enable_swing:
            signal = check_swing_smt(
                df_a1, df_a2,
                self.lookback_period,
                self.asset_names
            )
            if signal:
                signals.append(signal)

        # FVG SMT (fair value gap divergence)
        if self.enable_fvg:
            signal = check_fvg_smt(
                df_a1, df_a2,
                self.lookback_period,
                self.asset_names
            )
            if signal:
                signals.append(signal)

        return signals

    # ═══════════════════════════════════════════════════════════════════
    # Registry passthrough methods for convenience
    # ═══════════════════════════════════════════════════════════════════

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
