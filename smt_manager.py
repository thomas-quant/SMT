# smt_manager.py
"""
SMT Manager - Glue code orchestrating detector, registry, and break tracker.

This module demonstrates the complete SMT lifecycle:
1. Detection: smt_detector emits signals
2. Registration: signals are stored in smt_registry
3. Tracking: invalidation levels are monitored by smt_break
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
from typing import Any, Callable, Dict, List, Optional, Tuple

from smt_registry import SMTRegistry
from smt_break import SMTBreak
from smt_detector import check_micro_smt, check_swing_smt, check_fvg_smt


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
        
        # Maps SMT registry IDs to SMTBreak tracking IDs
        # (they are the same in this implementation, but kept explicit)
        self._id_map: Dict[str, str] = {}
    
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
            
            # Register invalidation level with break tracker
            # Use the registry ID as the break tracker ID for consistency
            self.break_tracker.add(signal, id=smt_id)
            self._id_map[smt_id] = smt_id
            
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
            
            # Clean up ID mapping
            self._id_map.pop(smt_id, None)
            
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
        """
        Run enabled detectors and collect signals.
        
        Args:
            df_a1: DataFrame for asset 1
            df_a2: DataFrame for asset 2
        
        Returns:
            List of detected SMT signals
        """
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
        self._id_map.clear()
    
    @property
    def active_count(self) -> int:
        """Count of active SMTs."""
        return self.registry.active_count()
    
    @property
    def broken_count(self) -> int:
        """Count of broken SMTs."""
        return self.registry.broken_count()


# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Example demonstrating the complete SMT lifecycle.
    
    This shows how signals flow through:
    detector → registry → break tracker → registry (status update)
    """
    import numpy as np
    
    # ─────────────────────────────────────────────────────────────────────
    # Setup: Create sample data
    # ─────────────────────────────────────────────────────────────────────
    
    def create_sample_data(n_candles: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create two correlated asset DataFrames for testing."""
        np.random.seed(42)
        
        timestamps = pd.date_range("2024-01-01 09:30", periods=n_candles, freq="5min")
        
        # Asset 1: Base price series
        base_price = 100 + np.cumsum(np.random.randn(n_candles) * 0.5)
        
        # Asset 2: Correlated but with divergence potential
        correlation_noise = np.random.randn(n_candles) * 0.3
        a2_price = base_price * 1.5 + correlation_noise
        
        def make_ohlc(prices):
            highs = prices + np.abs(np.random.randn(len(prices)) * 0.3)
            lows = prices - np.abs(np.random.randn(len(prices)) * 0.3)
            opens = prices + np.random.randn(len(prices)) * 0.1
            closes = prices
            return pd.DataFrame({
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes
            }, index=timestamps)
        
        return make_ohlc(base_price), make_ohlc(a2_price)
    
    # ─────────────────────────────────────────────────────────────────────
    # Example: Complete lifecycle demonstration
    # ─────────────────────────────────────────────────────────────────────
    
    print("=" * 70)
    print("SMT MANAGER - LIFECYCLE DEMONSTRATION")
    print("=" * 70)
    
    # Initialize manager
    manager = SMTManager(
        timeframe="5m",
        asset_names=("ES", "NQ"),
        lookback_period=10,
        enable_micro=True,
        enable_swing=True,
        enable_fvg=True
    )
    
    # Create sample data
    df_es, df_nq = create_sample_data(30)
    
    print(f"\nProcessing {len(df_es)} candles...")
    print("-" * 70)
    
    # Simulate candle-by-candle processing
    for i in range(5, len(df_es)):
        # Get data up to current candle
        df_es_slice = df_es.iloc[:i+1]
        df_nq_slice = df_nq.iloc[:i+1]
        
        # Current candle values (for break detection)
        current = df_es.iloc[i]
        ts = df_es.index[i]
        
        # Process update
        result = manager.update(
            df_es_slice,
            df_nq_slice,
            high=current["High"],
            low=current["Low"],
            close=current["Close"],
            ts=ts
        )
        
        # Report events
        if result["new_smts"]:
            for smt in result["new_smts"]:
                print(f"\n[NEW SMT] {ts}")
                print(f"  ID: {smt['id'][:8]}...")
                print(f"  Type: {smt['signal']['signal_type']}")
                print(f"  Status: {smt['status']}")
                print(f"  Invalidation: {smt['signal']['invalidation_level']:.2f}")
        
        if result["broken_smts"]:
            for smt in result["broken_smts"]:
                print(f"\n[BROKEN SMT] {ts}")
                print(f"  ID: {smt['id'][:8]}...")
                print(f"  Type: {smt['signal']['signal_type']}")
                print(f"  Status: {smt['status']}")
                print(f"  Created: {smt['created_ts']}")
                print(f"  Broken: {smt['broken_ts']}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Final state summary
    # ─────────────────────────────────────────────────────────────────────
    
    print("\n" + "=" * 70)
    print("FINAL STATE SUMMARY")
    print("=" * 70)
    print(f"Total SMTs detected: {len(manager.get_all_smts())}")
    print(f"Active SMTs: {manager.active_count}")
    print(f"Broken SMTs: {manager.broken_count}")
    
    if manager.active_count > 0:
        print("\nActive SMT Details:")
        for smt_id, smt in manager.get_active_smts().items():
            print(f"  - {smt['signal']['signal_type']} @ {smt['created_ts']}")
            print(f"    Invalidation level: {smt['signal']['invalidation_level']:.2f}")

