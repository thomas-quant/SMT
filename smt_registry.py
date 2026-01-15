# smt_registry.py
"""
SMT Registry - Manages lifecycle of SMT signals.

This module provides a single source of truth for SMT state management.
It tracks SMTs from creation (active) through invalidation (broken).

Responsibilities:
- Assign unique IDs to each SMT
- Store SMT metadata from smt_detector
- Track SMT state transitions (active → broken)
- Provide query interface for active/all SMTs

Design:
- In-memory only (no persistence)
- No scoring, weighting, or ranking logic
- No price logic (reacts to events only)
- Stateless regarding market data
"""

import uuid
from typing import Any, Dict, List, Optional


class SMTRegistry:
    """
    Central registry for SMT lifecycle management.
    
    Each SMT entry contains:
    - id: Unique identifier (UUID)
    - status: "active" or "broken"
    - timeframe: Chart timeframe string (e.g., "1m", "5m", "1h")
    - created_ts: Timestamp when SMT was detected
    - broken_ts: Timestamp when SMT was invalidated (None if active)
    - signal: Raw detector output dictionary
    """
    
    def __init__(self):
        self._smts: Dict[str, Dict[str, Any]] = {}
    
    def add_smt(self, signal: Dict[str, Any], timeframe: str) -> str:
        """
        Register a new SMT signal from the detector.
        
        Args:
            signal: Raw output from smt_detector (must contain 'timestamp')
            timeframe: Chart timeframe string (e.g., "1m", "5m", "1h")
        
        Returns:
            Unique SMT ID (UUID string)
        """
        smt_id = str(uuid.uuid4())
        
        self._smts[smt_id] = {
            "id": smt_id,
            "status": "active",
            "timeframe": timeframe,
            "created_ts": signal.get("timestamp"),
            "broken_ts": None,
            "signal": signal
        }
        
        return smt_id
    
    def mark_broken(self, smt_id: str, broken_ts: Any = None) -> bool:
        """
        Mark an SMT as broken (invalidated).
        
        Args:
            smt_id: ID of the SMT to mark as broken
            broken_ts: Timestamp when the SMT was invalidated
        
        Returns:
            True if SMT was found and updated, False otherwise
        """
        if smt_id not in self._smts:
            return False
        
        self._smts[smt_id]["status"] = "broken"
        self._smts[smt_id]["broken_ts"] = broken_ts
        
        return True
    
    def get_active_smts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all SMTs with status "active".
        
        Returns:
            Dictionary of active SMTs keyed by ID
        """
        return {
            smt_id: smt 
            for smt_id, smt in self._smts.items() 
            if smt["status"] == "active"
        }
    
    def get_all_smts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all SMTs regardless of status.
        
        Returns:
            Dictionary of all SMTs keyed by ID
        """
        return dict(self._smts)
    
    def get_smt(self, smt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific SMT by ID.
        
        Args:
            smt_id: ID of the SMT to retrieve
        
        Returns:
            SMT entry dictionary or None if not found
        """
        return self._smts.get(smt_id)
    
    def get_broken_smts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all SMTs with status "broken".
        
        Returns:
            Dictionary of broken SMTs keyed by ID
        """
        return {
            smt_id: smt 
            for smt_id, smt in self._smts.items() 
            if smt["status"] == "broken"
        }
    
    def get_smts_by_timeframe(self, timeframe: str) -> Dict[str, Dict[str, Any]]:
        """
        Get all SMTs for a specific timeframe.
        
        Args:
            timeframe: Timeframe string to filter by
        
        Returns:
            Dictionary of SMTs matching the timeframe
        """
        return {
            smt_id: smt 
            for smt_id, smt in self._smts.items() 
            if smt["timeframe"] == timeframe
        }
    
    def clear(self) -> None:
        """Clear all SMTs from the registry."""
        self._smts.clear()
    
    def __len__(self) -> int:
        """Return total number of SMTs in registry."""
        return len(self._smts)
    
    def active_count(self) -> int:
        """Return count of active SMTs."""
        return sum(1 for smt in self._smts.values() if smt["status"] == "active")
    
    def broken_count(self) -> int:
        """Return count of broken SMTs."""
        return sum(1 for smt in self._smts.values() if smt["status"] == "broken")

