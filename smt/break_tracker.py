# break_tracker.py
"""
SMT Break Tracker - Monitors invalidation levels from SMT detector signals.
"""

from typing import Any, Dict, List, Optional


class SMTBreak:
    """Tracks SMT invalidation levels and detects when price breaks them."""

    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.last_price: Optional[float] = None

    def add(self, signal: Dict[str, Any], entry_id: Optional[str] = None) -> str:
        """Add invalidation level from an SMT detector signal."""
        if entry_id is None:
            entry_id = f"{signal.get('signal_type', '')}_{signal.get('timestamp', 'unknown')}"

        self._entries[entry_id] = {
            'level': float(signal['invalidation_level']),
            'signal': signal
        }
        return entry_id

    def remove(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def clear(self) -> None:
        self._entries.clear()
        self.last_price = None

    def update_candle(self, high: float, low: float, close: float, ts: Any = None) -> List[Dict[str, Any]]:
        """Check for broken levels. Returns list of invalidated signals."""
        prev = self.last_price
        self.last_price = close

        if prev is None:
            return []

        broken = []
        for sid, e in list(self._entries.items()):
            level = e['level']

            # Level above prev = break up, level below prev = break down
            crossed = (
                (level >= prev and high >= level) or
                (level <= prev and low <= level)
            )

            if crossed:
                broken.append({'id': sid, 'level': level, 'price': close, 'ts': ts, 'signal': e['signal']})
                del self._entries[sid]

        return broken
