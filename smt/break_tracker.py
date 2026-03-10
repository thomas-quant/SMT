# break_tracker.py
"""
SMT Break Tracker - Monitors invalidation levels from SMT detector signals.
"""

from typing import Any, Dict, List, Optional


class SMTBreak:
    """Tracks SMT invalidation levels and detects when price breaks them."""

    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._last_prices: Dict[str, float] = {}

    def add(self, signal: Dict[str, Any], entry_id: Optional[str] = None) -> str:
        """Add invalidation level from an SMT detector signal."""
        if entry_id is None:
            entry_id = f"{signal.get('signal_type', '')}_{signal.get('timestamp', 'unknown')}"

        self._entries[entry_id] = {
            "asset": signal["invalidation_asset"],
            "direction": signal["invalidation_direction"],
            "level": float(signal["invalidation_level"]),
            "signal": signal,
        }
        return entry_id

    def remove(self, entry_id: str) -> None:
        self._entries.pop(entry_id, None)

    def clear(self) -> None:
        self._entries.clear()
        self._last_prices.clear()

    def update_asset(
        self,
        asset: str,
        high: float,
        low: float,
        close: float,
        ts: Any = None,
    ) -> List[Dict[str, Any]]:
        """Check whether tracked levels for one asset were invalidated."""
        prev = self._last_prices.get(asset)
        self._last_prices[asset] = close

        if prev is None:
            return []

        broken = []
        for sid, e in list(self._entries.items()):
            if e["asset"] != asset:
                continue

            level = e["level"]
            direction = e["direction"]
            if direction == "above":
                crossed = prev <= level <= high
            elif direction == "below":
                crossed = low <= level <= prev
            else:
                raise ValueError(f"Unsupported invalidation direction: {direction}")

            if crossed:
                broken.append(
                    {
                        "id": sid,
                        "asset": asset,
                        "level": level,
                        "price": close,
                        "ts": ts,
                        "signal": e["signal"],
                    }
                )
                del self._entries[sid]

        return broken
