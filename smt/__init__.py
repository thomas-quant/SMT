"""
smt — Smart Money Technique divergence detection toolkit.

Typical usage:

    from smt import SMTManager

    manager = SMTManager(timeframe="5m", asset_names=("ES", "NQ"))
    result = manager.update(df_es, df_nq, high, low, close, ts)

For lower-level access:

    from smt import check_micro_smt, check_swing_smt, check_fvg_smt
    from smt import SMTRegistry, SMTBreak
"""

from .detector import check_micro_smt, check_swing_smt, check_fvg_smt
from .manager import SMTManager
from .registry import SMTRegistry
from .break_tracker import SMTBreak

__all__ = [
    "SMTManager",
    "SMTRegistry",
    "SMTBreak",
    "check_micro_smt",
    "check_swing_smt",
    "check_fvg_smt",
]
