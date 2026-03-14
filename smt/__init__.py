"""
smt — Smart Money Technique divergence detection toolkit.

Typical usage:

    from smt import SMTManager

    manager = SMTManager(timeframe="5m", asset_names=("ES", "NQ"))
    result = manager.update(df_es, df_nq)

For lower-level access:

    from smt import check_micro_smt, check_swing_smt, check_fvg_smt
    from smt import SMTRegistry, SMTBreak
"""

from .detector import check_micro_smt, check_swing_smt, check_fvg_smt
from .historical import scan_smts_historical
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
    "scan_smts_historical",
]
