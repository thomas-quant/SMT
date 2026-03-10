from smt import SMTRegistry


def test_registry_read_methods_return_copies():
    registry = SMTRegistry()
    signal = {
        "signal_type": "Bearish Micro SMT",
        "timestamp": "2024-01-01 09:35:00",
        "sweeping_asset": "ES",
        "failing_asset": "NQ",
        "reference_price": 100.0,
        "invalidation_level": 105.0,
        "invalidation_asset": "NQ",
        "invalidation_direction": "above",
    }

    smt_id = registry.add_smt(signal, timeframe="5m")
    registry.mark_broken(smt_id, broken_ts="2024-01-01 09:40:00")

    single = registry.get_smt(smt_id)
    all_smts = registry.get_all_smts()
    active = registry.get_active_smts()
    broken = registry.get_broken_smts()

    single["status"] = "active"
    single["signal"]["signal_type"] = "mutated"
    all_smts[smt_id]["broken_ts"] = None
    broken[smt_id]["signal"]["invalidation_level"] = -1.0

    stored = registry.get_smt(smt_id)
    assert stored["status"] == "broken"
    assert stored["broken_ts"] == "2024-01-01 09:40:00"
    assert stored["signal"]["signal_type"] == "Bearish Micro SMT"
    assert stored["signal"]["invalidation_level"] == 105.0
    assert active == {}
