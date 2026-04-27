from vector_db_sizer.estimators.storage import (
    estimate_engine_overhead,
    estimate_final_disk,
    estimate_replicated_storage,
    estimate_wal_snapshot,
)


def test_engine_overhead() -> None:
    assert estimate_engine_overhead(1000, 200, 300, 0.10) == 150


def test_replicas() -> None:
    assert estimate_replicated_storage(1000, 3) == 3000


def test_wal_snapshot() -> None:
    wal, snapshot = estimate_wal_snapshot(1000, 0.10, 0.20)
    assert wal == 100
    assert snapshot == 200


def test_safety_applied_last() -> None:
    assert estimate_final_disk(1000, 100, 200, 1.25) == 1625
