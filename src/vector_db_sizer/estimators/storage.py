import math


def estimate_engine_overhead(
    raw_vectors: int, record_payload: int, index_disk: int, factor: float
) -> int:
    return math.ceil((raw_vectors + record_payload + index_disk) * factor)


def estimate_replicated_storage(base_storage: int, replicas: int) -> int:
    return base_storage * replicas


def estimate_wal_snapshot(
    replicated_storage: int, wal_factor: float, snapshot_factor: float
) -> tuple[int, int]:
    wal = math.ceil(replicated_storage * wal_factor)
    snapshot = math.ceil(replicated_storage * snapshot_factor)
    return wal, snapshot


def estimate_final_disk(
    replicated_storage: int, wal: int, snapshot: int, safety_factor: float
) -> int:
    return math.ceil((replicated_storage + wal + snapshot) * safety_factor)
