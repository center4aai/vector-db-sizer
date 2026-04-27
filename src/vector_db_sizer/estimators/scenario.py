import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents
from vector_db_sizer.engines.registry import get_engine_profile
from vector_db_sizer.estimators.datasets import estimate_record_count
from vector_db_sizer.estimators.indexes import estimate_index_bytes
from vector_db_sizer.estimators.records import estimate_record_payload_bytes
from vector_db_sizer.estimators.storage import (
    estimate_engine_overhead,
    estimate_final_disk,
    estimate_replicated_storage,
    estimate_wal_snapshot,
)
from vector_db_sizer.estimators.vectors import estimate_vector_bytes
from vector_db_sizer.models import (
    Confidence,
    EstimateResult,
    RamBytes,
    ScenarioSummary,
    StorageBytes,
)


def estimate_scenario(config: ScenarioConfig) -> EstimateResult:
    n_records = estimate_record_count(config.dataset)
    raw_vectors = estimate_vector_bytes(n_records, config.embedding)
    record_payload = estimate_record_payload_bytes(n_records, config.record)
    index_disk, index_ram, _, index_warnings = estimate_index_bytes(
        index_type=config.database.index_type,
        n_records=n_records,
        index=config.index,
        embedding=config.embedding,
        raw_vector_bytes=raw_vectors,
    )
    base = BaseEstimateComponents(
        n_records=n_records,
        raw_vector_bytes=raw_vectors,
        record_payload_bytes=record_payload,
        index_disk_bytes=index_disk,
        index_ram_bytes=index_ram,
    )
    profile = get_engine_profile(config.database.engine)
    adj = profile.apply(config, base)

    overhead_factor = config.storage.engine_overhead_factor
    if overhead_factor is None:
        overhead_factor = profile.engine_overhead_factor

    pre_overhead_base = (
        adj.vector_storage_bytes
        + adj.quantized_vector_bytes
        + adj.record_payload_storage_bytes
        + adj.index_disk_bytes
        + adj.source_vector_duplicate_bytes
        + adj.extra_disk_bytes
    )
    engine_overhead = estimate_engine_overhead(0, 0, pre_overhead_base, overhead_factor)
    pre_replication = pre_overhead_base + engine_overhead
    replicated_storage = estimate_replicated_storage(pre_replication, config.database.replicas)
    wal, snapshot = estimate_wal_snapshot(
        replicated_storage,
        config.storage.wal_factor,
        config.storage.snapshot_factor,
    )
    final_disk = estimate_final_disk(
        replicated_storage,
        wal,
        snapshot,
        config.storage.safety_factor,
    )
    final_ram = (
        adj.vector_ram_bytes
        + adj.quantized_vector_ram_bytes
        + adj.payload_ram_bytes
        + adj.index_ram_bytes
        + engine_overhead
    )

    engine_metrics = adj.engine_metrics
    if config.database.engine == "pinecone" and engine_metrics is None:
        namespace_size_gb = pre_replication / 1_000_000_000
        engine_metrics = {
            "namespace_size_gb": round(namespace_size_gb, 4),
            "estimated_ru_per_query_proxy": round(max(0.25, namespace_size_gb), 4),
        }

    return EstimateResult(
        scenario=ScenarioSummary(
            name=config.name,
            engine=config.database.engine,
            index_type=config.database.index_type,
            source_type=config.dataset.source_type,
            embedding_kind=config.embedding.kind,
            dimensions=config.embedding.dimensions,
            dtype=config.embedding.dtype,
            n_records=n_records,
        ),
        storage_bytes=StorageBytes(
            raw_vectors=adj.vector_storage_bytes,
            quantized_vectors=adj.quantized_vector_bytes,
            record_payload=adj.record_payload_storage_bytes,
            index_disk=adj.index_disk_bytes,
            source_vector_duplicates=adj.source_vector_duplicate_bytes,
            engine_overhead=engine_overhead,
            base_storage=pre_replication,
            replicated_storage=replicated_storage,
            wal=wal,
            snapshot=snapshot,
            final_disk=final_disk,
        ),
        ram_bytes=RamBytes(
            vectors=adj.vector_ram_bytes,
            quantized_vectors=adj.quantized_vector_ram_bytes,
            index=adj.index_ram_bytes,
            payload=adj.payload_ram_bytes,
            engine_overhead=engine_overhead,
            final_ram=math.ceil(final_ram),
        ),
        warnings=adj.warnings + index_warnings,
        notes=adj.notes,
        confidence=Confidence.model_validate(adj.confidence),
        engine_metrics=engine_metrics,
    )
