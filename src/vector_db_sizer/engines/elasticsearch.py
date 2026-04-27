import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment


class ElasticsearchProfile:
    name = "elasticsearch"
    engine_overhead_factor = 0.25

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        raw = base.raw_vector_bytes
        q = scenario.index.quantization
        quantized = 0
        if q == "scalar_int8":
            quantized = math.ceil(raw * 0.25)
        elif q == "int4":
            quantized = math.ceil(raw * 0.125)
        elif q in {"binary", "bbq"}:
            quantized = math.ceil(raw * 0.03125)

        extra = math.ceil((raw + base.record_payload_bytes + base.index_disk_bytes) * 0.15)
        source_dup = raw if scenario.storage.store_vectors_in_source else 0
        vector_ram = quantized if q != "none" else raw
        warnings = [
            (
                "Elasticsearch stores vectors internally for vector search "
                "and may optionally store original vectors in _source."
            ),
        ]
        if scenario.storage.store_vectors_in_source:
            warnings.insert(
                1,
                "Enabling vector storage in _source can substantially increase disk usage.",
            )
        if q != "none":
            warnings.append(
                "Elasticsearch quantization keeps raw float vectors on disk "
                "for reranking, reindexing, and quantization improvements."
            )
        return EngineAdjustment(
            vector_storage_bytes=raw,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=base.index_disk_bytes,
            vector_ram_bytes=vector_ram,
            payload_ram_bytes=base.record_payload_bytes,
            index_ram_bytes=base.index_ram_bytes,
            quantized_vector_bytes=quantized,
            source_vector_duplicate_bytes=source_dup,
            extra_disk_bytes=extra,
            warnings=warnings,
        )

    def explain(self) -> str:
        return (
            "Elasticsearch profile:\n"
            "- models Lucene-like segment overhead approximately;\n"
            "- dense vectors are stored internally for vector search;\n"
            "- storing vectors in _source duplicates vector storage;\n"
            "- quantization can reduce RAM but raw vectors may remain "
            "on disk for reranking and reindexing."
        )
