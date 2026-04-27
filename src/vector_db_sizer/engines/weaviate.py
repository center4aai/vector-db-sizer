import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment


class WeaviateProfile:
    name = "weaviate"
    engine_overhead_factor = 0.20

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        index_ram = base.index_ram_bytes
        if scenario.database.index_type == "hnsw":
            index_ram = max(base.index_ram_bytes, base.raw_vector_bytes)
        return EngineAdjustment(
            vector_storage_bytes=base.raw_vector_bytes,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=base.index_disk_bytes,
            vector_ram_bytes=base.raw_vector_bytes,
            payload_ram_bytes=math.ceil(base.record_payload_bytes * 0.25),
            index_ram_bytes=index_ram,
            extra_disk_bytes=math.ceil(base.record_payload_bytes * 0.10),
            warnings=[
                (
                    "Weaviate storage depends on object properties, inverted index settings, "
                    "vector index type, and quantization. Use this estimate as scenario "
                    "comparison, not exact sizing."
                ),
                (
                    "Weaviate HNSW memory usage is primarily driven by the number "
                    "of vectors and vector index configuration."
                ),
            ],
            confidence={
                "raw_vectors": "high",
                "record_payload": "medium",
                "index": "low",
                "engine_overhead": "low",
                "final_disk": "medium",
                "ram": "low",
            },
        )

    def explain(self) -> str:
        return (
            "Weaviate profile:\n"
            "- models object, inverted index, and vector index overhead approximately;\n"
            "- HNSW memory usage is primarily driven by vector count and "
            "vector index settings;\n"
            "- use estimates for scenario comparison, not exact production sizing."
        )
