import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment
from vector_db_sizer.engines.common import default_quantized_bytes


class MilvusProfile:
    name = "milvus"
    engine_overhead_factor = 0.20

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        quantized = 0
        if scenario.database.index_type == "ivf_pq" or scenario.index.quantization == "product":
            if scenario.storage.store_quantized_vectors:
                quantized = default_quantized_bytes(scenario, base.n_records)
        subtotal = base.raw_vector_bytes + base.record_payload_bytes + base.index_disk_bytes
        extra = math.ceil(subtotal * 0.10)
        warnings = [
            (
                "Milvus sizing depends on segment layout, index type, scalar fields, "
                "and deployment mode. Treat this as a pre-sizing estimate."
            ),
            (
                "Milvus graph-based indexes such as HNSW can require memory for both "
                "graph structure and vector embeddings."
            ),
        ]
        if scenario.database.index_type == "ivf_pq" or scenario.index.quantization == "product":
            warnings.append(
                "PQ/refinement behavior depends on index configuration and deployment mode."
            )
        return EngineAdjustment(
            vector_storage_bytes=base.raw_vector_bytes,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=base.index_disk_bytes,
            vector_ram_bytes=base.raw_vector_bytes,
            payload_ram_bytes=base.record_payload_bytes,
            index_ram_bytes=base.index_ram_bytes,
            quantized_vector_bytes=quantized,
            extra_disk_bytes=extra,
            warnings=warnings,
        )

    def explain(self) -> str:
        return (
            "Milvus profile:\n"
            "- models segment overhead approximately;\n"
            "- HNSW-like graph indexes may require memory for both graph "
            "structure and vector embeddings;\n"
            "- production sizing depends on segment layout, scalar fields, "
            "index type, and deployment mode."
        )
