import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment
from vector_db_sizer.engines.common import default_quantized_bytes


class QdrantProfile:
    name = "qdrant"
    engine_overhead_factor = 0.15

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        st = scenario.storage
        vector_ram = (
            math.ceil(base.raw_vector_bytes * 0.10)
            if st.vectors_on_disk
            else base.raw_vector_bytes
        )
        payload_ram = (
            math.ceil(base.record_payload_bytes * 0.10)
            if st.payload_on_disk
            else base.record_payload_bytes
        )
        index_ram = (
            math.ceil(base.index_ram_bytes * 0.15)
            if scenario.database.index_type == "hnsw" and st.hnsw_on_disk
            else base.index_ram_bytes
        )
        warnings = [
            (
                "Qdrant separates vector storage, payload storage, and HNSW storage. "
                "Disk and RAM estimates should be reviewed separately."
            ),
            (
                "Qdrant payload-on-disk and vector-on-disk settings can reduce RAM "
                "usage but may increase latency."
            ),
        ]
        notes = []
        if not st.store_original_vectors:
            warnings.append(
                "Qdrant profile conservatively keeps raw vectors in disk estimates; "
                "store_original_vectors=false is treated as an advisory scenario flag in this MVP."
            )
        quantized = 0
        if scenario.index.quantization != "none":
            quantized = default_quantized_bytes(scenario, base.n_records)
            if st.store_original_vectors:
                warnings.append(
                    "Qdrant-like quantization profile stores quantized vectors "
                    "in addition to original vectors."
                )
            elif not st.store_quantized_vectors:
                warnings.append(
                    "Disabling original vectors may affect reranking/reindexing/migration behavior."
                )
        if st.payload_on_disk:
            notes.append(
                "Qdrant profile uses reduced RAM contribution for payload "
                "when payload_on_disk=true."
            )
        return EngineAdjustment(
            vector_storage_bytes=base.raw_vector_bytes,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=base.index_disk_bytes,
            vector_ram_bytes=vector_ram,
            payload_ram_bytes=payload_ram,
            index_ram_bytes=index_ram,
            quantized_vector_bytes=quantized,
            warnings=warnings,
            notes=notes,
        )

    def explain(self) -> str:
        return (
            "Qdrant profile:\n"
            "- separates vector storage, payload storage, and HNSW storage;\n"
            "- supports on-disk vectors, on-disk payload, and on-disk "
            "HNSW-style memory reductions;\n"
            "- RAM and disk estimates should be interpreted separately."
        )
