from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment


class PineconeProfile:
    name = "pinecone"
    engine_overhead_factor = 0.05

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        _ = scenario
        return EngineAdjustment(
            vector_storage_bytes=base.raw_vector_bytes,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=0,
            vector_ram_bytes=base.raw_vector_bytes,
            payload_ram_bytes=base.record_payload_bytes,
            index_ram_bytes=0,
            warnings=[
                (
                    "Pinecone profile reports storage and RU-like proxies only; "
                    "it does not calculate vendor pricing."
                ),
                (
                    "Pinecone managed infrastructure may include implementation "
                    "details not represented by this analytical estimate."
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
            "Pinecone profile:\n"
            "- uses record-size style storage estimation for dense, sparse, and hybrid records;\n"
            "- reports storage-oriented estimates, not vendor pricing;\n"
            "- managed-service implementation details are intentionally abstracted."
        )
