from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment
from vector_db_sizer.engines.common import default_quantized_bytes


class GenericProfile:
    name = "generic"
    engine_overhead_factor = 0.10

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        quantized = (
            default_quantized_bytes(scenario, base.n_records)
            if scenario.storage.store_quantized_vectors
            else 0
        )
        return EngineAdjustment(
            vector_storage_bytes=base.raw_vector_bytes,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=base.index_disk_bytes,
            vector_ram_bytes=base.raw_vector_bytes,
            payload_ram_bytes=base.record_payload_bytes,
            index_ram_bytes=base.index_ram_bytes,
            quantized_vector_bytes=quantized,
            warnings=[
                "Generic engine profile uses approximate storage overhead.",
                "This is an analytical estimate, not a production sizing guarantee.",
            ],
        )

    def explain(self) -> str:
        return (
            "Generic profile:\n"
            "- uses approximate storage and RAM overhead;\n"
            "- does not model database-specific storage behavior;\n"
            "- useful as a baseline scenario."
        )
