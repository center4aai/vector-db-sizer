import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment


class OpenSearchProfile:
    name = "opensearch"
    engine_overhead_factor = 0.20

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        idx_disk = base.index_disk_bytes
        idx_ram = base.index_ram_bytes
        n = base.n_records
        dim = scenario.embedding.dimensions or 0
        m = scenario.index.m
        if scenario.database.index_type == "hnsw":
            if scenario.embedding.dtype == "binary":
                idx_disk = math.ceil(n * 1.1 * (math.ceil(dim / 8) + 8 * m))
            else:
                idx_disk = math.ceil(n * 1.1 * (4 * dim + 8 * m))
            idx_ram = idx_disk
        elif scenario.database.index_type == "ivf_flat":
            idx_disk = math.ceil(1.1 * ((dim * n) + (4 * (scenario.index.nlist or 0) * dim)))
            idx_ram = idx_disk

        vector_ram = base.raw_vector_bytes
        if scenario.storage.mode == "on_disk" and scenario.storage.compression_level != "none":
            divisor = int(scenario.storage.compression_level.replace("x", ""))
            vector_ram = math.ceil(vector_ram / divisor)
            idx_ram = math.ceil(idx_ram / divisor)

        warnings = [
            "OpenSearch profile uses documented HNSW/IVF memory formulas where applicable.",
        ]
        if scenario.storage.mode == "on_disk":
            warnings.append(
                "OpenSearch on_disk mode trades lower memory usage for "
                "higher latency and potential rescoring cost."
            )
        return EngineAdjustment(
            vector_storage_bytes=base.raw_vector_bytes,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=idx_disk,
            vector_ram_bytes=vector_ram,
            payload_ram_bytes=base.record_payload_bytes,
            index_ram_bytes=idx_ram,
            warnings=warnings,
        )

    def explain(self) -> str:
        return (
            "OpenSearch profile:\n"
            "- uses documented HNSW/IVF memory estimation formulas where applicable;\n"
            "- supports in-memory and on-disk style estimates;\n"
            "- on-disk mode may reduce memory but can increase latency and rescoring cost."
        )
