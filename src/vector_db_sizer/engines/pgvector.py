import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.engines.base import BaseEstimateComponents, EngineAdjustment


class PgvectorProfile:
    name = "pgvector"
    engine_overhead_factor = 0.20

    def _raw_bytes(self, scenario: ScenarioConfig, n_records: int, fallback: int) -> int:
        emb = scenario.embedding
        dense = 0
        sparse = 0
        if emb.kind in {"dense", "hybrid"}:
            dim = emb.dimensions or 0
            if emb.dtype == "float32":
                dense = n_records * (4 * dim + 8)
            elif emb.dtype in {"float16", "bfloat16"}:
                dense = n_records * (2 * dim + 8)
            elif emb.dtype == "binary":
                dense = n_records * (math.ceil(dim / 8) + 8)
            else:
                dense = fallback
        if emb.kind in {"sparse", "hybrid"}:
            sparse = n_records * (8 * (emb.sparse_non_zero_avg or 0) + 16)
        if emb.kind == "dense":
            return dense
        if emb.kind == "sparse":
            return sparse
        return dense + sparse

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment:
        raw = self._raw_bytes(scenario, base.n_records, base.raw_vector_bytes)
        warnings = [
            (
                "pgvector raw vector storage can be estimated with type-specific "
                "formulas, but table and index size should be validated after a pilot load."
            ),
            "pgvector profile uses an approximate table/storage overhead factor.",
        ]
        if (
            scenario.embedding.kind in {"dense", "hybrid"}
            and scenario.embedding.dtype in {"int8", "uint8"}
        ):
            warnings.append(
                "pgvector profile has no type-specific int8/uint8 storage formula in this MVP; "
                "using the generic dense vector byte estimate for this component."
            )
        return EngineAdjustment(
            vector_storage_bytes=raw,
            record_payload_storage_bytes=base.record_payload_bytes,
            index_disk_bytes=base.index_disk_bytes,
            vector_ram_bytes=raw,
            payload_ram_bytes=base.record_payload_bytes,
            index_ram_bytes=base.index_ram_bytes,
            warnings=warnings,
            notes=[
                (
                    "For exact PostgreSQL table and index size, validate with "
                    "pg_relation_size after loading a representative sample."
                )
            ],
        )

    def explain(self) -> str:
        return (
            "pgvector profile:\n"
            "- uses type-specific vector storage formulas for vector, halfvec, "
            "bit, and sparsevec-like estimates;\n"
            "- uses approximate table and index overhead;\n"
            "- validate production estimates with pg_relation_size after a "
            "representative pilot load."
        )
