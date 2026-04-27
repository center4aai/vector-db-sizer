from typing import Any, Literal

from pydantic import BaseModel, Field


class ScenarioSummary(BaseModel):
    name: str
    engine: str
    index_type: str
    source_type: str
    embedding_kind: str
    dimensions: int | None
    dtype: str | None
    n_records: int


class StorageBytes(BaseModel):
    raw_vectors: int
    quantized_vectors: int = 0
    record_payload: int
    index_disk: int
    source_vector_duplicates: int = 0
    engine_overhead: int
    base_storage: int
    replicated_storage: int
    wal: int
    snapshot: int
    final_disk: int


class RamBytes(BaseModel):
    vectors: int
    quantized_vectors: int = 0
    index: int
    payload: int
    engine_overhead: int
    final_ram: int


class Confidence(BaseModel):
    raw_vectors: Literal["high", "medium", "low"]
    record_payload: Literal["high", "medium", "low"]
    index: Literal["high", "medium", "low"]
    engine_overhead: Literal["high", "medium", "low"]
    final_disk: Literal["high", "medium", "low"]
    ram: Literal["high", "medium", "low"]


class EstimateResult(BaseModel):
    scenario: ScenarioSummary
    storage_bytes: StorageBytes
    ram_bytes: RamBytes
    warnings: list[str]
    notes: list[str] = Field(default_factory=list)
    confidence: Confidence
    engine_metrics: dict[str, Any] | None = None


class MultiEstimateResult(BaseModel):
    results: list[EstimateResult]
