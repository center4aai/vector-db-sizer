from dataclasses import dataclass, field
from typing import Literal, Protocol

from vector_db_sizer.config import ScenarioConfig

ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass
class BaseEstimateComponents:
    n_records: int
    raw_vector_bytes: int
    record_payload_bytes: int
    index_disk_bytes: int
    index_ram_bytes: int


@dataclass
class EngineAdjustment:
    vector_storage_bytes: int
    record_payload_storage_bytes: int
    index_disk_bytes: int
    vector_ram_bytes: int
    payload_ram_bytes: int
    index_ram_bytes: int
    quantized_vector_bytes: int = 0
    quantized_vector_ram_bytes: int = 0
    source_vector_duplicate_bytes: int = 0
    extra_disk_bytes: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: dict[str, ConfidenceLevel] = field(
        default_factory=lambda: {
            "raw_vectors": "high",
            "record_payload": "medium",
            "index": "medium",
            "engine_overhead": "low",
            "final_disk": "medium",
            "ram": "medium",
        }
    )
    engine_metrics: dict | None = None


class EngineProfile(Protocol):
    name: str
    engine_overhead_factor: float

    def apply(self, scenario: ScenarioConfig, base: BaseEstimateComponents) -> EngineAdjustment: ...

    def explain(self) -> str: ...
