from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    source_type: Literal["text", "image", "audio", "video", "tabular", "precomputed_vectors"]

    total_tokens: int | None = None
    chunk_tokens: int | None = None
    chunk_overlap: int = 0
    vectors_per_chunk: int = 1

    item_count: int | None = None
    vectors_per_item: int = 1

    duration_seconds: float | None = None
    segment_seconds: float | None = None
    segment_overlap_seconds: float = 0
    vectors_per_segment: int = 1

    row_count: int | None = None
    vectors_per_row: int = 1

    vector_count: int | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> "DatasetConfig":
        if self.source_type == "text":
            if self.total_tokens is None or self.chunk_tokens is None:
                raise ValueError("Text dataset requires total_tokens and chunk_tokens.")
            if self.chunk_tokens <= 0:
                raise ValueError("chunk_tokens must be > 0.")
            if self.chunk_overlap < 0:
                raise ValueError("chunk_overlap must be >= 0.")
            if self.chunk_overlap >= self.chunk_tokens:
                raise ValueError("chunk_overlap must be < chunk_tokens.")
            if self.vectors_per_chunk <= 0:
                raise ValueError("vectors_per_chunk must be > 0.")
        elif self.source_type == "image":
            if self.item_count is None:
                raise ValueError("Image dataset requires item_count.")
            if self.item_count <= 0 or self.vectors_per_item <= 0:
                raise ValueError("item_count and vectors_per_item must be > 0.")
        elif self.source_type in {"audio", "video"}:
            if self.duration_seconds is None or self.segment_seconds is None:
                raise ValueError(
                    "Audio/video dataset requires duration_seconds and segment_seconds."
                )
            if self.duration_seconds <= 0 or self.segment_seconds <= 0:
                raise ValueError("duration_seconds and segment_seconds must be > 0.")
            if self.segment_overlap_seconds < 0:
                raise ValueError("segment_overlap_seconds must be >= 0.")
            if self.segment_overlap_seconds >= self.segment_seconds:
                raise ValueError("segment_overlap_seconds must be < segment_seconds.")
            if self.vectors_per_segment <= 0:
                raise ValueError("vectors_per_segment must be > 0.")
        elif self.source_type == "tabular":
            if self.row_count is None:
                raise ValueError("Tabular dataset requires row_count.")
            if self.row_count <= 0 or self.vectors_per_row <= 0:
                raise ValueError("row_count and vectors_per_row must be > 0.")
        elif self.source_type == "precomputed_vectors":
            if self.vector_count is None or self.vector_count <= 0:
                raise ValueError("Precomputed vectors dataset requires vector_count > 0.")

        return self


class EmbeddingConfig(BaseModel):
    kind: Literal["dense", "sparse", "hybrid"]
    dimensions: int | None = None
    dtype: Literal["float32", "float16", "bfloat16", "int8", "uint8", "binary"] | None = None
    sparse_non_zero_avg: int | None = None
    sparse_pair_bytes: int = 8

    @model_validator(mode="after")
    def validate_embedding(self) -> "EmbeddingConfig":
        requires_dense = self.kind in {"dense", "hybrid"}
        requires_sparse = self.kind in {"sparse", "hybrid"}

        if requires_dense:
            if self.dimensions is None or self.dimensions <= 0:
                raise ValueError("Dense/hybrid embedding requires dimensions > 0.")
            if self.dtype is None:
                raise ValueError("Dense/hybrid embedding requires dtype.")

        if requires_sparse:
            if self.sparse_non_zero_avg is None or self.sparse_non_zero_avg <= 0:
                raise ValueError("Sparse/hybrid embedding requires sparse_non_zero_avg > 0.")
            if self.sparse_pair_bytes <= 0:
                raise ValueError("sparse_pair_bytes must be > 0.")

        return self


class RecordConfig(BaseModel):
    id_bytes_avg: int = 16
    metadata_bytes_avg: int = 0
    source_text_bytes_avg: int = 0
    provenance_bytes_avg: int = 0

    @model_validator(mode="after")
    def validate_record(self) -> "RecordConfig":
        if self.id_bytes_avg <= 0:
            raise ValueError("id_bytes_avg must be > 0.")
        for field_name in ["metadata_bytes_avg", "source_text_bytes_avg", "provenance_bytes_avg"]:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0.")
        return self


class DatabaseConfig(BaseModel):
    engine: Literal[
        "generic",
        "pgvector",
        "qdrant",
        "milvus",
        "elasticsearch",
        "opensearch",
        "weaviate",
        "pinecone",
    ] = "generic"
    index_type: Literal["none", "flat", "hnsw", "ivf_flat", "ivf_pq", "diskann"] = "hnsw"
    replicas: int = 1
    shards: int = 1

    @model_validator(mode="after")
    def validate_db(self) -> "DatabaseConfig":
        if self.replicas <= 0:
            raise ValueError("replicas must be > 0.")
        if self.shards <= 0:
            raise ValueError("shards must be > 0.")
        return self


class IndexConfig(BaseModel):
    m: int = 16
    hnsw_layer_factor: float = 1.2
    link_bytes: int = 8
    nlist: int | None = None
    pq_code_bytes: int | None = None
    diskann_graph_factor: float = 0.25
    diskann_ram_fraction: float = 0.05
    quantization: Literal["none", "scalar_int8", "int4", "binary", "product", "bbq"] = "none"

    @model_validator(mode="after")
    def validate_index(self) -> "IndexConfig":
        if self.m <= 0:
            raise ValueError("m must be > 0.")
        if self.link_bytes <= 0:
            raise ValueError("link_bytes must be > 0.")
        if self.hnsw_layer_factor < 1.0:
            raise ValueError("hnsw_layer_factor must be >= 1.0.")
        if self.nlist is not None and self.nlist <= 0:
            raise ValueError("nlist must be > 0 when provided.")
        if self.pq_code_bytes is not None and self.pq_code_bytes <= 0:
            raise ValueError("pq_code_bytes must be > 0 when provided.")
        if self.diskann_graph_factor < 0:
            raise ValueError("diskann_graph_factor must be >= 0.")
        if self.diskann_ram_fraction < 0:
            raise ValueError("diskann_ram_fraction must be >= 0.")
        return self


class StorageConfig(BaseModel):
    wal_factor: float = 0.10
    snapshot_factor: float = 0.00
    safety_factor: float = 1.25
    engine_overhead_factor: float | None = None

    store_original_vectors: bool = True
    store_quantized_vectors: bool = False

    payload_on_disk: bool = True
    vectors_on_disk: bool = False
    hnsw_on_disk: bool = False

    store_vectors_in_source: bool = False

    mode: Literal["in_memory", "on_disk"] = "in_memory"
    compression_level: Literal["none", "2x", "4x", "8x", "16x", "32x"] = "none"

    @model_validator(mode="after")
    def validate_storage(self) -> "StorageConfig":
        if self.wal_factor < 0:
            raise ValueError("wal_factor must be >= 0.")
        if self.snapshot_factor < 0:
            raise ValueError("snapshot_factor must be >= 0.")
        if self.safety_factor < 1.0:
            raise ValueError("safety_factor must be >= 1.0.")
        if self.engine_overhead_factor is not None and self.engine_overhead_factor < 0:
            raise ValueError("engine_overhead_factor must be >= 0.")
        return self


class ScenarioConfig(BaseModel):
    name: str = "default"
    dataset: DatasetConfig
    embedding: EmbeddingConfig
    record: RecordConfig = Field(default_factory=RecordConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "ScenarioConfig":
        index_type = self.database.index_type
        embedding = self.embedding

        if index_type in {"ivf_flat", "ivf_pq"}:
            if self.index.nlist is None:
                raise ValueError(f"{index_type} requires index.nlist.")
            if embedding.dimensions is None or embedding.dtype is None:
                raise ValueError(f"{index_type} requires dense embedding dimensions and dtype.")
            if embedding.dtype == "binary":
                raise ValueError(f"{index_type} does not support binary dtype.")

        if index_type == "ivf_pq" and self.index.pq_code_bytes is None:
            raise ValueError("ivf_pq requires index.pq_code_bytes.")

        if self.index.quantization != "none":
            if embedding.kind not in {"dense", "hybrid"}:
                raise ValueError("quantization requires dense-compatible embedding.")
            if embedding.dimensions is None or embedding.dtype is None:
                raise ValueError("quantization requires dimensions and dtype.")
            if self.index.quantization == "product" and self.index.pq_code_bytes is None:
                raise ValueError("product quantization requires index.pq_code_bytes.")

        return self


class MultiScenarioConfig(BaseModel):
    scenarios: list[ScenarioConfig]

    @model_validator(mode="after")
    def validate_scenarios(self) -> "MultiScenarioConfig":
        if not self.scenarios:
            raise ValueError("multi-scenario input requires at least one scenario")
        seen: set[str] = set()
        for scenario in self.scenarios:
            if scenario.name in seen:
                raise ValueError(f"duplicate scenario name: {scenario.name}")
            seen.add(scenario.name)
        return self


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be an object.")
    return data


def load_scenario(path: Path) -> ScenarioConfig:
    return ScenarioConfig.model_validate(load_yaml(path))
