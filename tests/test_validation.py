import pytest
from pydantic import ValidationError

from vector_db_sizer.config import ScenarioConfig


def _base() -> dict:
    return {
        "name": "x",
        "dataset": {
            "source_type": "text",
            "total_tokens": 1000,
            "chunk_tokens": 100,
            "chunk_overlap": 10,
            "vectors_per_chunk": 1,
        },
        "embedding": {"kind": "dense", "dimensions": 128, "dtype": "float32"},
        "database": {"engine": "generic", "index_type": "hnsw", "replicas": 1, "shards": 1},
        "index": {"m": 16, "hnsw_layer_factor": 1.2, "link_bytes": 8},
        "storage": {"wal_factor": 0.1, "snapshot_factor": 0.0, "safety_factor": 1.25},
    }


def test_chunk_overlap_invalid() -> None:
    data = _base()
    data["dataset"]["chunk_overlap"] = 100
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_segment_overlap_invalid() -> None:
    data = _base()
    data["dataset"] = {
        "source_type": "audio",
        "duration_seconds": 60,
        "segment_seconds": 5,
        "segment_overlap_seconds": 5,
        "vectors_per_segment": 1,
    }
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_dimensions_invalid() -> None:
    data = _base()
    data["embedding"]["dimensions"] = 0
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_sparse_missing_nnz() -> None:
    data = _base()
    data["embedding"] = {"kind": "sparse", "sparse_pair_bytes": 8}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_hybrid_missing_dense() -> None:
    data = _base()
    data["embedding"] = {"kind": "hybrid", "sparse_non_zero_avg": 10, "sparse_pair_bytes": 8}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_hybrid_missing_sparse() -> None:
    data = _base()
    data["embedding"] = {"kind": "hybrid", "dimensions": 128, "dtype": "float32"}
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_known_engine_qdrant_valid() -> None:
    data = _base()
    data["database"]["engine"] = "qdrant"
    ScenarioConfig.model_validate(data)


def test_unknown_engine_fails() -> None:
    data = _base()
    data["database"]["engine"] = "unknown_engine"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_unknown_dtype_fails() -> None:
    data = _base()
    data["embedding"]["dtype"] = "unknown_dtype"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_unknown_index_type_fails() -> None:
    data = _base()
    data["database"]["index_type"] = "unknown"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_ivf_flat_without_nlist_fails() -> None:
    data = _base()
    data["database"]["index_type"] = "ivf_flat"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_ivf_pq_without_pq_code_bytes_fails() -> None:
    data = _base()
    data["database"]["index_type"] = "ivf_pq"
    data["index"]["nlist"] = 10
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_product_quantization_without_pq_code_bytes_fails() -> None:
    data = _base()
    data["index"]["quantization"] = "product"
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_replicas_invalid() -> None:
    data = _base()
    data["database"]["replicas"] = 0
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_safety_factor_invalid() -> None:
    data = _base()
    data["storage"]["safety_factor"] = 0.9
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_wal_factor_invalid() -> None:
    data = _base()
    data["storage"]["wal_factor"] = -0.1
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)


def test_snapshot_factor_invalid() -> None:
    data = _base()
    data["storage"]["snapshot_factor"] = -0.1
    with pytest.raises(ValidationError):
        ScenarioConfig.model_validate(data)
