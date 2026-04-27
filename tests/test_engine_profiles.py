import math

from vector_db_sizer.config import ScenarioConfig
from vector_db_sizer.estimators.scenario import estimate_scenario


def _base() -> dict:
    return {
        "name": "x",
        "dataset": {"source_type": "precomputed_vectors", "vector_count": 10},
        "embedding": {"kind": "dense", "dimensions": 768, "dtype": "float16"},
        "record": {
            "id_bytes_avg": 16,
            "metadata_bytes_avg": 10,
            "source_text_bytes_avg": 0,
            "provenance_bytes_avg": 0,
        },
        "database": {
            "engine": "generic",
            "index_type": "hnsw",
            "replicas": 1,
            "shards": 1,
        },
        "index": {"m": 16, "hnsw_layer_factor": 1.2, "link_bytes": 8},
        "storage": {"wal_factor": 0.1, "snapshot_factor": 0.0, "safety_factor": 1.25},
    }


def test_pgvector_raw_formula() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "pgvector"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.storage_bytes.raw_vectors == 10 * (2 * 768 + 8)


def test_qdrant_disk_flags() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "qdrant"
    cfg["storage"].update({"payload_on_disk": True, "vectors_on_disk": True, "hnsw_on_disk": True})
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.ram_bytes.payload == math.ceil(res.storage_bytes.record_payload * 0.10)
    assert res.ram_bytes.vectors == math.ceil(res.storage_bytes.raw_vectors * 0.10)
    base_index = math.ceil(10 * 16 * 8 * 1.2)
    assert res.ram_bytes.index == math.ceil(base_index * 0.15)


def test_milvus_segment_overhead() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "milvus"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    raw = res.storage_bytes.raw_vectors
    payload = res.storage_bytes.record_payload
    index = res.storage_bytes.index_disk
    expected_extra = math.ceil((raw + payload + index) * 0.10)
    expected_overhead = math.ceil((raw + payload + index + expected_extra) * 0.20)
    assert res.storage_bytes.engine_overhead == expected_overhead


def test_elasticsearch_quantized_source_dup() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "elasticsearch"
    cfg["storage"]["store_vectors_in_source"] = True
    cfg["index"]["quantization"] = "scalar_int8"
    cfg["embedding"]["dtype"] = "float32"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.storage_bytes.source_vector_duplicates == res.storage_bytes.raw_vectors
    assert res.storage_bytes.quantized_vectors == math.ceil(res.storage_bytes.raw_vectors * 0.25)


def test_elasticsearch_quantization_warning_is_conditional() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "elasticsearch"
    cfg["index"]["quantization"] = "none"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert not any("quantization keeps raw float vectors" in w for w in res.warnings)

    cfg["index"]["quantization"] = "scalar_int8"
    res_quantized = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert any("quantization keeps raw float vectors" in w for w in res_quantized.warnings)


def test_qdrant_store_original_vectors_false_is_advisory() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "qdrant"
    cfg["index"]["quantization"] = "scalar_int8"
    cfg["storage"]["store_original_vectors"] = False
    cfg["storage"]["store_quantized_vectors"] = True
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.storage_bytes.raw_vectors > 0
    assert res.storage_bytes.quantized_vectors > 0
    assert any("conservatively keeps raw vectors" in w for w in res.warnings)


def test_pgvector_int8_warning() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "pgvector"
    cfg["embedding"]["dtype"] = "int8"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert any("no type-specific int8/uint8" in w for w in res.warnings)


def test_opensearch_hnsw_formula() -> None:
    cfg = _base()
    cfg["dataset"]["vector_count"] = 1000
    cfg["embedding"] = {"kind": "dense", "dimensions": 768, "dtype": "float32"}
    cfg["database"]["engine"] = "opensearch"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    expected = math.ceil(1000 * 1.1 * (4 * 768 + 8 * 16))
    assert res.storage_bytes.index_disk == expected


def test_weaviate_hnsw_ram_and_overhead() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "weaviate"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.ram_bytes.index >= res.storage_bytes.raw_vectors


def test_pinecone_index_disk_and_warning() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "pinecone"
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.storage_bytes.index_disk == 0
    assert any("pricing" in w.lower() for w in res.warnings)


def test_pinecone_namespace_size_uses_logical_storage() -> None:
    cfg = _base()
    cfg["database"]["engine"] = "pinecone"
    cfg["storage"]["wal_factor"] = 0.5
    cfg["storage"]["safety_factor"] = 2.0
    res = estimate_scenario(ScenarioConfig.model_validate(cfg))
    assert res.engine_metrics is not None
    assert res.engine_metrics["namespace_size_gb"] < (res.storage_bytes.final_disk / 1_000_000_000)
