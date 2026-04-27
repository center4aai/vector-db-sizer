import math

from vector_db_sizer.config import EmbeddingConfig, IndexConfig
from vector_db_sizer.estimators.indexes import estimate_index_bytes


def _dense_embedding() -> EmbeddingConfig:
    return EmbeddingConfig(kind="dense", dimensions=128, dtype="float32")


def test_none_index() -> None:
    disk, ram, confidence, warnings = estimate_index_bytes(
        "none", 100, IndexConfig(), _dense_embedding(), 1000
    )
    assert (disk, ram, confidence, warnings) == (0, 0, "high", [])


def test_flat_index() -> None:
    disk, ram, confidence, warnings = estimate_index_bytes(
        "flat", 100, IndexConfig(), _dense_embedding(), 1000
    )
    assert (disk, ram, confidence) == (0, 0, "high")
    assert warnings == [
        "Flat index has minimal index overhead but may require scanning raw vectors."
    ]


def test_hnsw_index() -> None:
    disk, ram, confidence, _ = estimate_index_bytes(
        "hnsw",
        100,
        IndexConfig(m=16, link_bytes=8, hnsw_layer_factor=1.2),
        _dense_embedding(),
        1000,
    )
    expected = math.ceil(100 * 16 * 8 * 1.2)
    assert disk == expected
    assert ram == expected
    assert confidence == "medium"


def test_ivf_flat_index() -> None:
    disk, ram, confidence, _ = estimate_index_bytes(
        "ivf_flat",
        100,
        IndexConfig(nlist=10),
        _dense_embedding(),
        1000,
    )
    expected = (10 * 128 * 4) + (100 * 4)
    assert disk == expected
    assert ram == expected
    assert confidence == "medium"


def test_ivf_pq_index() -> None:
    disk, ram, confidence, _ = estimate_index_bytes(
        "ivf_pq",
        100,
        IndexConfig(nlist=10, pq_code_bytes=16),
        _dense_embedding(),
        1000,
    )
    expected = (10 * 128 * 4) + (100 * 4) + (100 * 16)
    assert disk == expected
    assert ram == expected
    assert confidence == "medium"


def test_diskann_index() -> None:
    disk, ram, confidence, _ = estimate_index_bytes(
        "diskann",
        100,
        IndexConfig(diskann_graph_factor=0.25, diskann_ram_fraction=0.05),
        _dense_embedding(),
        1_000_000,
    )
    assert disk == 250_000
    assert ram == 50_000
    assert confidence == "low"
