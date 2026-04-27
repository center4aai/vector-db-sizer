import math

from vector_db_sizer.config import EmbeddingConfig
from vector_db_sizer.estimators.vectors import (
    dense_vector_bytes,
    estimate_vector_bytes,
    sparse_vector_bytes,
)


def test_dense_float32() -> None:
    assert dense_vector_bytes(10, 1536, "float32") == 10 * 1536 * 4


def test_dense_float16() -> None:
    assert dense_vector_bytes(10, 1536, "float16") == 10 * 1536 * 2


def test_dense_bfloat16() -> None:
    assert dense_vector_bytes(10, 1536, "bfloat16") == 10 * 1536 * 2


def test_dense_int8() -> None:
    assert dense_vector_bytes(10, 1536, "int8") == 10 * 1536


def test_dense_binary() -> None:
    assert dense_vector_bytes(10, 1537, "binary") == 10 * math.ceil(1537 / 8)


def test_sparse_bytes() -> None:
    assert sparse_vector_bytes(10, 96, 8) == 10 * 96 * 8


def test_hybrid_bytes() -> None:
    cfg = EmbeddingConfig(
        kind="hybrid",
        dimensions=1536,
        dtype="float32",
        sparse_non_zero_avg=96,
        sparse_pair_bytes=8,
    )
    expected = (10 * 1536 * 4) + (10 * 96 * 8)
    assert estimate_vector_bytes(10, cfg) == expected
