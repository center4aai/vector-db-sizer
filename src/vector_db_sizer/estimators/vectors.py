import math

from vector_db_sizer.config import EmbeddingConfig

DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "uint8": 1,
}


def dense_vector_bytes(n_records: int, dimensions: int, dtype: str) -> int:
    if dtype == "binary":
        return n_records * math.ceil(dimensions / 8)
    return n_records * dimensions * DTYPE_BYTES[dtype]


def sparse_vector_bytes(n_records: int, sparse_non_zero_avg: int, sparse_pair_bytes: int) -> int:
    return n_records * sparse_non_zero_avg * sparse_pair_bytes


def estimate_vector_bytes(n_records: int, embedding: EmbeddingConfig) -> int:
    total = 0

    if embedding.kind in {"dense", "hybrid"}:
        total += dense_vector_bytes(n_records, embedding.dimensions, embedding.dtype)

    if embedding.kind in {"sparse", "hybrid"}:
        total += sparse_vector_bytes(
            n_records,
            embedding.sparse_non_zero_avg,
            embedding.sparse_pair_bytes,
        )

    return total
