import math

from vector_db_sizer.config import EmbeddingConfig, IndexConfig
from vector_db_sizer.estimators.vectors import DTYPE_BYTES


def estimate_index_bytes(
    index_type: str,
    n_records: int,
    index: IndexConfig,
    embedding: EmbeddingConfig,
    raw_vector_bytes: int,
) -> tuple[int, int, str, list[str]]:
    warnings: list[str] = []

    if index_type == "none":
        return 0, 0, "high", warnings

    if index_type == "flat":
        warnings.append(
            "Flat index has minimal index overhead but may require scanning raw vectors."
        )
        return 0, 0, "high", warnings

    if index_type == "hnsw":
        disk = math.ceil(n_records * index.m * index.link_bytes * index.hnsw_layer_factor)
        return disk, disk, "medium", warnings

    if index_type == "ivf_flat":
        dtype_bytes = DTYPE_BYTES[embedding.dtype]
        centroids = index.nlist * embedding.dimensions * dtype_bytes
        assignments = n_records * 4
        total = centroids + assignments
        return total, total, "medium", warnings

    if index_type == "ivf_pq":
        dtype_bytes = DTYPE_BYTES[embedding.dtype]
        centroids = index.nlist * embedding.dimensions * dtype_bytes
        assignments = n_records * 4
        pq_codes = n_records * index.pq_code_bytes
        total = centroids + assignments + pq_codes
        return total, total, "medium", warnings

    if index_type == "diskann":
        disk = math.ceil(raw_vector_bytes * index.diskann_graph_factor)
        ram = math.ceil(raw_vector_bytes * index.diskann_ram_fraction)
        return disk, ram, "low", warnings

    raise ValueError(f"Unsupported index_type: {index_type}")
