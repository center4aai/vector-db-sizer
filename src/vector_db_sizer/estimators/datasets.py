import math

from vector_db_sizer.config import DatasetConfig


def estimate_record_count(dataset: DatasetConfig) -> int:
    if dataset.source_type == "text":
        effective_step = dataset.chunk_tokens - dataset.chunk_overlap
        return math.ceil(dataset.total_tokens / effective_step) * dataset.vectors_per_chunk

    if dataset.source_type == "image":
        return dataset.item_count * dataset.vectors_per_item

    if dataset.source_type in {"audio", "video"}:
        effective_step = dataset.segment_seconds - dataset.segment_overlap_seconds
        return math.ceil(dataset.duration_seconds / effective_step) * dataset.vectors_per_segment

    if dataset.source_type == "tabular":
        return dataset.row_count * dataset.vectors_per_row

    if dataset.source_type == "precomputed_vectors":
        return dataset.vector_count

    raise ValueError(f"Unsupported source_type: {dataset.source_type}")
