from vector_db_sizer.config import DatasetConfig
from vector_db_sizer.estimators.datasets import estimate_record_count


def test_text_dataset_count() -> None:
    cfg = DatasetConfig(
        source_type="text",
        total_tokens=50_000_000,
        chunk_tokens=512,
        chunk_overlap=64,
        vectors_per_chunk=1,
    )
    assert estimate_record_count(cfg) == 111_608


def test_image_dataset_count() -> None:
    cfg = DatasetConfig(source_type="image", item_count=1000, vectors_per_item=2)
    assert estimate_record_count(cfg) == 2000


def test_audio_dataset_count() -> None:
    cfg = DatasetConfig(
        source_type="audio",
        duration_seconds=100,
        segment_seconds=5,
        segment_overlap_seconds=0,
        vectors_per_segment=1,
    )
    assert estimate_record_count(cfg) == 20


def test_video_dataset_count() -> None:
    cfg = DatasetConfig(
        source_type="video",
        duration_seconds=100,
        segment_seconds=5,
        segment_overlap_seconds=0,
        vectors_per_segment=1,
    )
    assert estimate_record_count(cfg) == 20


def test_tabular_dataset_count() -> None:
    cfg = DatasetConfig(source_type="tabular", row_count=1000, vectors_per_row=3)
    assert estimate_record_count(cfg) == 3000


def test_precomputed_dataset_count() -> None:
    cfg = DatasetConfig(source_type="precomputed_vectors", vector_count=12345)
    assert estimate_record_count(cfg) == 12_345
