from vector_db_sizer.config import RecordConfig
from vector_db_sizer.estimators.records import estimate_record_payload_bytes


def test_record_payload_bytes() -> None:
    cfg = RecordConfig(
        id_bytes_avg=16,
        metadata_bytes_avg=512,
        source_text_bytes_avg=100,
        provenance_bytes_avg=64,
    )
    assert estimate_record_payload_bytes(10, cfg) == 10 * (16 + 512 + 100 + 64)
