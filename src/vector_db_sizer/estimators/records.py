from vector_db_sizer.config import RecordConfig


def estimate_record_payload_bytes(n_records: int, record: RecordConfig) -> int:
    per_record = (
        record.id_bytes_avg
        + record.metadata_bytes_avg
        + record.source_text_bytes_avg
        + record.provenance_bytes_avg
    )
    return n_records * per_record
