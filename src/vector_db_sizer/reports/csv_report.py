import csv
import io

from vector_db_sizer.models import EstimateResult

HEADER = [
    "scenario_name",
    "engine",
    "index_type",
    "n_records",
    "dimensions",
    "dtype",
    "final_disk_bytes",
    "final_ram_bytes",
    "raw_vectors_bytes",
    "payload_bytes",
    "index_disk_bytes",
    "engine_overhead_bytes",
    "confidence_final_disk",
    "confidence_ram",
]


def to_csv(results: list[EstimateResult]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=HEADER)
    writer.writeheader()
    for r in results:
        writer.writerow(
            {
                "scenario_name": r.scenario.name,
                "engine": r.scenario.engine,
                "index_type": r.scenario.index_type,
                "n_records": r.scenario.n_records,
                "dimensions": r.scenario.dimensions,
                "dtype": r.scenario.dtype,
                "final_disk_bytes": r.storage_bytes.final_disk,
                "final_ram_bytes": r.ram_bytes.final_ram,
                "raw_vectors_bytes": r.storage_bytes.raw_vectors,
                "payload_bytes": r.storage_bytes.record_payload,
                "index_disk_bytes": r.storage_bytes.index_disk,
                "engine_overhead_bytes": r.storage_bytes.engine_overhead,
                "confidence_final_disk": r.confidence.final_disk,
                "confidence_ram": r.confidence.ram,
            }
        )
    return output.getvalue()
