from vector_db_sizer.models import EstimateResult


def _one(res: EstimateResult, heading: str = "##") -> str:
    s = res.scenario
    st = res.storage_bytes
    ram = res.ram_bytes
    lines = [
        f"{heading} Scenario Summary",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Name | {s.name} |",
        f"| Engine | {s.engine} |",
        f"| Source type | {s.source_type} |",
        f"| Estimated records | {s.n_records:,} |",
        f"| Embedding kind | {s.embedding_kind} |",
        f"| Dimensions | {s.dimensions} |",
        f"| Vector dtype | {s.dtype} |",
        f"| Index type | {s.index_type} |",
        "",
        f"{heading} Storage Estimate",
        "",
        "| Component | Bytes |",
        "|---|---:|",
        f"| Raw vectors | {st.raw_vectors} |",
        f"| Quantized vectors | {st.quantized_vectors} |",
        f"| Record payload | {st.record_payload} |",
        f"| Index disk | {st.index_disk} |",
        f"| Source vector duplicates | {st.source_vector_duplicates} |",
        f"| Engine overhead | {st.engine_overhead} |",
        f"| Base storage | {st.base_storage} |",
        f"| Replicated storage | {st.replicated_storage} |",
        f"| WAL estimate | {st.wal} |",
        f"| Snapshot estimate | {st.snapshot} |",
        f"| Final disk estimate | {st.final_disk} |",
        "",
        f"{heading} RAM Estimate",
        "",
        "| Component | Bytes |",
        "|---|---:|",
        f"| Vector RAM | {ram.vectors} |",
        f"| Quantized vector RAM | {ram.quantized_vectors} |",
        f"| Index RAM | {ram.index} |",
        f"| Payload RAM | {ram.payload} |",
        f"| Engine RAM overhead | {ram.engine_overhead} |",
        f"| Final RAM estimate | {ram.final_ram} |",
        "",
        f"{heading} Warnings",
        "",
    ]
    lines.extend([f"- {w}" for w in res.warnings] or ["- None"])
    lines.extend(["", f"{heading} Notes", ""])
    lines.extend([f"- {n}" for n in res.notes] or ["- None"])
    c = res.confidence
    lines.extend(
        [
            "",
            f"{heading} Confidence",
            "",
            "| Component | Confidence |",
            "|---|---|",
            f"| Raw vectors | {c.raw_vectors} |",
            f"| Record payload | {c.record_payload} |",
            f"| Index | {c.index} |",
            f"| Engine overhead | {c.engine_overhead} |",
            f"| Final disk | {c.final_disk} |",
            f"| RAM | {c.ram} |",
        ]
    )
    return "\n".join(lines)


def to_markdown(results: list[EstimateResult]) -> str:
    if len(results) == 1:
        return "# Vector Storage Sizing Report\n\n" + _one(results[0])
    parts = ["# Vector Storage Sizing Report", ""]
    for res in results:
        parts.extend([f"## Scenario: {res.scenario.name}", "", _one(res, heading="###"), ""])
    return "\n".join(parts)
