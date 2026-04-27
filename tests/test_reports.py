import json

from typer.testing import CliRunner

from vector_db_sizer.cli import app


def test_markdown_contains_required_sections() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["estimate", "examples/qdrant_text_hnsw.yaml", "--format", "markdown"],
    )
    assert result.exit_code == 0
    for token in [
        "# Vector Storage Sizing Report",
        "## Scenario Summary",
        "## Storage Estimate",
        "## RAM Estimate",
        "## Warnings",
        "## Notes",
        "## Confidence",
        "| Field | Value |",
        "| Component | Bytes |",
        "| Component | Confidence |",
        "qdrant_text_hnsw",
        "qdrant",
        "hnsw",
    ]:
        assert token in result.stdout
    for row_name in [
        "Raw vectors",
        "Quantized vectors",
        "Record payload",
        "Index disk",
        "Source vector duplicates",
        "Engine overhead",
        "Base storage",
        "Replicated storage",
        "WAL estimate",
        "Snapshot estimate",
        "Final disk estimate",
    ]:
        assert row_name in result.stdout


def test_csv_multi_header_and_rows() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["estimate", "examples/multi_scenario.yaml", "--format", "csv"])
    assert result.exit_code == 0
    lines = [line for line in result.stdout.strip().splitlines() if line]
    assert lines[0] == (
        "scenario_name,engine,index_type,n_records,dimensions,dtype,"
        "final_disk_bytes,final_ram_bytes,raw_vectors_bytes,payload_bytes,"
        "index_disk_bytes,engine_overhead_bytes,confidence_final_disk,confidence_ram"
    )
    assert len(lines) == 5
    scenario_names = {line.split(",")[0] for line in lines[1:]}
    assert scenario_names == {
        "qdrant_text_hnsw",
        "pgvector_halfvec",
        "elasticsearch_quantized",
        "opensearch_hnsw",
    }


def test_json_snapshot_shapes() -> None:
    runner = CliRunner()
    single = runner.invoke(app, ["estimate", "examples/qdrant_text_hnsw.yaml", "--format", "json"])
    assert single.exit_code == 0
    single_payload = json.loads(single.stdout)
    assert set(single_payload.keys()) == {
        "scenario",
        "storage_bytes",
        "ram_bytes",
        "warnings",
        "notes",
        "confidence",
        "engine_metrics",
    }

    estimated = runner.invoke(app, ["estimate", "examples/multi_scenario.yaml", "--format", "json"])
    assert estimated.exit_code == 0
    payload = json.loads(estimated.stdout)
    assert set(payload.keys()) == {"results"}
    assert len(payload["results"]) == 4
