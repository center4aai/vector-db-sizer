import json

from typer.testing import CliRunner

from vector_db_sizer.cli import app


def test_validate_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validate", "examples/generic_text_hnsw.yaml"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "OK"


def test_estimate_json_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["estimate", "examples/generic_text_hnsw.yaml", "--format", "json"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert "n_records" in payload["scenario"]
    assert "raw_vectors" in payload["storage_bytes"]
    assert "quantized_vectors" in payload["storage_bytes"]
    assert "final_disk" in payload["storage_bytes"]
    assert "final_ram" in payload["ram_bytes"]
    assert "warnings" in payload
    assert "confidence" in payload


def test_list_engines() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list-engines"])
    assert result.exit_code == 0
    assert result.stdout.strip().splitlines() == [
        "generic",
        "pgvector",
        "qdrant",
        "milvus",
        "elasticsearch",
        "opensearch",
        "weaviate",
        "pinecone",
    ]


def test_list_indexes() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["list-indexes"])
    assert result.exit_code == 0
    assert result.stdout.strip().splitlines() == [
        "none",
        "flat",
        "hnsw",
        "ivf_flat",
        "ivf_pq",
        "diskann",
    ]


def test_markdown_csv_and_explain_commands() -> None:
    runner = CliRunner()
    md = runner.invoke(app, ["estimate", "examples/qdrant_text_hnsw.yaml", "--format", "markdown"])
    assert md.exit_code == 0
    assert "# Vector Storage Sizing Report" in md.stdout

    csv = runner.invoke(app, ["estimate", "examples/multi_scenario.yaml", "--format", "csv"])
    assert csv.exit_code == 0
    assert csv.stdout.splitlines()[0].startswith("scenario_name,engine,index_type")

    explain = runner.invoke(app, ["explain-engine", "elasticsearch"])
    assert explain.exit_code == 0
    assert "Elasticsearch profile:" in explain.stdout


def test_validate_missing_file_has_clean_error() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["validate", "does-not-exist.yaml"])
    assert result.exit_code == 1
    assert "scenario file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_validate_invalid_yaml_has_clean_error(tmp_path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("name: [", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(invalid)])
    assert result.exit_code == 1
    assert "validation failed" in result.stderr
    assert "Traceback" not in result.stderr


def test_validate_root_must_be_object(tmp_path) -> None:
    invalid = tmp_path / "list_root.yaml"
    invalid.write_text("- one\n- two\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(app, ["validate", str(invalid)])
    assert result.exit_code == 1
    assert "YAML root must be an object" in result.stderr
    assert "Traceback" not in result.stderr

def test_invalid_engine_has_clean_validation_error(tmp_path) -> None:
    scenario_file = tmp_path / "invalid_engine.yaml"
    scenario_file.write_text(
        """name: invalid_engine_case


dataset:
  source_type: precomputed_vectors
  vector_count: 10

embedding:
  kind: dense
  dimensions: 8
  dtype: float32

database:
  engine: unknown_engine
  index_type: hnsw
""",
        encoding="utf-8",
    )

    runner = CliRunner()

    validate_result = runner.invoke(app, ["validate", str(scenario_file)])
    assert validate_result.exit_code != 0
    validate_output = validate_result.output
    assert "validation failed" in validate_output.lower()
    assert "traceback" not in validate_output.lower()

    estimate_result = runner.invoke(app, ["estimate", str(scenario_file), "--format", "json"])
    assert estimate_result.exit_code != 0
    estimate_output = estimate_result.output
    assert "validation failed" in estimate_output.lower()
    assert "traceback" not in estimate_output.lower()


def test_invalid_dtype_has_clean_validation_error(tmp_path) -> None:
    scenario_file = tmp_path / "invalid_dtype.yaml"
    scenario_file.write_text(
        """name: invalid_dtype_case


dataset:
  source_type: precomputed_vectors
  vector_count: 10

embedding:
  kind: dense
  dimensions: 8
  dtype: unknown_dtype

database:
  engine: generic
  index_type: hnsw
""",
        encoding="utf-8",
    )

    runner = CliRunner()

    validate_result = runner.invoke(app, ["validate", str(scenario_file)])
    assert validate_result.exit_code != 0
    validate_output = validate_result.output
    assert "validation failed" in validate_output.lower()
    assert "traceback" not in validate_output.lower()

    estimate_result = runner.invoke(app, ["estimate", str(scenario_file), "--format", "json"])
    assert estimate_result.exit_code != 0
    estimate_output = estimate_result.output
    assert "validation failed" in estimate_output.lower()
    assert "traceback" not in estimate_output.lower()
