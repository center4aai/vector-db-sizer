from pathlib import Path

from typer.testing import CliRunner

from vector_db_sizer.cli import app


def test_all_examples_validate() -> None:
    runner = CliRunner()
    for path in sorted(Path("examples").glob("*.yaml")):
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0, f"{path}: {result.stdout}"
        assert result.stdout.strip() == "OK"


def test_examples_estimate_representative_formats() -> None:
    runner = CliRunner()
    checks = [
        ("examples/generic_text_hnsw.yaml", "json"),
        ("examples/qdrant_text_hnsw.yaml", "markdown"),
        ("examples/multi_scenario.yaml", "csv"),
    ]
    for path, fmt in checks:
        result = runner.invoke(app, ["estimate", path, "--format", fmt])
        assert result.exit_code == 0, f"{path} ({fmt}): {result.stdout}"
