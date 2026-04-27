import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from vector_db_sizer.cli import app
from vector_db_sizer.scenarios import load_scenarios


def test_load_single_scenario_sets_single_flag() -> None:
    loaded = load_scenarios(Path("examples/generic_text_hnsw.yaml"))
    assert loaded.is_multi_scenario is False
    assert len(loaded.scenarios) == 1


def test_load_multi_scenario_sets_multi_flag() -> None:
    loaded = load_scenarios(Path("examples/multi_scenario.yaml"))
    assert loaded.is_multi_scenario is True
    assert len(loaded.scenarios) == 4


def test_multi_scenario_with_single_item_renders_results_wrapper(tmp_path) -> None:
    scenario_file = tmp_path / "one_in_multi.yaml"
    scenario_file.write_text(
        "\n".join(
            [
                "scenarios:",
                "  - name: single_inside_multi",
                "    dataset:",
                "      source_type: precomputed_vectors",
                "      vector_count: 10",
                "    embedding:",
                "      kind: dense",
                "      dimensions: 8",
                "      dtype: float32",
            ]
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["estimate", str(scenario_file), "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "results" in payload
    assert len(payload["results"]) == 1


def test_empty_scenarios_fails() -> None:
    with pytest.raises(
        ValidationError, match="multi-scenario input requires at least one scenario"
    ):
        load_scenarios(Path("tests/fixtures/empty_scenarios.yaml"))


def test_duplicate_scenario_names_fail() -> None:
    with pytest.raises(ValidationError, match="duplicate scenario name: same"):
        load_scenarios(Path("tests/fixtures/duplicate_scenarios.yaml"))
