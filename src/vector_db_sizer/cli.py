from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from yaml import YAMLError

from vector_db_sizer.engines import explain_engine as explain_engine_text
from vector_db_sizer.engines import list_engine_names
from vector_db_sizer.estimators.scenario import estimate_scenario
from vector_db_sizer.reports import to_csv, to_json, to_markdown
from vector_db_sizer.scenarios import load_scenarios

app = typer.Typer(help="Estimate vector database storage and memory usage.")
SUPPORTED_INDEXES = ["none", "flat", "hnsw", "ivf_flat", "ivf_pq", "diskann"]


@app.command()
def estimate(
    scenario_path: Path,
    format: Annotated[str, typer.Option("--format")] = "json",
    out: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Estimate storage and RAM from a scenario YAML file."""
    try:
        loaded = load_scenarios(scenario_path)
        results = [estimate_scenario(config) for config in loaded.scenarios]

        if format == "json":
            rendered = to_json(results if loaded.is_multi_scenario else results[0])
        elif format == "markdown":
            rendered = to_markdown(results)
        elif format == "csv":
            rendered = to_csv(results)
        else:
            raise typer.BadParameter("Unknown format. Use json, markdown, or csv.")

        if out is None:
            typer.echo(rendered)
        else:
            out.write_text(rendered + ("" if rendered.endswith("\n") else "\n"), encoding="utf-8")
            typer.echo(f"Wrote report to {out}")
    except FileNotFoundError:
        typer.echo(f"Error: scenario file not found: {scenario_path}", err=True)
        raise typer.Exit(code=1) from None
    except YAMLError as exc:
        typer.echo(f"Error: invalid YAML: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except ValidationError as exc:
        typer.echo(f"Error: validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1) from None
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except typer.BadParameter as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None


@app.command()
def validate(scenario_path: Path) -> None:
    """Validate a scenario YAML file."""
    try:
        _ = load_scenarios(scenario_path)
        typer.echo("OK")
    except FileNotFoundError:
        typer.echo(f"Error: scenario file not found: {scenario_path}", err=True)
        raise typer.Exit(code=1) from None
    except (YAMLError, ValidationError, ValueError) as exc:
        typer.echo(f"Error: validation failed:\n{exc}", err=True)
        raise typer.Exit(code=1) from None


@app.command("list-engines")
def list_engines() -> None:
    """List supported engines."""
    for engine in list_engine_names():
        typer.echo(engine)


@app.command("list-indexes")
def list_indexes() -> None:
    """List supported index types."""
    for index_name in SUPPORTED_INDEXES:
        typer.echo(index_name)


@app.command("explain-engine")
def explain_engine(engine_name: str) -> None:
    """Explain an engine profile."""
    try:
        typer.echo(explain_engine_text(engine_name))
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
