"""CLI module for pokedata."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from loguru import logger
import typer

from pokedata.config import ConfigDict, ConfigError, load_config
from pokedata.cvat import CVATClient, CVATError
from pokedata.dataset_cli import dataset_app
from pokedata.dataset_layout import DatasetLayout

app = typer.Typer()
app.add_typer(dataset_app, name="dataset")


@dataclass
class CLIContext:
    config: ConfigDict
    dataset_layout: DatasetLayout


@app.callback()
def main(
    ctx: typer.Context,
    config_path: Path = typer.Option("config.yaml", help="Path to config file"),
    secrets_path: Path = typer.Option("secrets.yaml", help="Path to secrets file"),
    dataset_repo: Optional[Path] = typer.Option(
        None, help="Dataset repository directory (overrides config)"
    ),
) -> None:
    """Main entry point for the CLI."""
    config = load_config(config_path, secrets_path)
    if dataset_repo:
        config["datasets"]["dataset_repo"] = dataset_repo

    dataset_layout = DatasetLayout(Path(config["datasets"]["dataset_repo"]))

    ctx.obj = CLIContext(config=config, dataset_layout=dataset_layout)


@app.command()
def download_task(
    ctx: typer.Context,
    task_id: int,
    format: str = typer.Option("LabelMe 3.0", help="Annotation format"),
) -> None:
    """Download a task's dataset from CVAT."""
    try:
        cli_context = ctx.obj
        api_url = cli_context.config["cvat"]["url"]
        auth = cli_context.config["cvat"]["auth"]

        # Create CVAT client and download task
        cli_context.dataset_layout.cvat_raw.mkdir(parents=True, exist_ok=True)
        client = CVATClient(api_url=api_url, auth=auth)
        result_path = client.download_task(
            task_id=task_id,
            output_dir=cli_context.dataset_layout.cvat_raw,
            format=format,
        )

        typer.echo(f"✓ Task {task_id} downloaded successfully to: {result_path}")

    except ConfigError as e:
        typer.echo(f"✗ Configuration error: {e}", err=True)
        raise typer.Exit(1)
    except CVATError as e:
        typer.echo(f"✗ CVAT error: {e}", err=True)
        raise typer.Exit(1)
    except KeyError as e:
        typer.echo(f"✗ Missing configuration key: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"✗ Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
