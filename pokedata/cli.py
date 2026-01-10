"""CLI module for pokedata."""

from pathlib import Path
from typing import Optional

import typer

from pokedata.config import ConfigError, load_config
from pokedata.cvat import CVATClient, CVATError

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main() -> None:
    """Main entry point for the CLI."""
    pass


@app.command()
def download_task(
    task_id: int,
    config_path: Path = typer.Option("config.yaml", help="Path to config file"),
    secrets_path: Path = typer.Option("secrets.yaml", help="Path to secrets file"),
    output_dir: Optional[Path] = typer.Option(
        None, help="Output directory (overrides config)"
    ),
    format: str = typer.Option("COCO 1.0", help="Annotation format"),
) -> None:
    """Download a task's dataset from CVAT."""
    try:
        config = load_config(config_path, secrets_path)

        api_url = config["cvat"]["url"]
        auth = config["cvat"]["auth"]

        # Create CVAT client and download task
        output_dir = Path(output_dir or config["datasets"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        client = CVATClient(api_url=api_url, auth=auth)
        result_path = client.download_task(
            task_id=task_id, output_dir=output_dir, format=format
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
