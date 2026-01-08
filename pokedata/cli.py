"""CLI module for pokedata."""
import typer

app = typer.Typer()


@app.callback(invoke_without_command=True)
def main() -> None:
    """Main entry point for the CLI."""
    pass


if __name__ == "__main__":
    app()
