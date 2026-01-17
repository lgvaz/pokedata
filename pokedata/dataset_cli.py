# create a CLI command to build a dataset
# `pokedata dataset build`

from loguru import logger
import typer
from pokedata.dataset_build import build_dataset, delete_dataset

from pokedata.dataset_splits import CertIdSplitter, RatioSplitPolicy

dataset_app = typer.Typer(help="Dataset operations")


@dataset_app.command()
def rebuild(ctx: typer.Context) -> None:
    cli_context = ctx.obj
    dataset_layout = cli_context.dataset_layout

    confirm = typer.confirm(
        f"Are you sure you want to delete the previous dataset at {dataset_layout.canonical}?"
    )
    if not confirm:
        raise typer.Abort()

    logger.info(f"Deleting previous dataset {dataset_layout.canonical}")
    delete_dataset(dataset_layout)

    splits = cli_context.config["datasets"]["splits"]
    logger.info(f"Splitter seed: {splits['seed']}")
    split_policy = RatioSplitPolicy(
        train=splits["train"],
        val=splits["val"],
        test=splits["test"],
    )
    splitter = CertIdSplitter(split_policy, seed=splits["seed"])

    logger.info(f"Building dataset {dataset_layout.canonical}")
    build_dataset(dataset_layout=dataset_layout, splitter=splitter)
