from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Counter, List, Set, Tuple, TypeAlias
from loguru import logger
import polvo as pv

from pokedata.dataset_layout import DatasetLayout
from pokedata.dataset_splits import DatasetSplit, SplitMap, Splitter
from pokedata.record import Record

__all__ = ["DatasetBuildError", "build_dataset", "plan_dataset", "execute_dataset_plan"]


class DatasetBuildError(Exception):
    """Raised when dataset build fails."""

    pass


CvatTask: TypeAlias = str


def find_duplicate_filenames(paths: List[Path]) -> Set[Path]:
    by_name: dict[str, list[Path]] = defaultdict(list)

    for path in paths:
        by_name[path.name].append(path)

    return [paths for paths in by_name.values() if len(paths) > 1]


def records_from_cvat_raw(dataset_path: Path) -> Tuple[List[Record], Set[CvatTask]]:
    image_paths = pv.get_files(dataset_path, extensions=[".png"])
    annotation_paths = pv.get_files(dataset_path, extensions=[".xml"])

    if duplicate_filenames := find_duplicate_filenames(image_paths):
        raise DatasetBuildError(f"Duplicate images found: {duplicate_filenames}")
    if duplicate_filenames := find_duplicate_filenames(annotation_paths):
        raise DatasetBuildError(f"Duplicate annotations found: {duplicate_filenames}")

    stem2image_path = {path.stem: path for path in image_paths}
    stem2annotation_path = {path.stem: path for path in annotation_paths}

    if stem2image_path.keys() != stem2annotation_path.keys():
        missing_images = stem2annotation_path.keys() - stem2image_path.keys()
        missing_annotations = stem2image_path.keys() - stem2annotation_path.keys()
        raise DatasetBuildError(
            f"Mismatched images/annotations. "
            f"Missing images: {missing_images}, "
            f"Missing annotations: {missing_annotations}"
        )

    tasks = set()
    records = []
    for stem, image_path in stem2image_path.items():
        annotation_path = stem2annotation_path[stem]

        if image_path.stem != annotation_path.stem:
            raise DatasetBuildError(
                f"Image and annotation names do not match: {image_path} {annotation_path}"
            )

        record = Record(
            image_path=image_path,
            annotation_path=annotation_path,
        )
        records.append(record)
        if not (task_name := image_path.parent.parent.name).startswith("task_"):
            raise DatasetBuildError(f"Invalid task name: {task_name} for {image_path}")
        tasks.add(task_name)

    return records, tasks


def delete_dataset(dataset_layout: DatasetLayout) -> None:
    shutil.rmtree(dataset_layout.canonical, ignore_errors=True)


def _ensure_empty_directory(directory: Path) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise DatasetBuildError(
            f"Directory is not empty: {directory}\n"
            "Refusing to build into a non-empty directory.\n"
            "Delete it explicitly or use a new dataset root."
        )


@dataclass(frozen=True)
class RecordPlan:
    stem: str
    src_image: Path
    src_annotation: Path
    dst_image: Path
    dst_annotation: Path
    split: DatasetSplit


@dataclass(frozen=True)
class DatasetPlan:
    layout: DatasetLayout
    tasks: list[str]
    record_copies: list[RecordPlan]


def plan_dataset(
    records: list[Record],
    tasks: list[str],
    layout: DatasetLayout,
    splitter: Splitter,
) -> DatasetPlan:
    """Plan a dataset build."""
    record_copies = []
    for record in records:
        record_copies.append(
            RecordPlan(
                stem=record.stem,
                src_image=record.image_path,
                src_annotation=record.annotation_path,
                dst_image=layout.records / record.image_path.name,
                dst_annotation=layout.records / record.annotation_path.name,
                split=splitter.split(record),
            )
        )
    assert len(record_copies) == len(set(record_copies))
    return DatasetPlan(layout=layout, tasks=tasks, record_copies=record_copies)


def execute_dataset_plan(plan: DatasetPlan) -> Path:
    """Execute a dataset plan."""
    plan.layout.canonical.mkdir(parents=True)
    logger.info(f"Found {len(plan.tasks)} tasks")
    pv.save_txt("\n".join(plan.tasks), plan.layout.canonical / "tasks.txt")

    logger.info(
        f"Copying {len(plan.record_copies)} records to {plan.layout.records.absolute()}"
    )
    plan.layout.records.mkdir()
    splits = {split: [] for split in DatasetSplit}
    for record_copy in pv.pbar(plan.record_copies):
        shutil.copy(record_copy.src_image.absolute(), record_copy.dst_image.absolute())
        shutil.copy(
            record_copy.src_annotation.absolute(), record_copy.dst_annotation.absolute()
        )
        splits[record_copy.split].append(record_copy)

    plan.layout.splits.mkdir()
    for split, split_records in splits.items():
        logger.info(f"{split}: {len(split_records)} records")
        split_path = plan.layout.splits / f"{split.value}.txt"
        pv.save_txt("\n".join(record.stem for record in split_records), split_path)

    return plan.layout.canonical


def build_dataset(dataset_layout: DatasetLayout, splitter: Splitter) -> Path:
    """Build a dataset from a directory of images and annotations."""

    _ensure_empty_directory(dataset_layout.canonical)

    records, tasks = records_from_cvat_raw(dataset_layout.cvat_raw)
    plan = plan_dataset(records, tasks, dataset_layout, splitter)
    return execute_dataset_plan(plan)
