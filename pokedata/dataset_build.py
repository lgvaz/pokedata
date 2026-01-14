from pathlib import Path
import shutil
from typing import Dict, List, Set, Tuple, TypeAlias
from loguru import logger
import polvo as pv

from pokedata.dataset_layout import DatasetLayout
from pokedata.dataset_splits import DatasetSplit, Splitter
from pokedata.record import Record

__all__ = ["DatasetBuildError", "build_dataset"]


class DatasetBuildError(Exception):
    """Raised when dataset build fails."""

    pass


CvatTask: TypeAlias = str


def records_from_cvat_raw(dataset_path: Path) -> Tuple[List[Record], Set[CvatTask]]:
    image_paths = pv.get_files(dataset_path, extensions=[".png"])
    annotation_paths = pv.get_files(dataset_path, extensions=[".xml"])

    # assert there are no duplicate stems
    if len(image_paths) != len(set(image_paths)):
        raise DatasetBuildError(f"Duplicate images found in {dataset_path}")
    if len(annotation_paths) != len(set(annotation_paths)):
        raise DatasetBuildError(f"Duplicate annotations found in {dataset_path}")

    stem2image_path = {path.stem: path for path in image_paths}
    stem2annotation_path = {path.stem: path for path in annotation_paths}

    if stem2image_path.keys() != stem2annotation_path.keys():
        missing_images = stem2image_path.keys() - stem2annotation_path.keys()
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
            stem=stem,
        )
        records.append(record)
        tasks.add(image_path.parent.parent.name)

    return records, tasks


def delete_dataset(dataset_layout: DatasetLayout) -> None:
    shutil.rmtree(dataset_layout.canonical, ignore_errors=True)


def build_dataset(
    dataset_layout: DatasetLayout, splitter: Splitter
) -> Dict[DatasetSplit, List[Record]]:
    """Build a dataset from a directory of images and annotations."""

    canonical = dataset_layout.canonical
    if canonical.exists() and any(canonical.iterdir()):
        raise DatasetBuildError(
            f"Canonical dataset directory is not empty: {canonical}\n"
            "Refusing to build into a non-empty directory.\n"
            "Delete it explicitly (`pokedata dataset clean`) or use a new dataset root."
        )

    records, tasks = records_from_cvat_raw(dataset_layout.cvat_raw)
    canonical.mkdir(parents=True, exist_ok=True)

    logger.info(f"Found {len(tasks)} tasks")
    pv.save_txt("\n".join(tasks), dataset_layout.canonical / "tasks.txt")

    logger.info(f"Found {len(records)} records")
    splits = splitter.split_records(records)
    for split, split_records in splits.items():
        logger.info(f"{split}: {len(split_records)} records")

    for split, split_records in splits.items():
        dataset_layout.splits.mkdir(parents=True, exist_ok=True)
        split_path = dataset_layout.splits / f"{split.value}.txt"
        logger.info(f"Writing {split.value} split to {split_path}")
        pv.save_txt(
            "\n".join(record.image_path.stem for record in split_records), split_path
        )

    logger.info(f"Copying {len(records)} records to output directory")
    dataset_layout.records.mkdir(parents=True, exist_ok=True)
    for record in pv.pbar(records):
        copy_record_to_directory(record, dataset_layout.records)


def copy_record_to_directory(record: Record, output_dir: Path) -> Record:
    new_image_path = output_dir / record.image_path.name
    new_annotation_path = output_dir / record.annotation_path.name

    shutil.copy(record.image_path, new_image_path)
    shutil.copy(record.annotation_path, new_annotation_path)

    new_record = Record(
        image_path=new_image_path,
        annotation_path=new_annotation_path,
        stem=record.stem,
    )
    return new_record
