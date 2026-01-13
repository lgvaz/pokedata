from pathlib import Path
from typing import Dict, List
import polvo as pv

from pokedata.dataset_splits import DatasetSplit, Splitter
from pokedata.record import Record


class DatasetBuildError(Exception):
    """Raised when dataset build fails."""

    pass


def build_dataset(
    dataset_path: Path, splitter: Splitter
) -> Dict[DatasetSplit, List[Record]]:
    """Build a dataset from a directory of images and annotations."""
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

    splits = {split: [] for split in DatasetSplit}
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
        split = splitter.split(record)
        splits[split].append(record)

    return splits
