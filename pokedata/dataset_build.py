from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import polvo as pv
import hashlib


class DatasetBuildError(Exception):
    """Raised when dataset build fails."""

    pass


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass
class Record:
    image_path: Path
    annotation_path: Path
    stem: str


def build_dataset(dataset_path: Path) -> None:
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
        split = stem_to_split(record.stem)
        splits[split].append(record)

    return splits


def compute_first_hash_byte(stem: str, seed: int = 42) -> int:
    """Compute the first byte of the hash of a filename stem"""
    key = f"{seed}:{stem}".encode("utf-8")
    return hashlib.sha256(key).digest()[0]  # 0–255


def stem_to_split(stem: str, seed: int = 42) -> DatasetSplit:
    """Convert a filename stem to a train/val/test split."""
    b = compute_first_hash_byte(stem, seed)
    if b < 204:  # 204 / 256 ≈ 0.80
        return DatasetSplit.TRAIN
    elif b < 230:  # 230 / 256 ≈ 0.90
        return DatasetSplit.VAL
    else:
        return DatasetSplit.TEST
