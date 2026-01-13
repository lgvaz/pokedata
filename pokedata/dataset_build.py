from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List
import polvo as pv
import hashlib
from abc import ABC, abstractmethod


class DatasetBuildError(Exception):
    """Raised when dataset build fails."""

    pass


@dataclass
class Record:
    image_path: Path
    annotation_path: Path
    stem: str


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


@dataclass(frozen=True)
class SplitScore:
    score: float  # 0.0 <= score < 1.0

    def __post_init__(self):
        if not 0.0 <= self.score < 1.0:
            raise ValueError("SplitScore must be in [0, 1)")


class SplitPolicy(ABC):
    @abstractmethod
    def split(self, score: SplitScore) -> DatasetSplit: ...


@dataclass(frozen=True)
class RatioSplitPolicy(SplitPolicy):
    train: float
    val: float
    test: float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if not abs(total - 1.0) < 1e-9:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        object.__setattr__(
            self,
            "_thresholds",
            (
                (self.train, DatasetSplit.TRAIN),
                (self.train + self.val, DatasetSplit.VAL),
                (1.0, DatasetSplit.TEST),
            ),
        )

    def split(self, score: SplitScore) -> DatasetSplit:
        for limit, split in self._thresholds:
            if score.score < limit:
                return split

        raise RuntimeError("Unreachable")


class Splitter(ABC):
    @abstractmethod
    def split(self, record: Record) -> DatasetSplit: ...


def compute_first_hash_byte(stem: str, seed: int = 42) -> int:
    """Compute the first byte of the hash of a filename stem"""
    key = f"{seed}:{stem}".encode("utf-8")
    return hashlib.sha256(key).digest()[0]


def compute_hash_score(stem: str, seed: int) -> SplitScore:
    """Compute the score for a filename stem"""
    hash_byte = compute_first_hash_byte(stem, seed)
    return SplitScore(score=hash_byte / 256.0)


class HashSplitter(Splitter):
    def __init__(self, policy: SplitPolicy, seed: int):
        self.seed = seed
        self.policy = policy

    def split(self, record: Record) -> DatasetSplit:
        split_score = compute_hash_score(record.stem, self.seed)
        return self.policy.split(split_score)


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
