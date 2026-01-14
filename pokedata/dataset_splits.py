from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
from typing import Dict, List

from pokedata.record import Record


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

    def split_records(self, records: List[Record]) -> Dict[DatasetSplit, List[Record]]:
        splits = {
            split: [r for r in records if self.split(r) == split]
            for split in DatasetSplit
        }
        if sum(len(split_records) for split_records in splits.values()) != len(records):
            raise ValueError(
                f"The sums of the splits are not equal to the total number of records: "
                f"{sum(len(split_records) for split_records in splits.values())} != {len(records)}"
            )
        return splits


def compute_first_hash_byte(stem: str, seed: int) -> int:
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
