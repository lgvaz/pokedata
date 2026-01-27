from dataclasses import dataclass
from pathlib import Path

__all__ = ["DatasetLayout"]


@dataclass(frozen=True)
class DatasetLayout:
    dataset_repo: Path

    @property
    def cvat_raw(self) -> Path:
        return self.dataset_repo / "cvat_raw"

    @property
    def canonical(self) -> Path:
        return self.dataset_repo / "canonical"

    @property
    def records(self) -> Path:
        return self.canonical / "records"

    @property
    def splits(self) -> Path:
        return self.canonical / "splits"
