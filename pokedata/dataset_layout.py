from dataclasses import dataclass
from pathlib import Path

__all__ = ["DatasetLayout"]


@dataclass(frozen=True)
class DatasetLayout:
    root_dir: Path

    @property
    def cvat_raw(self) -> Path:
        return self.root_dir / "cvat_raw"

    @property
    def canonical(self) -> Path:
        return self.root_dir / "canonical"

    @property
    def records(self) -> Path:
        return self.canonical / "records"

    @property
    def splits(self) -> Path:
        return self.canonical / "splits"
