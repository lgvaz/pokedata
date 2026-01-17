from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    image_path: Path
    annotation_path: Path
    stem: str
