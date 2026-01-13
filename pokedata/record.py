from dataclasses import dataclass
from pathlib import Path


@dataclass
class Record:
    image_path: Path
    annotation_path: Path
    stem: str
