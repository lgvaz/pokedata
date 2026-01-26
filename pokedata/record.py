from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    image_path: Path
    annotation_path: Path

    @property
    def stem(self) -> str:
        return self.image_path.stem

    def __post_init__(self):
        image_stem = self.image_path.stem
        annotation_stem = self.annotation_path.stem

        if image_stem != annotation_stem:
            raise ValueError(
                f"Stem mismatch: image={image_stem}, annotation={annotation_stem}"
            )
