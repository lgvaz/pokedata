import pytest
from pathlib import Path

from pokedata.dataset_splits import DatasetSplit
from pokedata.record import Record


@pytest.fixture
def record_factory(tmp_path):
    """
    Factory for creating valid Record objects.

    - By default: NO filesystem IO (paths may or may not exist).
    - with_files=True: creates real image + annotation files.
    - Centralizes filename + path invariants.
    """

    def _make(
        *,
        stem: str,
        image_ext: str = ".png",
        annotation_ext: str = ".xml",
        with_files: bool = False,
        base_dir: Path | None = None,
    ):
        base = base_dir or tmp_path

        image_path = base / f"{stem}{image_ext}"
        annotation_path = base / f"{stem}{annotation_ext}"

        if with_files:
            image_path.parent.mkdir(parents=True, exist_ok=True)
            annotation_path.parent.mkdir(parents=True, exist_ok=True)

            image_path.write_bytes(b"fake image bytes")
            annotation_path.write_text("{}")

        return Record(
            image_path=image_path,
            annotation_path=annotation_path,
        )

    return _make


@pytest.fixture
def pinned_split_stems() -> tuple[list[str], list[DatasetSplit]]:
    """
    Create a tuple of lists of stems and splits.
    Returns known splits for usage with `CertIdSplitter` and
    `RatioSplitPolicy(train=0.8, val=0.1, test=0.1)`.
    """
    stems = [
        "RG123456789-+00000005-+front_laser",
        "RG123456789-+00000005-+back_laser",
        "RG123456789-+00000008-+front_laser",
        "RG123456789-+00000008-+back_laser",
        "RG123456789-+00000026-+front_laser",
        "RG123456789-+00000026-+back_laser",
        "RG123456789-+00000016-+front_laser",
        "RG123456789-+00000016-+back_laser",
    ]
    splits = [
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
        DatasetSplit.TEST,
        DatasetSplit.TRAIN,
        DatasetSplit.TRAIN,
    ]
    return stems, splits
