from pathlib import Path

from pokedata.dataset_layout import DatasetLayout


def test_dataset_layout_paths():
    layout = DatasetLayout(dataset_repo=Path("data"))

    assert layout.cvat_raw == Path("data/cvat_raw")
    assert layout.canonical == Path("data/canonical")
    assert layout.records == Path("data/canonical/records")
    assert layout.splits == Path("data/canonical/splits")
