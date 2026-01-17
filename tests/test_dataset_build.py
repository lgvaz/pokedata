from pathlib import Path

import pytest

from pokedata.dataset_build import (
    DatasetBuildError,
    build_dataset,
    plan_dataset,
    DatasetPlan,
    RecordPlan,
    execute_dataset_plan,
    records_from_cvat_raw,
)
from pokedata.dataset_layout import DatasetLayout
from pokedata.dataset_splits import DatasetSplit, StaticSplitter


def test_records_from_cvat_raw_single_task(tmp_path):
    cvat_raw = tmp_path / "cvat_raw"
    task_dir = cvat_raw / "task_123"
    task_dir.mkdir(parents=True)

    (task_dir / "x.png").write_bytes(b"x")
    (task_dir / "x.xml").write_text("<xml />")

    records, tasks = records_from_cvat_raw(cvat_raw)

    assert len(records) == 1
    assert records[0].stem == "x"
    assert records[0].image_path.name == "x.png"
    assert records[0].annotation_path.name == "x.xml"

    assert tasks == {"task_123"}


def test_records_from_cvat_raw_fails_on_duplicate_stems_across_tasks(tmp_path):
    # TODO: Implement checking of same stem in different tasks. Should it error out here or in build_dataset?
    cvat_raw = tmp_path / "cvat_raw"

    for task in ["task_1", "task_2"]:
        task_dir = cvat_raw / task
        task_dir.mkdir(parents=True)
        (task_dir / "x.png").write_bytes(b"x")
        (task_dir / "x.xml").write_text("<xml />")

    with pytest.raises(DatasetBuildError):
        records_from_cvat_raw(cvat_raw)


def test_records_from_cvat_raw_fails_when_annotation_is_missing(tmp_path):
    cvat_raw = tmp_path / "cvat_raw"
    task_dir = cvat_raw / "task_1"
    task_dir.mkdir(parents=True)

    # image without annotation
    (task_dir / "x.png").write_bytes(b"x")

    with pytest.raises(DatasetBuildError, match="Mismatched images/annotations"):
        records_from_cvat_raw(cvat_raw)


def test_records_from_cvat_raw_fails_on_mixed_mismatch(tmp_path):
    cvat_raw = tmp_path / "cvat_raw"
    task_dir = cvat_raw / "task_1"
    task_dir.mkdir(parents=True)

    # image with no annotation
    (task_dir / "a.png").write_bytes(b"a")

    # annotation with no image
    (task_dir / "b.xml").write_text("<xml />")

    with pytest.raises(DatasetBuildError, match="Mismatched images/annotations"):
        records_from_cvat_raw(cvat_raw)


def test_plan_dataset_integrates_splitter_and_layout(record_factory):
    records = [
        record_factory(stem="x"),
        record_factory(stem="y"),
    ]
    splitter = StaticSplitter(
        {
            "x": DatasetSplit.TRAIN,
            "y": DatasetSplit.VAL,
        }
    )

    layout = DatasetLayout(root_dir=Path("data"))
    tasks = ["task_123"]

    plan = plan_dataset(
        records=records,
        tasks=tasks,
        layout=layout,
        splitter=splitter,
    )

    assert plan.tasks == tasks

    copy = plan.record_copies[0]
    assert copy.dst_image == layout.records / records[0].image_path.name
    assert copy.split == DatasetSplit.TRAIN
    copy = plan.record_copies[1]
    assert copy.dst_image == layout.records / records[1].image_path.name
    assert copy.split == DatasetSplit.VAL


def test_execute_dataset_plan_copies_files_and_writes_splits(tmp_path):
    layout = DatasetLayout(root_dir=tmp_path)

    cvat_raw = tmp_path / "cvat_raw"
    cvat_raw.mkdir()

    img = cvat_raw / "a.png"
    ann = cvat_raw / "a.xml"
    img.write_bytes(b"image")
    ann.write_text("<xml />")

    plan = DatasetPlan(
        layout=layout,
        tasks=["task_1"],
        record_copies=[
            RecordPlan(
                stem="a",
                src_image=img,
                src_annotation=ann,
                dst_image=layout.records / "a.png",
                dst_annotation=layout.records / "a.xml",
                split=DatasetSplit.TRAIN,
            )
        ],
    )

    result = execute_dataset_plan(plan)

    assert result == layout.canonical

    assert (layout.records / "a.png").exists()
    assert (layout.records / "a.xml").exists()

    tasks_file = layout.canonical / "tasks.txt"
    assert tasks_file.exists()
    assert tasks_file.read_text().strip() == "task_1"

    train_file = layout.splits / "train.txt"
    val_file = layout.splits / "val.txt"
    test_file = layout.splits / "test.txt"

    assert train_file.exists()
    assert train_file.read_text().strip() == "a"
    # empty splits still exist
    assert val_file.exists()
    assert test_file.exists()


def test_build_dataset_creates_canonical_dataset(tmp_path):
    layout = DatasetLayout(root_dir=tmp_path)

    # layout.cvat_raw / <task_name> / files
    task_dir = layout.cvat_raw / "task_123"
    task_dir.mkdir(parents=True)

    (task_dir / "x.png").write_bytes(b"x")
    (task_dir / "x.xml").write_text("<xml />")

    splitter = StaticSplitter({"x": DatasetSplit.TRAIN})

    result = build_dataset(layout, splitter)

    assert result == layout.canonical
    assert result.exists()

    assert layout.records.exists()
    assert layout.splits.exists()

    assert (layout.records / "x.png").exists()
    assert (layout.records / "x.xml").exists()

    tasks_file = layout.canonical / "tasks.txt"
    assert tasks_file.exists()
    assert tasks_file.read_text().strip() == "task_123"

    train_file = layout.splits / "train.txt"
    assert train_file.exists()
    assert train_file.read_text().strip() == "x"


def test_build_dataset_refuses_non_empty_canonical_dir(tmp_path):
    layout = DatasetLayout(root_dir=tmp_path)

    layout.canonical.mkdir(parents=True)
    (layout.canonical / "junk.txt").write_text("nope")

    with pytest.raises(DatasetBuildError):
        build_dataset(layout, splitter=None)
