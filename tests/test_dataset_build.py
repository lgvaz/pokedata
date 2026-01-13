"""Tests for pokedata.dataset_build module."""

from pathlib import Path
from unittest.mock import patch

import pytest

from pokedata.dataset_build import (
    DatasetBuildError,
    DatasetSplit,
    Record,
    build_dataset,
    compute_first_hash_byte,
    stem_to_split,
)


def test_compute_first_hash_byte():
    """Test compute_first_hash_byte."""

    assert compute_first_hash_byte("test_image_0", 42) == 15
    assert compute_first_hash_byte("test_image_3", 42) == 183
    assert compute_first_hash_byte("test_image_4", 42) == 109
    assert compute_first_hash_byte("test_image_5", 42) == 205
    assert compute_first_hash_byte("test_image_7", 42) == 247


def test_stem_to_split():
    """Test stem_to_split."""
    # fix tests like above, no extension, only stem, results should be equal above
    assert stem_to_split("test_image_0", 42) == DatasetSplit.TRAIN
    assert stem_to_split("test_image_3", 42) == DatasetSplit.TRAIN
    assert stem_to_split("test_image_4", 42) == DatasetSplit.TRAIN
    assert stem_to_split("test_image_5", 42) == DatasetSplit.VAL
    assert stem_to_split("test_image_7", 42) == DatasetSplit.TEST


class TestBuildDatasetSuccess:
    """Tests for build_dataset function - success cases."""

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_success_single_file(self, mock_get_files, tmp_path):
        """Test successful dataset build with a single matching image/annotation pair."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        image_path = dataset_path / "test_image_0.png"
        annotation_path = dataset_path / "test_image_0.xml"

        mock_get_files.side_effect = [[image_path], [annotation_path]]

        splits = build_dataset(dataset_path)

        assert len(splits[DatasetSplit.TRAIN]) == 1
        assert len(splits[DatasetSplit.VAL]) == 0
        assert len(splits[DatasetSplit.TEST]) == 0

        record = splits[DatasetSplit.TRAIN][0]
        assert isinstance(record, Record)
        assert record.image_path == image_path
        assert record.annotation_path == annotation_path
        assert record.stem == "test_image_0"

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_success_multiple_files(self, mock_get_files, tmp_path):
        """Test successful dataset build with multiple matching image/annotation pairs."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        # Create paths for multiple files
        image_paths = [
            dataset_path / "test_image_0.png",
            dataset_path / "test_image_3.png",
            dataset_path / "test_image_4.png",
            dataset_path / "test_image_5.png",
            dataset_path / "test_image_7.png",
        ]
        annotation_paths = [
            dataset_path / "test_image_0.xml",
            dataset_path / "test_image_3.xml",
            dataset_path / "test_image_4.xml",
            dataset_path / "test_image_5.xml",
            dataset_path / "test_image_7.xml",
        ]

        mock_get_files.side_effect = [image_paths, annotation_paths]

        splits = build_dataset(dataset_path)

        # Verify splits are correct
        assert len(splits[DatasetSplit.TRAIN]) == 3
        assert len(splits[DatasetSplit.VAL]) == 1
        assert len(splits[DatasetSplit.TEST]) == 1

        # Verify records are correctly assigned
        train_record = splits[DatasetSplit.TRAIN][0]
        assert train_record.stem == "test_image_0"
        val_record = splits[DatasetSplit.VAL][0]
        assert val_record.stem == "test_image_5"
        test_record = splits[DatasetSplit.TEST][0]
        assert test_record.stem == "test_image_7"

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_success_empty_dataset(self, mock_get_files, tmp_path):
        """Test build_dataset with empty dataset (no files)."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        mock_get_files.side_effect = [[], []]

        splits = build_dataset(dataset_path)

        assert isinstance(splits, dict)
        assert DatasetSplit.TRAIN in splits
        assert DatasetSplit.VAL in splits
        assert DatasetSplit.TEST in splits
        assert len(splits[DatasetSplit.TRAIN]) == 0
        assert len(splits[DatasetSplit.VAL]) == 0
        assert len(splits[DatasetSplit.TEST]) == 0

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_returns_correct_structure(self, mock_get_files, tmp_path):
        """Test that build_dataset returns the correct dictionary structure."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        image_path = dataset_path / "test_image_0.png"
        annotation_path = dataset_path / "test_image_0.xml"

        mock_get_files.side_effect = [[image_path], [annotation_path]]

        splits = build_dataset(dataset_path)

        # Verify structure
        assert isinstance(splits, dict)
        assert len(splits) == 3
        assert all(isinstance(key, DatasetSplit) for key in splits.keys())
        assert all(isinstance(value, list) for value in splits.values())


class TestBuildDatasetErrors:
    """Tests for build_dataset function - error cases."""

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_duplicate_images(self, mock_get_files, tmp_path):
        """Test that duplicate images raise DatasetBuildError."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        image_path = dataset_path / "test_image_0.png"
        # Simulate duplicate by returning the same path twice
        mock_get_files.side_effect = [
            [image_path, image_path],  # Duplicate PNG files
            [dataset_path / "test_image_0.xml"],
        ]

        with pytest.raises(DatasetBuildError, match="Duplicate images found"):
            build_dataset(dataset_path)

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_duplicate_annotations(self, mock_get_files, tmp_path):
        """Test that duplicate annotations raise DatasetBuildError."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        annotation_path = dataset_path / "test_image_0.xml"
        # Simulate duplicate by returning the same path twice
        mock_get_files.side_effect = [
            [dataset_path / "test_image_0.png"],
            [annotation_path, annotation_path],  # Duplicate XML files
        ]

        with pytest.raises(DatasetBuildError, match="Duplicate annotations found"):
            build_dataset(dataset_path)

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_missing_annotations(self, mock_get_files, tmp_path):
        """Test that missing annotations raise DatasetBuildError."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        mock_get_files.side_effect = [
            [
                dataset_path / "test_image_0.png",
                dataset_path / "test_image_3.png",
            ],  # Two images
            [dataset_path / "test_image_0.xml"],  # Only one annotation
        ]

        with pytest.raises(DatasetBuildError, match="Mismatched images/annotations"):
            build_dataset(dataset_path)

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_missing_images(self, mock_get_files, tmp_path):
        """Test that missing images raise DatasetBuildError."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        mock_get_files.side_effect = [
            [dataset_path / "test_image_0.png"],  # Only one image
            [
                dataset_path / "test_image_0.xml",
                dataset_path / "test_image_3.xml",
            ],  # Two annotations
        ]

        with pytest.raises(DatasetBuildError, match="Mismatched images/annotations"):
            build_dataset(dataset_path)

    @patch("pokedata.dataset_build.pv.get_files")
    def test_build_dataset_name_mismatch(self, mock_get_files, tmp_path):
        """Test that mismatched image/annotation names raise DatasetBuildError."""
        dataset_path = tmp_path / "dataset"
        dataset_path.mkdir()

        # Create paths where stems don't match
        # This will trigger the "Mismatched images/annotations" error
        # because the stems are different, so they won't match in the dictionary keys
        image_path = dataset_path / "test_image_0.png"
        annotation_path = dataset_path / "test_image_3.xml"

        mock_get_files.side_effect = [
            [image_path],
            [annotation_path],
        ]

        with pytest.raises(DatasetBuildError, match="Mismatched images/annotations"):
            build_dataset(dataset_path)


class TestRecord:
    """Tests for Record dataclass."""

    def test_record_creation(self, tmp_path):
        """Test that Record can be created with correct fields."""
        image_path = tmp_path / "test_image_0.png"
        annotation_path = tmp_path / "test_image_0.xml"
        stem = "test_image_0"

        record = Record(
            image_path=image_path, annotation_path=annotation_path, stem=stem
        )

        assert record.image_path == image_path
        assert record.annotation_path == annotation_path
        assert record.stem == stem


class TestDatasetBuildError:
    """Tests for DatasetBuildError exception."""

    def test_dataset_build_error_is_exception(self):
        """Test that DatasetBuildError is an Exception."""
        assert issubclass(DatasetBuildError, Exception)

    def test_dataset_build_error_can_be_raised(self):
        """Test that DatasetBuildError can be raised and caught."""
        with pytest.raises(DatasetBuildError):
            raise DatasetBuildError("Test error")

    def test_dataset_build_error_message(self):
        """Test that DatasetBuildError preserves error message."""
        error_msg = "Custom error message"
        with pytest.raises(DatasetBuildError, match=error_msg):
            raise DatasetBuildError(error_msg)


class TestDatasetSplit:
    """Tests for DatasetSplit enum."""

    def test_dataset_split_values(self):
        """Test that DatasetSplit enum has correct values."""
        assert DatasetSplit.TRAIN.value == "train"
        assert DatasetSplit.VAL.value == "val"
        assert DatasetSplit.TEST.value == "test"

    def test_dataset_split_all_values_present(self):
        """Test that all expected DatasetSplit values are present."""
        expected_values = {"train", "val", "test"}
        actual_values = {split.value for split in DatasetSplit}
        assert actual_values == expected_values
