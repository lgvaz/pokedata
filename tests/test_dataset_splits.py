from pokedata.dataset_splits import compute_first_hash_byte
from pokedata.dataset_splits import DatasetSplit, Splitter
from pokedata.record import Record
from pokedata.dataset_splits import RatioSplitPolicy, HashSplitter


def test_compute_first_hash_byte():
    """Test compute_first_hash_byte."""

    assert compute_first_hash_byte("test_image_0", 42) == 15
    assert compute_first_hash_byte("test_image_3", 42) == 183
    assert compute_first_hash_byte("test_image_4", 42) == 109
    assert compute_first_hash_byte("test_image_5", 42) == 205
    assert compute_first_hash_byte("test_image_7", 42) == 247


def test_stem_to_split():
    """Test HashSplitter with RatioSplitPolicy."""
    # Using the same splitter configuration that produces the expected results
    # Based on hash scores: 0->0.0586, 3->0.7148, 4->0.4258, 5->0.8008, 7->0.9648
    # We need train < 0.75, val < 0.9, test < 1.0
    policy = RatioSplitPolicy(train=0.75, val=0.15, test=0.10)
    splitter = HashSplitter(policy=policy, seed=42)

    record_0 = Record(image_path=None, annotation_path=None, stem="test_image_0")
    record_3 = Record(image_path=None, annotation_path=None, stem="test_image_3")
    record_4 = Record(image_path=None, annotation_path=None, stem="test_image_4")
    record_5 = Record(image_path=None, annotation_path=None, stem="test_image_5")
    record_7 = Record(image_path=None, annotation_path=None, stem="test_image_7")

    assert splitter.split(record_0) == DatasetSplit.TRAIN
    assert splitter.split(record_3) == DatasetSplit.TRAIN
    assert splitter.split(record_4) == DatasetSplit.TRAIN
    assert splitter.split(record_5) == DatasetSplit.VAL
    assert splitter.split(record_7) == DatasetSplit.TEST
