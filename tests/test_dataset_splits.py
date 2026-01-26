from pathlib import Path
import pytest

from pokedata.dataset_splits import (
    CardIdentity,
    CertIdSplitter,
    StaticSplitter,
    SplitScore,
    compute_first_hash_byte,
    extract_card_identity,
)
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

    assert compute_first_hash_byte("00000005", 42) == 3
    assert compute_first_hash_byte("00000008", 42) == 212
    assert compute_first_hash_byte("00000026", 42) == 254
    assert compute_first_hash_byte("00000016", 42) == 13


def test_extract_card_identity():
    """Test extract_card_identity."""
    expected = CardIdentity(
        order_id="RG123456789", certificate_id="12345678", orientation="front"
    )
    assert extract_card_identity("RG123456789-+12345678-+front_laser") == expected

    expected = CardIdentity(
        order_id="RG123456789", certificate_id="12345678", orientation="back"
    )
    assert extract_card_identity("RG123456789-+12345678-+back_laser") == expected
    expected = CardIdentity(
        order_id="RG123456789", certificate_id="12345678", orientation="back"
    )
    assert extract_card_identity("RG123456789_part4-+12345678-+back_laser") == expected
    expected = CardIdentity(
        order_id="RG123456789", certificate_id="00000005", orientation="front"
    )
    assert extract_card_identity("RG123456789-+00000005-+front_laser") == expected

    # invalid orientation
    with pytest.raises(ValueError):
        extract_card_identity("RG123456789-+12345678-+invalid_orientation.png")
    # 8 digits for certificate id
    with pytest.raises(ValueError):
        extract_card_identity("RG123456789-+1234567-+front_laser")
    with pytest.raises(ValueError):
        extract_card_identity("RG123456789-+123456789-+front_laser")
    with pytest.raises(ValueError):
        extract_card_identity("RG123456789-+134invalid-+front_laser")
    # 9 digits for order id
    with pytest.raises(ValueError):
        extract_card_identity("RG12345678-+12345678-+front_laser")
    with pytest.raises(ValueError):
        extract_card_identity("RG1234567890-+12345678-+front_laser")
    with pytest.raises(ValueError):
        extract_card_identity("RGa123456789-+12345678-+front_laser")
    with pytest.raises(ValueError):
        extract_card_identity("RGa12345678-+12345678-+front_laser")


def test_dummy_splitter(record_factory):
    """Test DummySplitter."""
    mapping = {
        "x": DatasetSplit.TRAIN,
        "y": DatasetSplit.VAL,
        "z": DatasetSplit.TEST,
    }
    splitter = StaticSplitter(mapping=mapping)

    assert splitter.split(record_factory(stem="x")) == DatasetSplit.TRAIN
    assert splitter.split(record_factory(stem="y")) == DatasetSplit.VAL
    assert splitter.split(record_factory(stem="z")) == DatasetSplit.TEST


def test_split_records_detects_missing_assignment(record_factory):
    records = [record_factory(stem="a")]
    splitter = StaticSplitter({})
    with pytest.raises(KeyError):
        splitter.split_records(records)


def test_hash_splitter():
    """Test HashSplitter with RatioSplitPolicy."""
    # Using the same splitter configuration that produces the expected results
    # Based on hash scores: 0->0.0586, 3->0.7148, 4->0.4258, 5->0.8008, 7->0.9648
    # We need train < 0.75, val < 0.9, test < 1.0
    policy = RatioSplitPolicy(train=0.8, val=0.10, test=0.10)
    splitter = HashSplitter(policy=policy, seed=42)

    record_0 = Record(
        image_path=Path("test_image_0.png"), annotation_path=Path("test_image_0.xml")
    )
    record_3 = Record(
        image_path=Path("test_image_3.png"), annotation_path=Path("test_image_3.xml")
    )
    record_4 = Record(
        image_path=Path("test_image_4.png"), annotation_path=Path("test_image_4.xml")
    )
    record_5 = Record(
        image_path=Path("test_image_5.png"), annotation_path=Path("test_image_5.xml")
    )
    record_7 = Record(
        image_path=Path("test_image_7.png"), annotation_path=Path("test_image_7.xml")
    )

    assert splitter.split(record_0) == DatasetSplit.TRAIN
    assert splitter.split(record_3) == DatasetSplit.TRAIN
    assert splitter.split(record_4) == DatasetSplit.TRAIN
    assert splitter.split(record_5) == DatasetSplit.VAL
    assert splitter.split(record_7) == DatasetSplit.TEST


def test_ratio_split_policy_thresholds():
    policy = RatioSplitPolicy(train=0.7, val=0.2, test=0.1)

    assert policy.split(SplitScore(0.0)) == DatasetSplit.TRAIN
    assert policy.split(SplitScore(0.699999)) == DatasetSplit.TRAIN

    assert policy.split(SplitScore(0.7)) == DatasetSplit.VAL
    assert policy.split(SplitScore(0.899999)) == DatasetSplit.VAL

    assert policy.split(SplitScore(0.9)) == DatasetSplit.TEST
    assert policy.split(SplitScore(0.999999)) == DatasetSplit.TEST


def test_ratio_split_policy_ratios_must_sum_to_one():
    with pytest.raises(ValueError):
        RatioSplitPolicy(train=0.5, val=0.3, test=0.3)


def test_cert_id_splitter(record_factory, pinned_split_stems):
    """Test CertIdSplitter."""
    policy = RatioSplitPolicy(train=0.8, val=0.1, test=0.1)
    splitter = CertIdSplitter(policy=policy, seed=42)

    stems, splits = pinned_split_stems
    records = [record_factory(stem=stem) for stem in stems]

    for record, split in zip(records, splits):
        assert splitter.split(record) == split
