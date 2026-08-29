import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "common"))
from bio_labels import build_word_bio, decode_bio_spans


def test_round_trip_single_span():
    tags = build_word_bio(5, [(1, 3)])
    assert tags == ["O", "B", "I", "O", "O"]
    assert decode_bio_spans(tags) == [(1, 3)]


def test_round_trip_multiple_spans():
    tags = build_word_bio(6, [(0, 1), (3, 5)])
    assert decode_bio_spans(tags) == [(0, 1), (3, 5)]


def test_implicit_span_skipped():
    tags = build_word_bio(4, [(-1, -1)])
    assert tags == ["O", "O", "O", "O"]
    assert decode_bio_spans(tags) == []
