import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "common"))
from pair_utils import (
    build_span_mask,
    compute_class_weights,
    compute_log_priors,
    explicit_quads,
    word_span_to_subword_range,
)


def test_explicit_quads_filter():
    quads = [
        {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3, "category": "CAT_A"},
        {"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3, "category": "CAT_B"},
        {"a_start": 1, "a_end": 2, "o_start": -1, "o_end": -1, "category": "CAT_C"},
    ]
    res = explicit_quads(quads)
    assert len(res) == 1
    assert res[0]["category"] == "CAT_A"


def test_word_span_to_subword_range():
    # word_ids: [None, 0, 0, 1, 2, 2, None]
    word_ids = [None, 0, 0, 1, 2, 2, None]
    # span [0, 1) -> word 0 -> subwords [1, 2]
    rng = word_span_to_subword_range(word_ids, 0, 1)
    assert rng == (1, 2)

    # span [1, 3) -> words 1, 2 -> subwords [3, 5]
    rng2 = word_span_to_subword_range(word_ids, 1, 3)
    assert rng2 == (3, 5)

    # out of bounds / truncated word index
    rng_none = word_span_to_subword_range(word_ids, 10, 12)
    assert rng_none is None


def test_build_span_mask():
    seq_len = 6
    mask = build_span_mask(seq_len, (1, 3))
    assert mask == [0, 1, 1, 1, 0, 0]

    mask_none = build_span_mask(seq_len, None)
    assert mask_none == [0, 0, 0, 0, 0, 0]


def test_compute_class_weights():
    counts = {"catA": 100, "catB": 10, "catC": 0}
    weights = compute_class_weights(counts, num_classes=3, cap=15.0)
    # total = 110
    # catA: 110 / (3 * 100) = 0.3667
    # catB: 110 / (3 * 10) = 3.6667
    # catC: capped at 15.0
    assert abs(weights["catA"] - 110 / 300) < 1e-4
    assert abs(weights["catB"] - 110 / 30) < 1e-4
    assert weights["catC"] == 15.0


def test_compute_log_priors():
    import math

    counts = {0: 100, 1: 10}
    priors = compute_log_priors(counts, num_classes=3)
    assert len(priors) == 3
    # class 0: 100/110
    assert abs(priors[0] - math.log(100 / 110)) < 1e-4
    # class 1: 10/110
    assert abs(priors[1] - math.log(10 / 110)) < 1e-4
    # class 2 (zero count): epsilon floor
    assert priors[2] == math.log(1e-12)
