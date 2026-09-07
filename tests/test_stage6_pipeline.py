import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pipeline"))
from assemble_quads import SentencePredictions, assemble_quads
from evaluate import evaluate_quads, is_implicit_quad, quad_key


def test_is_implicit_quad():
    assert not is_implicit_quad({"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3})
    assert is_implicit_quad({"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3})
    assert is_implicit_quad({"a_start": 0, "a_end": 1, "o_start": -1, "o_end": -1})
    assert is_implicit_quad({"a_start": -1, "a_end": -1, "o_start": -1, "o_end": -1})


def test_assemble_quads_explicit_and_implicit():
    preds = SentencePredictions(
        aspect_spans=[(0, 1), (5, 6)],
        opinion_spans=[(2, 3), (8, 9)],
        explicit_pairs={(0, 1, 2, 3)},
        pair_category={(0, 1, 2, 3): "FOOD#QUALITY"},
        pair_sentiment={(0, 1, 2, 3): "POSITIVE"},
        implicit_aspect_flag={(8, 9): True},
        implicit_aspect_category={(8, 9): "SERVICE#GENERAL"},
        implicit_aspect_sentiment={(8, 9): "NEGATIVE"},
        implicit_opinion_flag={(5, 6): True},
        implicit_opinion_category={(5, 6): "AMBIENCE#GENERAL"},
        implicit_opinion_sentiment={(5, 6): "NEUTRAL"},
        fully_implicit_flag=True,
        fully_implicit_category="RESTAURANT#PRICES",
        fully_implicit_sentiment="NEGATIVE",
    )

    quads = assemble_quads(preds)
    assert len(quads) == 4

    # 1. explicit pair
    assert {"a_start": 0, "a_end": 1, "category": "FOOD#QUALITY", "sentiment": "POSITIVE", "o_start": 2, "o_end": 3} in quads
    # 2. implicit aspect (anchored on opinion 8,9)
    assert {"a_start": -1, "a_end": -1, "category": "SERVICE#GENERAL", "sentiment": "NEGATIVE", "o_start": 8, "o_end": 9} in quads
    # 3. implicit opinion (anchored on aspect 5,6)
    assert {"a_start": 5, "a_end": 6, "category": "AMBIENCE#GENERAL", "sentiment": "NEUTRAL", "o_start": -1, "o_end": -1} in quads
    # 4. fully implicit
    assert {"a_start": -1, "a_end": -1, "category": "RESTAURANT#PRICES", "sentiment": "NEGATIVE", "o_start": -1, "o_end": -1} in quads


def test_evaluate_quads_perfect_match():
    gold = {
        0: [
            {"a_start": 0, "a_end": 1, "category": "FOOD#QUALITY", "sentiment": "POSITIVE", "o_start": 2, "o_end": 3},
            {"a_start": -1, "a_end": -1, "category": "SERVICE#GENERAL", "sentiment": "NEGATIVE", "o_start": 8, "o_end": 9},
        ]
    }
    pred = {
        0: [
            {"a_start": 0, "a_end": 1, "category": "FOOD#QUALITY", "sentiment": "POSITIVE", "o_start": 2, "o_end": 3},
            {"a_start": -1, "a_end": -1, "category": "SERVICE#GENERAL", "sentiment": "NEGATIVE", "o_start": 8, "o_end": 9},
        ]
    }
    res = evaluate_quads(pred, gold)
    assert res["all"]["f1"] == 1.0
    assert res["explicit"]["f1"] == 1.0
    assert res["implicit"]["f1"] == 1.0


def test_evaluate_quads_partial_miss():
    gold = {
        0: [
            {"a_start": 0, "a_end": 1, "category": "FOOD#QUALITY", "sentiment": "POSITIVE", "o_start": 2, "o_end": 3},
            {"a_start": -1, "a_end": -1, "category": "SERVICE#GENERAL", "sentiment": "NEGATIVE", "o_start": 8, "o_end": 9},
        ]
    }
    pred = {
        0: [
            {"a_start": 0, "a_end": 1, "category": "FOOD#QUALITY", "sentiment": "POSITIVE", "o_start": 2, "o_end": 3},
            {"a_start": -1, "a_end": -1, "category": "SERVICE#GENERAL", "sentiment": "POSITIVE", "o_start": 8, "o_end": 9},
            {"a_start": 5, "a_end": 6, "category": "AMBIENCE#GENERAL", "sentiment": "NEUTRAL", "o_start": -1, "o_end": -1},
        ]
    }
    res = evaluate_quads(pred, gold)
    assert res["explicit"]["tp"] == 1
    assert res["explicit"]["fp"] == 0
    assert res["explicit"]["fn"] == 0
    assert res["explicit"]["f1"] == 1.0

    assert res["implicit"]["tp"] == 0
    assert res["implicit"]["fp"] == 2
    assert res["implicit"]["fn"] == 1
    assert res["implicit"]["f1"] == 0.0
