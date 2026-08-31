import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage2_pairing"))
from candidates import explicit_quads, build_candidates, span_distance, heuristic_pairs


def test_explicit_quads_filters_implicit():
    quads = [
        {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3, "category": "X", "sentiment": "POSITIVE"},
        {"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3, "category": "X", "sentiment": "POSITIVE"},
    ]
    result = explicit_quads(quads)
    assert len(result) == 1
    assert result[0]["a_start"] == 0


def test_span_distance():
    assert span_distance((0, 1), (2, 3)) == 1  # a_end=1 to o_start=2
    assert span_distance((5, 6), (5, 6)) == 0  # identical span


def test_heuristic_pairs_simple_case():
    # one aspect, one opinion -> trivially paired
    aspect_spans = [(0, 1)]
    opinion_spans = [(3, 4)]
    pairs = heuristic_pairs(aspect_spans, opinion_spans)
    assert pairs == {(0, 1, 3, 4)}


def test_heuristic_pairs_picks_nearest():
    # realistic ambiguous shape: 2 aspects, 2 opinions, each aspect closer to
    # one specific opinion than the other (mirrors the (2,2) shape that makes
    # up the vast majority of real ambiguous sentences in the dataset)
    aspect_spans = [(0, 1), (20, 21)]
    opinion_spans = [(2, 3), (19, 20)]
    pairs = heuristic_pairs(aspect_spans, opinion_spans)
    assert (0, 1, 2, 3) in pairs        # aspect@0 nearest to opinion@2
    assert (20, 21, 19, 20) in pairs    # aspect@20 nearest to opinion@19
    assert (0, 1, 19, 20) not in pairs  # far cross-pair should not appear
    assert (20, 21, 2, 3) not in pairs


def test_build_candidates_label_correctness():
    quads = [
        {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3, "category": "X", "sentiment": "POSITIVE"},
    ]
    examples = build_candidates(quads)
    assert len(examples) == 1
    assert examples[0].label == 1
