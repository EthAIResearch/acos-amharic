import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_aope_sdrn"))
from relation_utils import (
    build_word_relation_matrix,
    correlation_degree,
    explicit_pairs,
)


def test_build_word_relation_matrix_empty():
    matrix = build_word_relation_matrix(4, [])
    assert len(matrix) == 4
    assert all(row == [0, 0, 0, 0] for row in matrix)


def test_build_word_relation_matrix_single_pair():
    # Aspect at (0, 1), Opinion at (2, 3)
    pairs = [((0, 1), (2, 3))]
    matrix = build_word_relation_matrix(4, pairs)
    assert matrix[0][2] == 1
    assert matrix[2][0] == 1
    assert matrix[0][0] == 0
    assert matrix[0][1] == 0
    assert matrix[1][2] == 0


def test_build_word_relation_matrix_multi_token_spans():
    # Aspect span (0, 2) [words 0, 1], Opinion span (3, 5) [words 3, 4]
    pairs = [((0, 2), (3, 5))]
    matrix = build_word_relation_matrix(6, pairs)
    for a in (0, 1):
        for o in (3, 4):
            assert matrix[a][o] == 1
            assert matrix[o][a] == 1
    # Check unaffected tokens
    assert matrix[2][3] == 0
    assert matrix[5][0] == 0


def test_correlation_degree_perfect_and_zero():
    pairs = [((1, 2), (3, 4))]
    matrix = build_word_relation_matrix(5, pairs)

    # Exact match
    deg = correlation_degree(matrix, (1, 2), (3, 4))
    assert deg == 1.0

    # Unrelated spans
    deg_unrelated = correlation_degree(matrix, (0, 1), (3, 4))
    assert deg_unrelated == 0.0

    # Empty span
    assert correlation_degree(matrix, (1, 1), (3, 4)) == 0.0


def test_correlation_degree_partial_match():
    # Multi-token spans with partial relation
    # Suppose aspect is (0, 2) [tokens 0, 1], opinion is (2, 4) [tokens 2, 3]
    # Matrix only links (0, 2)
    matrix = [[0] * 5 for _ in range(5)]
    matrix[0][2] = 1
    matrix[2][0] = 1

    deg = correlation_degree(matrix, (0, 2), (2, 4))
    # Aspect len 2, Opinion len 2 -> 4 pairs total, only 1 linked -> 1/4 = 0.25
    assert abs(deg - 0.25) < 1e-6


def test_explicit_pairs_filters_implicit():
    quads = [
        {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3},
        {"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3},
        {"a_start": 0, "a_end": 1, "o_start": -1, "o_end": -1},
        {"a_start": -1, "a_end": -1, "o_start": -1, "o_end": -1},
        {"a_start": 4, "a_end": 6, "o_start": 7, "o_end": 8},
    ]
    pairs = explicit_pairs(quads)
    assert len(pairs) == 2
    assert pairs[0] == ((0, 1), (2, 3))
    assert pairs[1] == ((4, 6), (7, 8))


def test_subword_relation_projection():
    # Verify the projection rule used in JointAOPEDataset:
    # subwords inherit word-level relations
    word_rel = [[0, 1], [1, 0]]
    word_ids = [None, 0, 0, 1, None]  # CLS, word0_sub1, word0_sub2, word1, SEP
    L = 5
    sub_rel = [[0] * L for _ in range(L)]
    n = 2
    for si, wi in enumerate(word_ids):
        if wi is None:
            continue
        for sj, wj in enumerate(word_ids):
            if wj is None:
                continue
            if wi < n and wj < n:
                sub_rel[si][sj] = word_rel[wi][wj]

    # Both subwords of word 0 (indices 1, 2) should be linked to word 1 (index 3)
    assert sub_rel[1][3] == 1
    assert sub_rel[2][3] == 1
    assert sub_rel[3][1] == 1
    assert sub_rel[3][2] == 1
    # Subwords of word 0 should not be linked to each other
    assert sub_rel[1][2] == 0
    # Special tokens should not be linked
    assert sub_rel[0][3] == 0
    assert sub_rel[4][3] == 0
