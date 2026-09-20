import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_aope_sdrn"))
from relation_utils import (
    build_word_relation_matrix,
    compute_prf,
    correlation_degree,
    explicit_pairs,
    sweep_thresholds,
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


def test_compute_prf_simple():
    preds = [[((0, 1), (2, 3)), ((4, 5), (6, 7))]]
    golds = [[((0, 1), (2, 3)), ((8, 9), (10, 11))]]
    metrics = compute_prf(preds, golds)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_sweep_thresholds():
    # 1 example with 2 candidates:
    # candidate 1: score 0.45, gold
    # candidate 2: score 0.15, not gold
    candidates = [[
        (((0, 1), (2, 3)), 0.45),
        (((4, 5), (6, 7)), 0.15),
    ]]
    golds = [[((0, 1), (2, 3))]]

    sweep = sweep_thresholds(candidates, golds, thresholds=[0.1, 0.3, 0.5])
    assert sweep["candidate_ceiling_recall"] == 1.0

    # At threshold 0.1: both accepted -> TP=1, FP=1 -> P=0.5, R=1.0, F1=0.667
    row_01 = next(r for r in sweep["sweep"] if r["threshold"] == 0.1)
    assert row_01["tp"] == 1
    assert row_01["fp"] == 1
    assert row_01["recall"] == 1.0

    # At threshold 0.3: only candidate 1 accepted -> TP=1, FP=0 -> P=1.0, R=1.0, F1=1.0
    row_03 = next(r for r in sweep["sweep"] if r["threshold"] == 0.3)
    assert row_03["tp"] == 1
    assert row_03["fp"] == 0
    assert row_03["f1"] == 1.0

    # At threshold 0.5: none accepted -> TP=0, FP=0, FN=1 -> F1=0.0
    row_05 = next(r for r in sweep["sweep"] if r["threshold"] == 0.5)
    assert row_05["tp"] == 0
    assert row_05["fn"] == 1
    assert row_05["recall"] == 0.0

    # Best threshold should be 0.3
    assert sweep["best_threshold"] == 0.3
    assert sweep["best_metrics"]["f1"] == 1.0


def test_stage_aope_sdrn_config_explicit_and_weights():
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "stage_aope_sdrn.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("model_name") == "Davlan/afro-xlmr-base"
    assert cfg.get("data", {}).get("train") == "data/prepared_explicit/train.jsonl"
    assert cfg.get("data", {}).get("dev") == "data/prepared_explicit/dev.jsonl"
    assert cfg.get("data", {}).get("test") == "data/prepared_explicit/test.jsonl"
    assert cfg.get("training", {}).get("bio_class_weights") == [1.0, 3.0, 3.0]
    assert cfg.get("training", {}).get("span_loss_weight") == 2.0
    assert cfg.get("use_crf") is True


if __name__ == "__main__":
    import inspect

    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} unit tests in test_stage_aope_sdrn...")
    for fn in test_funcs:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print("All tests passed successfully!")


