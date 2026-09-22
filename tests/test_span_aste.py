"""
Unit Tests for Span-ASTE (ACL 2021)
===================================
Tests span utilities, bucketing, gold target building, evaluation metrics,
and model forward pass with mock encoder.
"""
import inspect
import os
import sys

# Add src directories to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_span_aste"))
from evaluate import compute_prf, evaluate_batch_predictions, summarize_metrics
from span_utils import (
    MENTION2ID,
    RELATION2ID,
    bucket_value,
    build_gold_pair_labels,
    build_gold_span_labels,
    compute_span_distance,
    enumerate_spans,
    extract_explicit_triplets,
)


def test_bucket_value_boundaries():
    assert bucket_value(0) == 0
    assert bucket_value(1) == 1
    assert bucket_value(2) == 2
    assert bucket_value(3) == 3
    assert bucket_value(4) == 4
    assert bucket_value(5) == 5
    assert bucket_value(7) == 5
    assert bucket_value(8) == 6
    assert bucket_value(15) == 6
    assert bucket_value(16) == 7
    assert bucket_value(31) == 7
    assert bucket_value(32) == 8
    assert bucket_value(63) == 8
    assert bucket_value(64) == 9
    assert bucket_value(100) == 9


def test_enumerate_spans():
    # Sentence length 4, max_len 2 -> spans of length 1 and 2
    # (0,1), (0,2), (1,2), (1,3), (2,3), (2,4), (3,4) = 7 spans
    spans = enumerate_spans(4, max_span_length=2)
    expected = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    assert spans == expected


def test_compute_span_distance():
    # Disjoint: [0, 2) and [4, 6) -> distance = 4 - 2 = 2
    assert compute_span_distance((0, 2), (4, 6)) == 2
    # Disjoint: [5, 7) and [1, 3) -> distance = 5 - 3 = 2
    assert compute_span_distance((5, 7), (1, 3)) == 2
    # Adjacent: [0, 2) and [2, 4) -> distance = 0
    assert compute_span_distance((0, 2), (2, 4)) == 0
    # Overlapping: [1, 4) and [2, 5) -> distance = 0
    assert compute_span_distance((1, 4), (2, 5)) == 0


def test_build_gold_span_labels():
    spans = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    quads = [
        {"a_start": 0, "a_end": 2, "o_start": 3, "o_end": 4, "sentiment": "POSITIVE"},
    ]
    labels = build_gold_span_labels(spans, quads)
    # (0,1): INVALID (0), (0,2): TARGET (1), (1,2): INVALID (0), (2,3): INVALID (0), (3,4): OPINION (2)
    assert labels == [0, 1, 0, 0, 2]


def test_build_gold_pair_labels():
    target_spans = [(0, 2), (5, 6)]
    opinion_spans = [(3, 4), (8, 9)]
    quads = [
        {"a_start": 0, "a_end": 2, "o_start": 3, "o_end": 4, "sentiment": "POSITIVE"},
        {"a_start": 5, "a_end": 6, "o_start": 8, "o_end": 9, "sentiment": "NEGATIVE"},
    ]
    matrix = build_gold_pair_labels(target_spans, opinion_spans, quads)
    # row 0: [(0,2),(3,4)] -> POSITIVE (1), [(0,2),(8,9)] -> INVALID (0)
    # row 1: [(5,6),(3,4)] -> INVALID (0), [(5,6),(8,9)] -> NEGATIVE (2)
    assert matrix == [[1, 0], [0, 2]]


def test_extract_explicit_triplets():
    quads = [
        {"a_start": 1, "a_end": 3, "o_start": 4, "o_end": 5, "sentiment": "POSITIVE"},
        {"a_start": -1, "a_end": -1, "o_start": 6, "o_end": 7, "sentiment": "NEGATIVE"},  # Implicit aspect
    ]
    triplets = extract_explicit_triplets(quads)
    assert triplets == [((1, 3), (4, 5), "POSITIVE")]


def test_compute_prf_perfect():
    res = compute_prf(10, 0, 0)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0


def test_compute_prf_zero():
    res = compute_prf(0, 5, 5)
    assert res["precision"] == 0.0
    assert res["recall"] == 0.0
    assert res["f1"] == 0.0


def test_evaluate_batch_predictions():
    batch_outputs = [
        {
            "target_candidates": [(0, 2)],
            "opinion_candidates": [(3, 4)],
            "pred_triplets": [((0, 2), (3, 4), 1)],  # 1: POSITIVE
        }
    ]
    raw_quads = [
        [{"a_start": 0, "a_end": 2, "o_start": 3, "o_end": 4, "sentiment": "POSITIVE"}]
    ]
    counts = evaluate_batch_predictions(batch_outputs, raw_quads)
    metrics = summarize_metrics(counts)

    assert metrics["aste"]["f1"] == 1.0
    assert metrics["aope"]["f1"] == 1.0
    assert metrics["ate"]["f1"] == 1.0
    assert metrics["ote"]["f1"] == 1.0
    assert metrics["candidate_ceiling_recall"] == 1.0
    assert metrics["candidate_conversion_efficiency"] == 1.0


def test_span_aste_model_mock_forward():
    try:
        import torch
        from model import MLP, SpanASTEModel
        from torch import nn
    except ImportError:
        return

    class MockEncoder(nn.Module):
        def __init__(self, hidden_size: int = 64):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden_size})()

        def forward(self, input_ids, attention_mask=None):
            B, L = input_ids.shape
            d = self.config.hidden_size
            dummy = torch.randn(B, L, d, device=input_ids.device)
            return type("Output", (), {"last_hidden_state": dummy})()

    model = SpanASTEModel(
        model_name="Davlan/afro-xlmr-base",
        max_span_length=4,
        pruning_ratio=0.5,
        width_dim=8,
        distance_dim=16,
        hidden_dim=32,
        dropout=0.0,
    )
    # Inject mock encoder to avoid downloading transformer weights in unit tests
    model.encoder = MockEncoder(hidden_size=64)
    # Re-initialize span classifiers matching mock dimension: 2*64 + 8 = 136
    span_dim = 2 * 64 + 8
    model.mention_classifier = MLP(span_dim, hidden_dim=32, out_dim=3)
    model.relation_classifier = MLP(2 * span_dim + 16, hidden_dim=32, out_dim=4)

    B, L, M = 2, 10, 6
    input_ids = torch.randint(0, 100, (B, L))
    attention_mask = torch.ones((B, L), dtype=torch.long)

    subword_to_word = torch.zeros((B, M, L), dtype=torch.float32)
    for b in range(B):
        for m in range(M):
            subword_to_word[b, m, min(m, L - 1)] = 1.0

    seq_spans = [
        enumerate_spans(M, max_span_length=4),
        enumerate_spans(M, max_span_length=4),
    ]
    gold_mention_labels = [
        torch.zeros(len(seq_spans[0]), dtype=torch.long),
        torch.zeros(len(seq_spans[1]), dtype=torch.long),
    ]
    gold_pairs = [
        [((0, 2), (3, 4), 1)],
        [],
    ]

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        subword_to_word=subword_to_word,
        seq_spans=seq_spans,
        gold_mention_labels=gold_mention_labels,
        gold_pairs=gold_pairs,
        num_words=[M, M],
    )

    assert "loss" in out
    assert "mention_loss" in out
    assert "relation_loss" in out
    assert out["loss"].item() > 0.0
    assert len(out["batch_outputs"]) == B


if __name__ == "__main__":
    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} unit tests in test_span_aste...")
    for fn in test_funcs:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print("All Span-ASTE tests passed successfully!")
