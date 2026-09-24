"""
Unit Tests for ByT5 Amharic ACOS Quadruple Extraction
=====================================================
Tests serialization, robust parsing, dataset collation with -100 label padding,
and metric calculation without requiring external model weights.
"""
import inspect
import os
import sys

# Add src directories to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_byt5_acos"))
from linearization import (
    CATEGORIES,
    SENTIMENTS,
    calculate_metrics,
    compute_set_prf,
    extract_gold_quad_tuples,
    parse_target_to_quads,
    quads_to_target,
)


def test_quads_to_target_explicit_and_implicit():
    tokens = ["ምግቡ", "በጣም", "ጥሩ", "ነው", "ግን", "አገልግሎቱ", "ደካማ"]
    quads = [
        # Explicit quad
        {"a_start": 0, "a_end": 1, "category": "PUBLIC_SERVICES#HEALTHCARE", "sentiment": "POSITIVE", "o_start": 1, "o_end": 3},
        # Implicit opinion quad
        {"a_start": 5, "a_end": 6, "category": "PUBLIC_SERVICES#COMMUNITY_SUPPORT", "sentiment": "NEGATIVE", "o_start": -1, "o_end": -1},
        # Fully implicit quad
        {"a_start": -1, "a_end": -1, "category": "GOVERNANCE#TRANSPARENCY", "sentiment": "NEGATIVE", "o_start": -1, "o_end": -1},
    ]

    target_str = quads_to_target(quads, tokens)
    expected_blocks = [
        "[ ምግቡ | PUBLIC_SERVICES#HEALTHCARE | POSITIVE | በጣም ጥሩ ]",
        "[ አገልግሎቱ | PUBLIC_SERVICES#COMMUNITY_SUPPORT | NEGATIVE | NULL ]",
        "[ NULL | GOVERNANCE#TRANSPARENCY | NEGATIVE | NULL ]",
    ]
    expected_str = " ; ".join(expected_blocks)
    assert target_str == expected_str

    # Test empty quads
    assert quads_to_target([], tokens) == "None"


def test_parse_target_to_quads_roundtrip():
    target_str = (
        "[ ምግቡ | PUBLIC_SERVICES#HEALTHCARE | POSITIVE | በጣም ጥሩ ] ; "
        "[ አገልግሎቱ | PUBLIC_SERVICES#COMMUNITY_SUPPORT | NEGATIVE | NULL ] ; "
        "[ NULL | GOVERNANCE#TRANSPARENCY | NEGATIVE | NULL ]"
    )

    parsed = parse_target_to_quads(target_str)
    assert len(parsed) == 3
    assert parsed[0] == ("ምግቡ", "PUBLIC_SERVICES#HEALTHCARE", "POSITIVE", "በጣም ጥሩ")
    assert parsed[1] == ("አገልግሎቱ", "PUBLIC_SERVICES#COMMUNITY_SUPPORT", "NEGATIVE", "NULL")
    assert parsed[2] == ("NULL", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "NULL")


def test_parse_target_to_quads_none_and_malformed():
    # Empty / none targets
    assert parse_target_to_quads("None") == []
    assert parse_target_to_quads("none.") == []
    assert parse_target_to_quads("") == []

    # Extra whitespace and lowercase sentiment normalization
    raw_str = "[  ምግቡ   |  PUBLIC_SERVICES#HEALTHCARE  |  positive  |  በጣም ጥሩ  ]"
    parsed = parse_target_to_quads(raw_str)
    assert len(parsed) == 1
    assert parsed[0] == ("ምግቡ", "PUBLIC_SERVICES#HEALTHCARE", "POSITIVE", "በጣም ጥሩ")

    # Invalid category rejected in strict mode
    bad_cat = "[ ምግቡ | NONEXISTENT_CATEGORY | POSITIVE | ጥሩ ]"
    assert parse_target_to_quads(bad_cat, strict_categories=True) == []

    # Incomplete pipe block rejected safely
    bad_block = "[ ምግቡ | PUBLIC_SERVICES#HEALTHCARE | POSITIVE ]"
    assert parse_target_to_quads(bad_block) == []


def test_extract_gold_quad_tuples():
    sample = {
        "text": "አገልግሎቱ ጥሩ ነው",
        "tokens": ["አገልግሎቱ", "ጥሩ", "ነው"],
        "quads": [
            {"a_start": 0, "a_end": 1, "category": "PUBLIC_SERVICES#UTILITIES", "sentiment": "POSITIVE", "o_start": 1, "o_end": 2},
            {"a_start": -1, "a_end": -1, "category": "GOVERNANCE#TRANSPARENCY", "sentiment": "NEGATIVE", "o_start": -1, "o_end": -1},
        ],
    }

    tuples = extract_gold_quad_tuples(sample)
    assert len(tuples) == 2
    assert tuples[0] == ("አገልግሎቱ", "PUBLIC_SERVICES#UTILITIES", "POSITIVE", "ጥሩ")
    assert tuples[1] == ("NULL", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "NULL")


def test_compute_set_prf_and_metrics():
    gold = [
        ("ምግቡ", "PUBLIC_SERVICES#HEALTHCARE", "POSITIVE", "ጥሩ"),
        ("አገልግሎቱ", "PUBLIC_SERVICES#UTILITIES", "NEGATIVE", "ደካማ"),
    ]
    pred = [
        ("ምግቡ", "PUBLIC_SERVICES#HEALTHCARE", "POSITIVE", "ጥሩ"),      # TP
        ("አገልግሎቱ", "PUBLIC_SERVICES#UTILITIES", "POSITIVE", "ደካማ"),    # FP (sentiment mismatch)
        ("ሌላ", "GOVERNANCE#TRANSPARENCY", "NEGATIVE", "መጥፎ"),           # FP
    ]

    tp, fp, fn = compute_set_prf(pred, gold)
    assert tp == 1
    assert fp == 2
    assert fn == 1

    m = calculate_metrics(tp, fp, fn)
    assert round(m["precision"], 4) == round(1 / 3, 4)
    assert round(m["recall"], 4) == 0.50
    assert round(m["f1"], 4) == round(2 * (1/3) * 0.5 / (1/3 + 0.5), 4)


def test_byt5_collate_fn():
    try:
        import torch
        from dataset import collate_fn
    except ImportError:
        return

    batch = [
        {
            "input_ids": [10, 20, 30],
            "attention_mask": [1, 1, 1],
            "labels": [100, 101],
            "raw_text": "አንድ",
            "raw_target": "[ a | c | s | o ]",
            "gold_quads": [("a", "c", "s", "o")],
        },
        {
            "input_ids": [40, 50],
            "attention_mask": [1, 1],
            "labels": [102, 103, 104, 105],
            "raw_text": "ሁለት",
            "raw_target": "[ a2 | c2 | s2 | o2 ]",
            "gold_quads": [("a2", "c2", "s2", "o2")],
        },
    ]

    collated = collate_fn(batch, pad_token_id=0)

    # Check source padding
    assert collated["input_ids"].shape == (2, 3)
    assert collated["input_ids"][1, 2].item() == 0  # Padded token
    assert collated["attention_mask"][1, 2].item() == 0

    # Check label padding with -100
    assert collated["labels"].shape == (2, 4)
    assert collated["labels"][0, 2].item() == -100
    assert collated["labels"][0, 3].item() == -100
    assert collated["labels"][1, 3].item() == 105


if __name__ == "__main__":
    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} unit tests in test_byt5_acos...")
    for fn in test_funcs:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print("All ByT5 ACOS tests passed successfully!")
