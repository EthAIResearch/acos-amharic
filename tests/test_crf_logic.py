"""
Unit tests for Linear-Chain CRF decoding and transition logic.
Runs dependency-free in CI environments without requiring torch.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_aope_sdrn"))
from crf import viterbi_decode_reference


def test_viterbi_decode_reference_basic():
    # 3 tags: 0=O, 1=B, 2=I
    # Strong emissions for B at pos 0, I at pos 1, O at pos 2
    emissions = [
        [0.1, 5.0, 0.2],  # pos 0: B
        [0.2, 0.5, 6.0],  # pos 1: I
        [4.0, 0.3, 0.1],  # pos 2: O
    ]
    # Standard neutral transitions
    transitions = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    start_transitions = [0.0, 0.0, 0.0]
    end_transitions = [0.0, 0.0, 0.0]

    path = viterbi_decode_reference(emissions, transitions, start_transitions, end_transitions)
    assert path == [1, 2, 0]  # B -> I -> O


def test_viterbi_bio_constraints_rejects_start_to_i():
    # Emissions strongly favor I (tag 2) at position 0
    emissions = [
        [1.0, 2.0, 10.0],  # pos 0 favors I heavily
        [1.0, 1.0, 1.0],
    ]
    transitions = [
        [0.0, 0.0, -10000.0],  # O cannot go to I
        [0.0, 0.0, 0.0],        # B can go to I
        [0.0, 0.0, 0.0],        # I can go to I
    ]
    # START cannot go to I
    start_transitions = [0.0, 0.0, -10000.0]
    end_transitions = [0.0, 0.0, 0.0]

    path = viterbi_decode_reference(emissions, transitions, start_transitions, end_transitions)
    # Pos 0 cannot be I (tag 2); must be B (tag 1)
    assert path[0] == 1


def test_viterbi_bio_constraints_rejects_o_to_i():
    # Pos 0 is forced to O
    # Pos 1 strongly favors I
    emissions = [
        [10.0, 0.0, 0.0],  # pos 0: O
        [0.0, 1.0, 8.0],   # pos 1: favors I (8.0) over B (1.0)
    ]
    transitions = [
        [0.0, 0.0, -10000.0],  # O -> I is forbidden
        [0.0, 0.0, 0.0],        # B -> I is allowed
        [0.0, 0.0, 0.0],
    ]
    start_transitions = [0.0, 0.0, -10000.0]
    end_transitions = [0.0, 0.0, 0.0]

    path = viterbi_decode_reference(emissions, transitions, start_transitions, end_transitions)
    # Pos 0 is O (0). Pos 1 cannot be I (2) because O -> I penalty (-10000) outweighs emission advantage (8 vs 1).
    assert path[0] == 0
    assert path[1] != 2  # Must NOT be I


def test_viterbi_empty_and_single_token():
    assert viterbi_decode_reference([], [], [], []) == []

    # Single token
    emissions = [[1.0, 5.0, 0.0]]
    transitions = [[0.0] * 3 for _ in range(3)]
    start_transitions = [0.0, 0.0, -10000.0]
    end_transitions = [0.0, 0.0, 0.0]
    path = viterbi_decode_reference(emissions, transitions, start_transitions, end_transitions)
    assert path == [1]


def test_config_stage_aope_sdrn_crf_enabled():
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "stage_aope_sdrn.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("use_crf") is True
    assert cfg.get("output_dir") == "results/stage_aope_sdrn/afroxlmr_crf_run1"


if __name__ == "__main__":
    import inspect

    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} unit tests in test_crf_logic...")
    for fn in test_funcs:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print("All CRF logic tests passed successfully!")
