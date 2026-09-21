"""
Unit tests for Linear-Chain CRF decoding and transition logic.
Runs dependency-free in CI environments without requiring torch.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_aope_sdrn"))
from crf import (
    validate_5way_bio_path_reference,
    validate_bio_path_reference,
    viterbi_decode_reference,
)


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


def test_validate_bio_path_reference():
    # Valid paths
    assert validate_bio_path_reference([]) is True
    assert validate_bio_path_reference([0]) is True
    assert validate_bio_path_reference([1]) is True
    assert validate_bio_path_reference([0, 1, 2, 0]) is True
    assert validate_bio_path_reference([1, 2, 2, 0]) is True
    assert validate_bio_path_reference([0, 0, 1, 2, 0, 1, 0]) is True

    # Invalid: starts with I (2)
    assert validate_bio_path_reference([2]) is False
    assert validate_bio_path_reference([2, 0, 1]) is False
    assert validate_bio_path_reference([2, 2]) is False

    # Invalid: O -> I (0 -> 2)
    assert validate_bio_path_reference([0, 2]) is False
    assert validate_bio_path_reference([1, 0, 2]) is False
    assert validate_bio_path_reference([0, 1, 2, 0, 2]) is False


def test_validate_5way_bio_path_reference():
    # Valid: 0: O, 1: B-ASP, 2: I-ASP, 3: B-OPN, 4: I-OPN
    assert validate_5way_bio_path_reference([]) is True
    assert validate_5way_bio_path_reference([0, 1, 2, 0, 3, 4, 0]) is True
    assert validate_5way_bio_path_reference([1, 3]) is True  # adjacent aspect then opinion
    assert validate_5way_bio_path_reference([1, 2, 3, 4]) is True

    # Invalid starts (I-ASP or I-OPN)
    assert validate_5way_bio_path_reference([2]) is False
    assert validate_5way_bio_path_reference([4]) is False
    assert validate_5way_bio_path_reference([2, 1, 3]) is False
    assert validate_5way_bio_path_reference([4, 1, 3]) is False

    # Invalid transitions
    assert validate_5way_bio_path_reference([0, 2]) is False  # O -> I-ASP
    assert validate_5way_bio_path_reference([0, 4]) is False  # O -> I-OPN
    assert validate_5way_bio_path_reference([1, 4]) is False  # B-ASP -> I-OPN
    assert validate_5way_bio_path_reference([2, 4]) is False  # I-ASP -> I-OPN
    assert validate_5way_bio_path_reference([3, 2]) is False  # B-OPN -> I-ASP
    assert validate_5way_bio_path_reference([4, 2]) is False  # I-OPN -> I-ASP


def test_viterbi_bio_constraints_with_inf():
    # Verify that -inf works identically to large negative penalty
    emissions = [
        [10.0, 0.0, 0.0],  # pos 0: O
        [0.0, 1.0, 8.0],   # pos 1: favors I (8.0) over B (1.0)
    ]
    transitions = [
        [0.0, 0.0, -float("inf")],  # O -> I is forbidden with -inf
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    start_transitions = [0.0, 0.0, -float("inf")]
    end_transitions = [0.0, 0.0, 0.0]

    path = viterbi_decode_reference(emissions, transitions, start_transitions, end_transitions)
    assert path[0] == 0
    assert path[1] != 2  # Must not be I
    assert path[1] == 1  # Must fall back to B


def test_linear_chain_crf_torch_if_available():
    try:
        import torch
        from crf import LinearChainCRF
    except ImportError:
        return

    # 3-tag CRF check
    crf = LinearChainCRF(num_tags=3, enforce_bio_constraints=True)
    assert crf.transitions[0, 2].item() != -float("inf")
    constrained_t, constrained_s = crf._get_constrained_transitions()
    assert constrained_t[0, 2].item() == -float("inf")
    assert constrained_s[2].item() == -float("inf")

    # 5-tag CRF check
    crf5 = LinearChainCRF(num_tags=5, enforce_bio_constraints=True)
    t5, s5 = crf5._get_constrained_transitions()
    # START -> 2 (I-ASP) and START -> 4 (I-OPN) forbidden
    assert s5[2].item() == -float("inf")
    assert s5[4].item() == -float("inf")
    # Transitions forbidden
    assert t5[0, 2].item() == -float("inf")
    assert t5[0, 4].item() == -float("inf")
    assert t5[1, 4].item() == -float("inf")
    assert t5[2, 4].item() == -float("inf")
    assert t5[3, 2].item() == -float("inf")
    assert t5[4, 2].item() == -float("inf")
    # Transitions allowed
    assert t5[1, 2].item() != -float("inf")
    assert t5[3, 4].item() != -float("inf")
    assert t5[1, 3].item() != -float("inf")
    assert t5[2, 3].item() != -float("inf")

    # Validation rejects forbidden start
    emissions5 = torch.randn(1, 3, 5)
    bad_tags = torch.tensor([[4, 0, 1]])
    try:
        crf5(emissions5, bad_tags)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "forbidden tag 4" in str(e)

    # Validation rejects forbidden transition (B-ASP -> I-OPN: 1 -> 4)
    bad_trans = torch.tensor([[1, 4, 0]])
    try:
        crf5(emissions5, bad_trans)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "forbidden transition 1 -> 4" in str(e)

    # Valid path passes
    good_tags5 = torch.tensor([[1, 2, 0]])
    loss = crf5(emissions5, good_tags5)
    assert not torch.isnan(loss) and not torch.isinf(loss)



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
