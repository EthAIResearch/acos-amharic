"""
Unit Tests for ByT5 ACOSByteFSM Constrained Decoder
===================================================
Tests prefix trie construction, slot state transitions, delimiter enforcement,
and category/sentiment constraint logic with a mock byte tokenizer.
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_byt5_acos"))
from constrained_decoder import ACOSByteFSM, ByteTrie
from linearization import CATEGORIES, SENTIMENTS


class MockByT5Tokenizer:
    """Mock tokenizer simulating ByT5 raw UTF-8 byte encoding."""
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # ByT5 maps byte b -> b + 3
        raw_bytes = text.encode("utf-8")
        return [b + 3 for b in raw_bytes]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        clean = []
        for tid in token_ids:
            if skip_special_tokens and tid in (self.pad_token_id, self.eos_token_id, self.unk_token_id):
                continue
            if tid >= 3 and tid <= 258:
                clean.append(tid - 3)
        return bytes(clean).decode("utf-8", errors="replace")


def test_fsm_initial_state():
    tok = MockByT5Tokenizer()
    fsm = ACOSByteFSM(tok)

    # Initial state on empty sequence
    allowed = fsm._get_allowed_tokens_for_sequence([])
    open_bracket_id = tok.encode("[")[0]
    none_first_id = tok.encode("N")[0]

    assert open_bracket_id in allowed
    assert none_first_id in allowed
    # Random byte should NOT be allowed at sequence start
    z_id = tok.encode("Z")[0]
    assert z_id not in allowed


def test_fsm_none_sequence():
    tok = MockByT5Tokenizer()
    fsm = ACOSByteFSM(tok)

    # Sequence is typing "None"
    prefix = tok.encode("Non")
    allowed = fsm._get_allowed_tokens_for_sequence(prefix)
    e_id = tok.encode("e")[0]
    assert allowed == {e_id}

    # Once "None" is complete, ONLY EOS allowed
    prefix_complete = tok.encode("None")
    allowed_end = fsm._get_allowed_tokens_for_sequence(prefix_complete)
    assert allowed_end == {tok.eos_token_id}


def test_fsm_category_slot_constraints():
    tok = MockByT5Tokenizer()
    fsm = ACOSByteFSM(tok)

    # Inside quad at Category slot: "[ aspect | "
    prefix = tok.encode("[ ምግብ | ")
    allowed = fsm._get_allowed_tokens_for_sequence(prefix)

    # Allowed tokens must be the first characters of the 22 categories
    first_cat_chars = {tok.encode(cat[0])[0] for cat in CATEGORIES}
    assert allowed == first_cat_chars

    # Partial category: "[ ምግብ | PUBLIC_SAFETY#"
    prefix_partial = tok.encode("[ ምግብ | PUBLIC_SAFETY#")
    allowed_partial = fsm._get_allowed_tokens_for_sequence(prefix_partial)
    # The valid continuations in INSA categories are C (CRIME, CRIME_SERVICES) and E (EMERGENCY_SERVICES)
    c_id = tok.encode("C")[0]
    e_id = tok.encode("E")[0]
    assert c_id in allowed_partial
    assert e_id in allowed_partial
    # Disallowed char like 'Z'
    z_id = tok.encode("Z")[0]
    assert z_id not in allowed_partial


def test_fsm_sentiment_slot_constraints():
    tok = MockByT5Tokenizer()
    fsm = ACOSByteFSM(tok)

    # Inside quad at Sentiment slot: "[ ምግብ | PUBLIC_SAFETY#CRIME | "
    prefix = tok.encode("[ ምግብ | PUBLIC_SAFETY#CRIME | ")
    allowed = fsm._get_allowed_tokens_for_sequence(prefix)

    # Sentiment must start with P (POSITIVE) or N (NEGATIVE, NEUTRAL)
    p_id = tok.encode("P")[0]
    n_id = tok.encode("N")[0]
    assert allowed == {p_id, n_id}


def test_fsm_quad_end_and_continuation():
    tok = MockByT5Tokenizer()
    fsm = ACOSByteFSM(tok)

    # Quad completed: "[ ምግብ | PUBLIC_SAFETY#CRIME | POSITIVE | ጥሩ ]"
    prefix = tok.encode("[ ምግብ | PUBLIC_SAFETY#CRIME | POSITIVE | ጥሩ ]")
    allowed = fsm._get_allowed_tokens_for_sequence(prefix)

    # Can finish with EOS or continue with " ; "
    semi_id = tok.encode(" ; ")[0]
    assert tok.eos_token_id in allowed
    assert semi_id in allowed


if __name__ == "__main__":
    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} unit tests in test_constrained_decoder...")
    for fn in test_funcs:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print("All constrained decoder tests passed successfully!")
