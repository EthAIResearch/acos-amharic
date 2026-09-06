import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "common"))
from pair_utils import is_implicit_side, quad_kind


def test_is_implicit_side():
    q_explicit = {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3}
    q_imp_a = {"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3}
    q_imp_o = {"a_start": 0, "a_end": 1, "o_start": -1, "o_end": -1}
    q_fully_imp = {"a_start": -1, "a_end": -1, "o_start": -1, "o_end": -1}

    assert not is_implicit_side(q_explicit, "aspect")
    assert not is_implicit_side(q_explicit, "opinion")

    assert is_implicit_side(q_imp_a, "aspect")
    assert not is_implicit_side(q_imp_a, "opinion")

    assert not is_implicit_side(q_imp_o, "aspect")
    assert is_implicit_side(q_imp_o, "opinion")

    assert is_implicit_side(q_fully_imp, "aspect")
    assert is_implicit_side(q_fully_imp, "opinion")


def test_quad_kind():
    q1 = {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3}
    q2 = {"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3}
    q3 = {"a_start": 0, "a_end": 1, "o_start": -1, "o_end": -1}
    q4 = {"a_start": -1, "a_end": -1, "o_start": -1, "o_end": -1}

    assert quad_kind(q1) == "explicit_both"
    assert quad_kind(q2) == "implicit_aspect_only"
    assert quad_kind(q3) == "implicit_opinion_only"
    assert quad_kind(q4) == "fully_implicit"
