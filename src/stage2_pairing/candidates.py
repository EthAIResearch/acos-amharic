"""
Stage 2: aspect-opinion pairing.

Scope: EXPLICIT spans only. A quad where either the aspect or opinion is
implicit (-1,-1) is excluded here -- that's Stage 5's job (implicit
detection needs to both find AND place implicit terms; keeping that logic
together rather than split across stages).

This module builds candidate (aspect_span, opinion_span) pairs per sentence
plus their gold labels, and a distance-based heuristic pairer -- see
docs/stage2_pairing_analysis.md for why the heuristic is the primary
approach here rather than a learned classifier: >99% of quads are 1:1
aspect<->opinion mappings, real pairing ambiguity occurs in <1% of
sentences, and true pairs sit at a median token distance of 1 vs. 6 for
false candidates. There simply isn't enough negative signal in the data
(704 negative pairs total across 39k training sentences) to train a neural
classifier that would reliably beat this heuristic.
"""
from dataclasses import dataclass


@dataclass
class PairExample:
    aspect_span: tuple[int, int]
    opinion_span: tuple[int, int]
    label: int  # 1 = true pair (co-occur in some quad), 0 = false candidate


def explicit_quads(quads: list[dict]) -> list[dict]:
    return [q for q in quads if q["a_start"] != -1 and q["o_start"] != -1]


def build_candidates(quads: list[dict]) -> list[PairExample]:
    """All (aspect_span, opinion_span) candidate pairs for one sentence,
    with gold labels, restricted to explicit-both quads."""
    ex_quads = explicit_quads(quads)
    if not ex_quads:
        return []

    aspect_spans = sorted(set((q["a_start"], q["a_end"]) for q in ex_quads))
    opinion_spans = sorted(set((q["o_start"], q["o_end"]) for q in ex_quads))
    true_pairs = set((q["a_start"], q["a_end"], q["o_start"], q["o_end"]) for q in ex_quads)

    examples = []
    for a_span in aspect_spans:
        for o_span in opinion_spans:
            label = 1 if (*a_span, *o_span) in true_pairs else 0
            examples.append(PairExample(a_span, o_span, label))
    return examples


def span_distance(a_span: tuple[int, int], o_span: tuple[int, int]) -> int:
    """Minimum token-index distance between the two spans' endpoints."""
    a1, a2 = a_span
    o1, o2 = o_span
    return min(abs(a1 - o1), abs(a1 - o2), abs(a2 - o1), abs(a2 - o2))


def heuristic_pairs(
    aspect_spans: list[tuple[int, int]],
    opinion_spans: list[tuple[int, int]],
) -> set[tuple[int, int, int, int]]:
    """Nearest-token-distance pairing, symmetric union:
    - each aspect paired with its nearest opinion span(s) (ties included)
    - each opinion paired with its nearest aspect span(s) (ties included)
    Union of both directions -- gives a little extra recall on the rare
    one-to-many cases without much precision cost, since both directions
    are still constrained to "nearest", not arbitrary pairing.
    """
    if not aspect_spans or not opinion_spans:
        return set()

    pairs = set()

    for a_span in aspect_spans:
        dists = [(span_distance(a_span, o_span), o_span) for o_span in opinion_spans]
        min_d = min(d for d, _ in dists)
        for d, o_span in dists:
            if d == min_d:
                pairs.add((*a_span, *o_span))

    for o_span in opinion_spans:
        dists = [(span_distance(a_span, o_span), a_span) for a_span in aspect_spans]
        min_d = min(d for d, _ in dists)
        for d, a_span in dists:
            if d == min_d:
                pairs.add((*a_span, *o_span))

    return pairs
