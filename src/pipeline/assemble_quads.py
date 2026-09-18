"""
Stage 6a: combine Stage 1-5 outputs into full ACOS quadruples for one
sentence. Pure orchestration logic -- no model inference here (that's
run_inference.py); this module just defines how predictions from each
stage combine into the final quad list, and is fully unit-testable without
torch/transformers.

Quad schema (matches data_prep.py's Quad dataclass):
    {"a_start": int, "a_end": int, "category": str, "sentiment": str,
     "o_start": int, "o_end": int}
    -1 for a_start/o_start means implicit (no span), matching the raw
    dataset convention throughout this project.
"""
from dataclasses import dataclass, field

IMPLICIT = (-1, -1)


@dataclass
class SentencePredictions:
    """All Stage 1-5 model outputs for one sentence, already computed
    elsewhere (run_inference.py). Kept as one dataclass so assemble() has a
    single, clearly-typed input rather than a long positional argument list."""
    aspect_spans: list          # Stage 1: list of (start, end)
    opinion_spans: list         # Stage 1: list of (start, end)
    explicit_pairs: set         # Stage 2: set of (a_start, a_end, o_start, o_end)
    pair_category: dict         # Stage 3: (a_start,a_end,o_start,o_end) -> category
    pair_sentiment: dict        # Stage 4: (a_start,a_end,o_start,o_end) -> sentiment
    implicit_aspect_flag: dict = field(default_factory=dict)      # Stage 5a: opinion_span -> bool
    implicit_aspect_category: dict = field(default_factory=dict)  # Stage 5: opinion_span -> category
    implicit_aspect_sentiment: dict = field(default_factory=dict)  # Stage 5: opinion_span -> sentiment
    implicit_opinion_flag: dict = field(default_factory=dict)     # Stage 5b: aspect_span -> bool
    implicit_opinion_category: dict = field(default_factory=dict)  # Stage 5: aspect_span -> category
    implicit_opinion_sentiment: dict = field(default_factory=dict)  # Stage 5: aspect_span -> sentiment
    fully_implicit_flag: bool = False                              # Stage 5c
    fully_implicit_category: str = None                            # Stage 5
    fully_implicit_sentiment: str = None                           # Stage 5


def assemble_quads(preds: SentencePredictions) -> list:
    """Combine one sentence's Stage 1-5 predictions into a final quad list.

    Coverage:
      1. Explicit-both quads: every pair Stage 2 confirmed, with Stage 3/4's
         category/sentiment.
      2. Implicit-aspect quads: explicit opinion spans NOT consumed by any
         Stage 2 pair, where Stage 5a confirmed an implicit aspect.
      3. Implicit-opinion quads: explicit aspect spans NOT consumed by any
         Stage 2 pair, where Stage 5b confirmed an implicit opinion.
      4. Fully-implicit quad: at most one per sentence (see
         docs/stage5_implicit_analysis.md's documented simplification),
         from Stage 5c.

    An aspect or opinion span that is neither paired (2) nor confirmed
    implicit-paired (5a/5b) produces NO quad -- this is the expected
    outcome for a Stage 1 false-positive span, not an error to work around.
    """
    quads = []

    matched_aspects = set()
    matched_opinions = set()
    for a1, a2, o1, o2 in preds.explicit_pairs:
        key = (a1, a2, o1, o2)
        matched_aspects.add((a1, a2))
        matched_opinions.add((o1, o2))
        quads.append({
            "a_start": a1, "a_end": a2,
            "category": preds.pair_category.get(key),
            "sentiment": preds.pair_sentiment.get(key),
            "o_start": o1, "o_end": o2,
        })

    for o_span in preds.opinion_spans:
        if o_span in matched_opinions:
            continue
        if preds.implicit_aspect_flag.get(o_span):
            quads.append({
                "a_start": IMPLICIT[0], "a_end": IMPLICIT[1],
                "category": preds.implicit_aspect_category.get(o_span),
                "sentiment": preds.implicit_aspect_sentiment.get(o_span),
                "o_start": o_span[0], "o_end": o_span[1],
            })

    for a_span in preds.aspect_spans:
        if a_span in matched_aspects:
            continue
        if preds.implicit_opinion_flag.get(a_span):
            quads.append({
                "a_start": a_span[0], "a_end": a_span[1],
                "category": preds.implicit_opinion_category.get(a_span),
                "sentiment": preds.implicit_opinion_sentiment.get(a_span),
                "o_start": IMPLICIT[0], "o_end": IMPLICIT[1],
            })

    if preds.fully_implicit_flag:
        quads.append({
            "a_start": IMPLICIT[0], "a_end": IMPLICIT[1],
            "category": preds.fully_implicit_category,
            "sentiment": preds.fully_implicit_sentiment,
            "o_start": IMPLICIT[0], "o_end": IMPLICIT[1],
        })

    return quads
