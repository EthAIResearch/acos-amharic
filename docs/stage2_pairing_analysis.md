# Stage 2: Aspect-Opinion Pairing — Design Decision

## Decision: heuristic, not learned

A distance-based heuristic pairer is used instead of a trained neural
classifier, based on data analysis (not assumption):

| Metric (train split, explicit-both quads) | Value |
|---|---|
| Sentences with any pairing ambiguity (candidates > true pairs) | 331 / 39,261 (0.84%) |
| Total negative candidate pairs | 704 (out of 24,146 total candidates) |
| True-pair median token distance | 1 |
| False-pair median token distance | 6 |
| Quads with exactly 1 opinion per aspect | 99.1% |
| Quads with exactly 1 aspect per opinion | 99.4% |

With only 704 negative examples spread across 39k sentences, there isn't
enough signal to train a reliable neural classifier — and the heuristic
already achieves near-ceiling performance:

| Split | Precision | Recall | F1 |
|---|---|---|---|
| Train | 0.9889 | 0.9976 | 0.9932 |
| Test | 0.9883 | 0.9976 | 0.9929 |

Matching train/test performance (no gap) confirms this isn't overfitting —
there's nothing to overfit, it's a fixed rule.

## How it works

For each aspect span, pair it with its nearest opinion span(s) by token
distance (ties included); do the same in reverse (each opinion → nearest
aspect). Take the union. See `candidates.py::heuristic_pairs`.

## Important caveat for Stage 6

This evaluates pairing **in isolation using gold spans**. End-to-end
pipeline performance will still be bottlenecked by Stage 1's span
extraction quality (aspect F1 ~0.60, opinion F1 ~0.46 as of the AfroXLMR
baseline) — a missed or wrong span means no correct pair is possible
regardless of how good the pairing logic is. The isolated 0.993 F1 here is
a ceiling for Stage 2 alone, not a preview of the full pipeline number.

## If ambiguity matters more in future data

If a future dataset revision has meaningfully more multi-quad sentences,
revisit this decision — the heuristic's blind spot is genuine semantic
ambiguity (e.g. two aspects, two opinions, but the *nearest* opinion isn't
the *correct* one), which a learned model could in principle resolve given
enough training examples. At current data volumes, that's not a fixable gap
with more model capacity — it's a data volume gap.
