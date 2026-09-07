# Stage 6: End-to-End Assembly and Evaluation — Design Notes

## Why three metric views, not one
Reporting a single combined exact-match F1 would hide the exact comparison
this project's earlier design decisions depend on: whether the
implicit-handling stages (5) actually help, versus what Stage 1-4 already
achieve on explicit-only quads. `evaluate.py` reports "all", "explicit",
and "implicit" separately, filtering both gold and predicted quads by
`is_implicit_quad()` per view -- consistent with Cai et al. (2021)'s own
practice of reporting implicit-involving performance separately.

## Verified against real data, not just synthetic cases
Beyond the 6 synthetic assembly scenarios and 4 synthetic evaluation
scenarios in `tests/`, both modules were sanity-checked with predictions
set equal to gold on the full real test set (5,138 quads): every subset
scores a perfect 1.0, and the explicit/implicit split's true-positive
counts (2,983 / 2,155) match the dataset statistics established when
Stage 5 was built, byte-for-byte. This confirms the evaluation logic
itself introduces no bias before a single model prediction is involved.

## Exact-match, no partial credit
A quad counts as correct only if aspect span, category, sentiment, and
opinion span all match gold exactly. Getting 3 of 4 fields right (e.g.
correct spans and category, wrong sentiment) scores as a full miss on that
quad -- the standard, if strict, ACOS convention. This makes the metric
harsh but comparable to other ACOS work using the same convention.

## Where errors compound
`run_inference.py`'s pipeline calls Stage 1 -> heuristic pairing -> Stage
3/4 -> Stage 5, in sequence, using each stage's actual *predictions* (not
gold) as input to the next. This matters because Stage 2's heuristic
pairing was validated at F1=0.993 only against **gold** spans
(docs/stage2_pairing_analysis.md) -- against Stage 1's predicted spans
(themselves at F1 ~0.60/0.46 for aspect/opinion as of the AfroXLMR
baseline), pairing quality will likely be lower, since a missed or
spuriously-extracted span changes the candidate set pairing operates on.
This is expected error propagation, not a sign Stage 2's heuristic
decision was wrong -- report Stage 6's end-to-end number as the honest
full-pipeline result, and the isolated per-stage numbers (already reported
in Stages 1-5's own docs) as diagnostics for where the loss happens.

## Partial pipelines are supported
Not every Stage 5 sub-model needs to be trained before running Stage 6 --
`configs/stage6_pipeline.yaml` treats the three category/sentiment
anchor-specific checkpoints for aspect-anchor and fully-implicit as
optional (`null` is fine). Sentences that would have needed an untrained
sub-model simply don't get that quad, which shows up as a straightforward
recall gap in the "implicit" subset rather than a crash -- useful for
tracking incremental progress as Stage 5's 9 sub-models get trained one at
a time rather than requiring all-or-nothing completion.
