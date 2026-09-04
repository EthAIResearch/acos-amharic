# Stage 3: Category Classification — Design Notes

## Scope
Trains only on explicit-both quads (23,617 of 41,224 total, 57.3%) --
consistent with Stage 2. Implicit-involving quads are Stage 5's job.

## Architecture
Shared encoder (same backbone family as Stage 1) → masked mean-pool over
the aspect span's subword tokens, the opinion span's subword tokens, and
the full sentence → concatenate all three → 2-layer MLP → 22-way softmax.
Sentence-level pooling is included alongside the two span poolings because
category is often a property of the broader topic, not just the aspect
term in isolation (e.g. "ግብር" (tax) alone is unambiguous, but a vaguer
aspect term may need sentence context to disambiguate ECONOMY#TAXATION
from GOVERNANCE#TRANSPARENCY).

## Known unlearnable categories (do not expect non-zero recall)
Two categories have **zero examples** in the explicit-pairs training data,
because they only ever co-occur with an implicit aspect or opinion in this
dataset:
- `PUBLIC_SERVICES#COMMUNITY_SUPPORT`
- `PUBLIC_SERVICES#INFRASTRUCTURE`

Separately, `ECONOMY#UTILITIES` has zero examples in the *full* training
set (both explicit and implicit) and only appears once in test -- it isn't
even in the 22-category label space Stage 3 trains against (see
`data/prepared/label_space.json`).

None of these are bugs to fix by relabeling -- the taxonomy is fixed and
external. They're documented limitations: report 0 recall for these
explicitly in the results table rather than omitting them, and note in the
paper that Stage 5 (implicit handling) is the only path to ever predicting
the first two, while the third needs more raw data collection regardless
of modeling approach.

## Imbalance handling
**Default: logit-adjusted loss** (Menon et al. 2021, "Long-Tail Learning via
Logit Adjustment") -- adds `tau * log(class_prior)` to each class's logit
before the softmax during training, directly correcting the classifier's
decision-boundary bias rather than just reweighting example loss magnitude.
This is the more effective fix per current research; naive class weighting
is a documented weak fix for severe imbalance (models often keep collapsing
to majority-class predictions even with capped inverse-frequency weights).

The older **class-weighted loss** (inverse-frequency, capped at 15x) is kept
as `--loss_type class_weighted` for comparison, but isn't the default.
Don't combine both approaches -- `PairClassifier` explicitly rejects passing
both `class_weights` and `log_priors` together, since stacking them tends to
over-correct.

## Evaluation
Report per-category P/R/F1 (`train.py`'s `per_class_prf`), not just
macro/micro-averaged numbers -- with 10 low-support categories, an average
alone would hide exactly the failure modes worth discussing in the paper.
