# Stage 5: Implicit Aspect/Opinion Detection — Design Notes

## Research grounding
Cai et al. (2021), who introduced the ACOS task, report that over 30% of
sentiment expressions contain implicit language -- our measured 32.1%
implicit-aspect rate matches this closely, suggesting our dataset's
implicit-phenomenon rate is consistent with established ABSA benchmarks,
not an artifact of the generation/annotation process. Current literature
(2024-2025) confirms performance on implicit cases remains the hardest
open problem in ABSA generally, and that extract-then-classify decomposition
(our approach) remains a standard, valid paradigm alongside newer
generative and graph-based methods.

## Three sub-problems, quantified against real data
Unlike Stage 2 (where real pairing ambiguity was <1% of sentences and a
learned classifier was rejected for lack of signal), all three Stage 5
sub-problems have thousands of genuine training examples:

| Sub-problem | Anchor | Positive | Negative | % positive |
|---|---|---|---|---|
| Detect implicit aspect | explicit opinion span | 9,693 | 23,617 | 29.1% |
| Detect implicit opinion | explicit aspect span | 4,560 | 23,617 | 16.2% |
| Detect fully-implicit quad | none (sentence-level) | 3,324 sentences | 35,937 sentences | 8.5% |

This justifies building real learned classifiers here, unlike Stage 2.

## Architecture: no new model needed
`PairClassifier` (Stage 3/4's shared architecture) pools the aspect span,
opinion span, and full sentence via masked mean-pooling, then concatenates
and classifies. An all-zero mask correctly reduces to a zero-vector in the
masked-mean computation (division is clamped to avoid NaN), so the exact
same class handles:
- **Detection** (binary, num_labels=2): one span mask real, one zero.
- **Category/sentiment for single-side-implicit quads**: same as detection,
  but classifying into the 22-category or 3-sentiment space using only the
  confirmed-implicit examples.
- **Category/sentiment for fully-implicit quads**: both masks zero,
  classification relies entirely on the sentence-pooled representation.

No architecture change was required -- only new dataset construction
(`stage5_implicit/dataset.py`) routing each quad to the right sub-problem
via `pair_utils.quad_kind()`.

## Nine trainable sub-models, one script
`train.py` is parameterized by `--subtask` and (for category/sentiment)
`--anchor`, reusing the identical AMP + logit-adjustment + per-class-F1
training loop from Stage 3/4. Nine total configurations:
3 binary detectors + (category, sentiment) x (opinion-anchor, aspect-anchor,
no-anchor). Only 4 example configs are included in `configs/` (the 3
detectors plus one category example); the remaining 5 follow by copying
the pattern documented in `stage5_category_opinion_anchor.yaml`'s comments.

## Known simplification
`FullyImplicitCategorySentimentDataset` takes only the *first* fully-implicit
quad per sentence when multiple exist. Only ~30 of 3,324 train sentences
with a fully-implicit quad have more than one (per earlier analysis),
so this affects under 1% of this sub-problem's examples -- documented here
so it isn't rediscovered as a surprise later, not because it's expected to
matter much in practice.

## Expected accuracy ceiling, going in
Each anchored sub-model trains on far fewer examples than Stage 3's 23,617
(9,693 / 4,560 / 3,354 respectively), and the fully-implicit case has no
span signal at all -- expect Stage 5's category/sentiment classifiers to
underperform Stage 3/4's explicit-pairs numbers. This is a data volume
limitation, not necessarily a sign the approach is wrong; report Stage 5
results separately rather than pooling with Stage 3/4's numbers.
