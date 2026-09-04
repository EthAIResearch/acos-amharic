# Stage 4: Sentiment Classification — Design Notes

## Architecture
Identical to Stage 3 -- same `PairClassifier` (see `src/common/pair_model.py`),
just `num_labels=3` instead of 22 and the sentiment field instead of category.
Both stages were refactored to share this model class rather than duplicate it.

## The imbalance is worse than it first looked
Full-dataset sentiment distribution (all quads, including implicit):
63.8% NEGATIVE / 27.5% POSITIVE / 8.7% NEUTRAL.

But Stage 4 trains only on explicit-both quads (same scope as Stage 3), and
within that subset NEUTRAL drops further, to **3.1%** (741 of 23,617
examples). Implicit-opinion cases disproportionately carry neutral
sentiment -- plausibly because a neutral statement is less likely to
contain a strong, explicit opinion word for the tagger to find in the
first place. This wasn't obvious from the full-dataset stats alone.

## Backbone choice: worth testing roberta-base-amharic first, not AfroXLMR
On rasyosef's own Amharic sentiment benchmark (document-level, likely more
balanced than our aspect-level task -- see caveat below), `roberta-base-
amharic` (110M) scores **F1=0.88**, beating `afro-xlmr-base` (278M, F1=0.83)
and even `afro-xlmr-large` (560M, F1=0.86):

| Model | Params | F1 |
|---|---|---|
| roberta-base-amharic | 110M | 0.88 |
| afro-xlmr-large | 560M | 0.86 |
| afro-xlmr-base | 278M | 0.83 |
| xlm-roberta-base | 279M | 0.83 |

**Caveat**: that benchmark is document-level sentiment on a different
dataset, not aspect-level span-conditioned classification with our severe
NEUTRAL scarcity (3.1%). The absolute numbers don't transfer -- but it's a
task-specific (not just generic-benchmark) signal that the Amharic-native
tokenizer's advantage may matter most on sentiment specifically. Worth
running `configs/stage4_roberta_amharic.yaml` as a primary comparison, not
just AfroXLMR-base -- and comparing NEUTRAL_f1 between the two specifically,
since that's the class where representation quality would matter most.

## What the research says (2024-2025 ABSA literature)
- Neutral is consistently the hardest class under imbalance across ABSA
  papers, independent of dataset or language -- this isn't specific to
  Amharic or to this dataset's generation process.
- Class-weighted loss and macro-F1 (not accuracy) are standard baseline
  mitigations -- both already implemented (Stage 3's `pair_utils.py` is
  reused as-is).
- **Logit-adjusted loss** (Menon et al. 2021) is the more effective fix per
  current research, same as Stage 3 -- used as the default here too.
- **Synthetic data augmentation** (backtranslation, paraphrasing) is a
  documented ABSA-specific technique for class imbalance, shown to
  meaningfully raise minority-class F1 in recent work. Not implemented yet
  -- worth trying specifically for NEUTRAL if logit adjustment alone isn't
  enough, since 741 real examples is a thin base to learn from regardless
  of loss function.

## What to watch when evaluating a run
Report NEUTRAL's precision/recall/F1 individually, not just accuracy or
macro-F1 -- with NEGATIVE at 65% of the data, a model that never predicts
NEUTRAL at all can still reach ~65% accuracy while being useless for that
class. `train.py` prints `NEUTRAL_f1` every epoch specifically so this
doesn't get missed the way Stage 3's macro-F1 artifact did.
