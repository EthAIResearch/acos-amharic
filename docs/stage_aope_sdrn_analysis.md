# Joint AOPE (SDRN-style): Design Notes

## Why this, and why now
The Stage 6 end-to-end run scored 14% accuracy -- far below any single
stage's isolated numbers (Stage 1 aspect F1 ~0.60, opinion F1 ~0.46; Stage
2 pairing F1 0.993 *on gold spans*; Stage 3 category ~0.59-0.64 macro-F1).
That gap is the signature of compounding pipeline error, and one link was
never actually measured: Stage 2's heuristic was validated only against
gold spans (docs/stage2_pairing_analysis.md's whole point was that real
pairing ambiguity is rare -- true when the spans themselves are correct,
untested when they aren't). Against Stage 1's actual noisy predictions,
a nearest-token-distance rule has no mechanism to recover from a missed or
spuriously-extracted span; whatever it decides for that sentence carries
straight into Stage 3/4/5 as normal-looking (wrong) input.

## Source
Chen, Liu, Wang, Zhang, Chi. "Synchronous Double-channel Recurrent Network
for Aspect-Opinion Pair Extraction." ACL 2020. Explicitly targets AOPE
(the exact task Stage 1+2 together perform) and reports that even the
best pipeline baseline (SPAN+RD) underperforms the joint synchronized
model by 1.14-3.39 F1 points on three SemEval benchmarks -- and that a
non-synchronized joint model (`SDRN w/o ESM&RSM`) is *less* competitive
than pipeline baselines, confirming synchronization specifically (not just
"joint training") is what closes the gap.

## What changed vs. the paper (and why)
- **Entity tagging**: kept this project's existing dual-head (separate
  aspect/opinion BIO) scheme instead of SDRN's single 5-way BA/IA/BP/IP/O
  sequence -- functionally equivalent, and reuses already-tested
  `bio_labels.py`/`align.py` rather than duplicating that logic.
- **Entity semantics (ESM)**: SDRN computes a soft same-entity probability
  per token pair from the CRF's label distribution (Eq. 8-9). This
  implementation uses a discrete approximation instead: decode BIO spans
  from the current step's argmax predictions, group tokens by shared span
  membership. Simpler, and avoids implementing a full CRF forward-backward
  pass from scratch -- worth revisiting if soft probabilities prove to
  matter empirically.
- **No CRF decoder**: uses independent per-token softmax (as Stage 1
  already does) rather than SDRN's CRF option. SDRN's own ablation (Table
  2) shows CRF and softmax-decoder variants trade wins depending on the
  dataset's implicit-target proportion -- not a clear-cut win either way.
- **Supervision timing**: matches the paper exactly -- only the FINAL
  recurrent step's entity and relation predictions are supervised (Eq. 3,
  6, 14, 15); earlier steps only refine hidden states via ESM/RSM.

## Verified before any model code ran
`relation_utils.py`'s matrix-building and correlation-degree functions are
pure Python (no torch) and were checked three ways before being wired into
the model: synthetic symmetric-matrix cases, a partial-match case (confirms
multi-word span scores average rather than round up), and a self-consistency
check against real training data (a matrix built from N real gold pairs
recovers all N pairs at exactly degree=1.0). The subword-projection logic
(word-level relation matrix -> subword-level, needed because the model
operates on tokenizer subwords) was separately verified to correctly
propagate a multi-subword aspect's relations to all of its subwords.

## What this does NOT change
Category (Stage 3) and sentiment (Stage 4) classification are unaffected
-- this model only replaces how (aspect_span, opinion_span) pairs are
produced. Stage 5's implicit handling and Stage 6's assembly/evaluation
also stay as built; `run_inference.py` will need updating to call this
model instead of Stage 1 + Stage 2's heuristic once this is trained and
validated, since it currently produces a richer output (spans + a scored
relation matrix) than the two-stage pipeline it replaces.

## Cost tradeoff, stated plainly
The N x N relation matrix (Eq. 4-5) costs roughly O(N^2) in both compute
and memory versus Stage 1's O(N) tagging -- default batch size dropped to
8 (from 16) accordingly. Expect a meaningfully slower training run than
Stage 1's AfroXLMR baseline; this is the direct cost of the architecture
that's supposed to fix the compounding-error problem, not a bug.
