# References

## Directly implemented / adapted
- **Cai, Xia, Yu. "Aspect-Category-Opinion-Sentiment Quadruple Extraction
  with Implicit Aspects and Opinions." ACL-IJCNLP 2021.** — defines the
  ACOS task this whole project targets; our pipeline decomposition
  (extract -> pair -> classify -> handle implicit) converges independently
  on a structure close to their Extract-Classify-ACOS baseline.
- **Chen, Liu, Wang, Zhang, Chi. "Synchronous Double-channel Recurrent
  Network for Aspect-Opinion Pair Extraction." ACL 2020 (SDRN).** —
  adapted for `src/stage_aope_sdrn/`, replacing the Stage 1+2 pipeline
  with a jointly-trained, synchronized extraction+pairing model. See
  `docs/stage_aope_sdrn_analysis.md`.
- **Menon, Jayasumana, Rawat, Jain, Veit, Kumar. "Long-Tail Learning via
  Logit Adjustment." ICLR 2021.** — the default loss for Stage 3/4/5's
  severe class imbalance.
- **Alabi, Adelani, Mosbach, Klakow. "Adapting Pre-trained Language Models
  to African Languages via Multilingual Adaptive Fine-Tuning." COLING
  2022.** — AfroXLMR, the default backbone throughout.

## Read, not (yet) implemented — candidates for future stages
- **Wan, Yang, Du, Liu, Qi, Pan. "Target-Aspect-Sentiment Joint Detection
  for Aspect-Based Sentiment Analysis." AAAI 2020 (TAS-BERT).** — jointly
  predicts a yes/no "does an implicit target exist" flag alongside span
  extraction, in a single model with combined loss. Directly relevant to
  Stage 5: TAS-BERT's binary-existence-flag + span-extraction design is
  close to what Stage 5's `detect_implicit_aspect`/`detect_implicit_opinion`
  sub-models do *separately* — worth exploring whether folding them into
  one jointly-trained model (rather than 9 independent sub-models) reduces
  Stage 5's own error propagation the same way SDRN does for Stage 1+2.
- **Phan & Ogunbona. "Modelling Context and Syntactical Features for
  Aspect-based Sentiment Analysis." ACL 2020 (CSAE / LCFS-ASC).** —
  POS + dependency-embedding-augmented extraction and syntactic-relative-
  distance-weighted sentiment classification. Requires a dependency parser
  for Amharic, which may not exist at adequate quality — worth checking
  before investing effort here.

## Curated further reading
https://github.com/1429904852/Aspect-Based-Sentiment-Analysis — comprehensive
ABSA paper list. Notable entries not yet explored for this project:
- **Xu et al., "Learning Span-Level Interactions for Aspect Sentiment
  Triplet Extraction." ACL 2021 (Span-ASTE)** — span-level (not token-level)
  joint extraction+pairing, a different architectural family from SDRN's
  token-pair attention; possible alternative to `stage_aope_sdrn` if the
  SDRN adaptation's discrete ESM approximation underperforms.
- **Wu et al., "Grid Tagging Scheme for Aspect-oriented Fine-grained
  Opinion Extraction." Findings of EMNLP 2020 (GTS)** — unified table-filling
  formulation for pair/triplet extraction in one tagging pass.
- **Zhang et al., "Aspect Sentiment Quad Prediction as Paraphrase
  Generation." EMNLP 2021 (ASBA-QUAD)** — the generative-paraphrase
  alternative to extraction-based ACOS, noted but not adopted (Section
  2.1 of the paper draft) given Amharic's small share of generative
  multilingual models' pretraining data.
