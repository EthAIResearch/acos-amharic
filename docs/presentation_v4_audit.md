# Presentation v4.0 Evidence Audit

Audit date: 2026-10-02.

Checkout: `7e40773050395fcd5a9c0408ec0c3206a81550d4`. Repository: https://github.com/EthAIResearch/acos-amharic.

Original: [Amharic_ACOS_Research_Presentation(v3.0).pptx](Amharic_ACOS_Research_Presentation%28v3.0%29.pptx).
Updated: [Amharic_ACOS_Research_Presentation(v4.0).pptx](Amharic_ACOS_Research_Presentation%28v4.0%29.pptx).

## Scope

All 66 PRs and 39 issues through #105 were reviewed across open and closed states. 65 PRs are merged; #9 was closed unmerged. All 39 issues are closed. The paginated snapshot includes 73 conversation comments, 26 PR reviews, 43 inline comments and 351 changed-file entries; no nested review/file connection required another page.

The codebase review followed every source family and its configs, tests and result artifacts. This is a presentation/content audit, not a claim that every line has passed a correctness or security review. Local uncommitted benchmark files were inspected without modifying or executing their model requests.

## Important Corrections

- Domain is civic/governance commentary, not e-commerce/hospitality.
- 22 declared categories, 23 observed across splits; one test-only ECONOMY#UTILITIES annotation is outside the allowed list. Neither 13 nor 29 is the current declared count.
- Original train/dev/test sizes are 39,261/4,907/4,909 sentences and 41,224/5,129/5,138 quads. Implicitness is 42.71% in train but 41.94% in original test.
- Clean train/dev/test sizes are 38,385/4,892/4,908 sentences and 40,426/5,110/5,133 quads. No existing model score is relabeled as a clean-data result.
- Stage 5 contains learned implicit detectors and optional category/sentiment paths. Generation is not the only possible NULL-handling method.
- Do not multiply component F1 scores to derive end-to-end recall. Issue #51 reports approximately 14% pipeline F1; no standalone pipeline metric JSON was found.
- Span-ASTE default result files give Run 1/2/3 AOPE F1 of 30.80%/32.82%/31.24%, OTE F1 of 39.48%/42.87%/42.89%, and OTE recall of 30.52%/35.57%/46.73%.
- Run 2's 34.02% and Run 3 sweep's 45.78% are candidate conversion rates, not AOPE F1 or pair recall. Run 3 best_metrics conversion is 45.04%; separate sweep/default artifacts disagree and remain labeled separately.
- Span-ASTE Run 2 dev-best AOPE threshold is 0.50, with 32.90% test sweep F1. Its 33.31% at 0.35 is a test-selected diagnostic peak. Run 3 dev-best threshold is 0.65, with 31.97% test sweep F1; 32.04% at 0.60 is test-selected.
- Run 3 enabled mean pooling and weighted cross-entropy, with weights [0.1, 1.0, 1.0, 2.0]. use_focal_loss is false. These changes were not isolated ablations.
- Prefix pooling uses half-open [s,e) spans: (P[e]-P[s])/(e-s), after an O(n) prefix pass. Endpoints are contextual representations, so saying they completely ignore internal words is too strong.
- 55.20% aspect F1 is best among the three Span-ASTE runs, not repository-wide: SDRN CRF Run 1 records 60.41%.
- ByT5 Run 1 effective batch is 4 x 4 = 16, source/target limits are 768/256, and best-dev metrics cover the first 1,000 sentences (1,033 gold quads), not full dev.
- ACSE F1 is not ATE F1. Explicit-only AOPE/ASTE and all-quad ByT5 scores are not a directly comparable leaderboard.
- Constrained decoding and clitic metrics are implemented in PR #100. Canonical ordering and smoothing are implemented in #101. No corresponding completed ByT5 result is recorded.
- Reproduced FSM delimiter stall after a complete category/sentiment followed by space. The required pipe is disallowed. Free-text slots also allow structural bytes. Do not claim guaranteed valid outputs.
- ByT5 Run 2 still uses prepared/. Training's internal evaluator does not enable the configured standalone FSM. Clean SDRN config uses clean paths but reuses an existing output directory.
- Local 500-sample hybrid result is 8.02% strict / 8.22% normalized F1, from 100 fixed predictions plus 400 keyword-rule outputs. The other local file records 1.96% strict F1 with 90 predictions over 500 sentences and lacks model/run provenance. Neither is a controlled LLM leaderboard entry.
- Unseen phrases are not tokenizer unknown tokens. Byte representation coverage does not prove linguistic generalization. Claims of inherent BIO incompatibility and universally necessary generation were removed.

## Newly Added Evidence

Dataset provenance, split sizes and authentic Amharic examples; duplicate/leakage cleanup; schema mismatch; Stage 1/3/4 backbone comparisons; Stage 5 detector results; SDRN reweighting, Dice and word-level history; seen/unseen opinion diagnostics; Run 2 status; metric definitions; local benchmark caveats; code/CI limitations; prioritized validation roadmap.

## Verification and Limits

- Original PowerPoint hash checked before and after each build; original PDF also left unchanged.
- Generated deck reopened; slide count and unique ZIP package parts checked.
- Dataset counts, declared/observed labels, normalized-text duplicates and pairwise split overlap recomputed from JSONL.
- Clean data independently checked for zero normalized-text duplicates, overlap, duplicate quads and same-span sentiment conflicts. This does not establish human annotation correctness or absence of semantic/near-duplicate leakage.
- Decoder limitation reproduced with the repository's MockByT5Tokenizer. No ML source files were changed to repair it.
- Static test inventory: 12 files, 88 test functions. Full ML tests, training and GPU evaluation were not run. CI's minimal dependencies and optional import guards do not establish tensor-path coverage.
- No external LLM requests, training jobs, GitHub edits, commits or branch changes were performed.
- Final visual and PDF checks are performed after generation; see the delivered PDF for the rendered artifact.

## Slide-Level Sources

### 1. Research scope and current status

- [data/README.md](../data/README.md)
- [results/stage_byt5_acos/byt5_base_run1/test_metrics.json](../results/stage_byt5_acos/byt5_base_run1/test_metrics.json)

### 2. Amharic ACOS: Four Fields, One Exact Match

- [src/common/data_prep.py](../src/common/data_prep.py)
- [src/stage_byt5_acos/linearization.py](../src/stage_byt5_acos/linearization.py)
- [src/pipeline/evaluate.py](../src/pipeline/evaluate.py)

### 3. A Civic-Domain Dataset, Not a Review Benchmark

Counts recomputed from the JSONL files. No independent human annotation-quality study is recorded.

- [data/README.md](../data/README.md)
- [data/prepared/train.jsonl](../data/prepared/train.jsonl)
- [data/prepared/test.jsonl](../data/prepared/test.jsonl)

### 4. A Cleaned Dataset Is Now Available

Independent checks found zero normalized-text duplicates, split overlap, duplicate quads, and same-span sentiment conflicts in prepared_clean. Sentiment normalization is an automatic heuristic, not human adjudication.

- [data/clean_dataset.py](../data/clean_dataset.py)
- [data/prepared_clean/cleaning_report.json](../data/prepared_clean/cleaning_report.json)
- [https://github.com/EthAIResearch/acos-amharic/pull/104](https://github.com/EthAIResearch/acos-amharic/pull/104)

### 5. 22 Declared Categories; 23 Observed Labels

ECONOMY#UTILITIES occurs once in original and clean test data, never in train/dev, and is absent from CATEGORIES and label_space.json. Older README says 13; v3.0 says 29. Neither matches current declared metadata.

- [data/prepared/label_space.json](../data/prepared/label_space.json)
- [src/stage_byt5_acos/linearization.py](../src/stage_byt5_acos/linearization.py)
- [data/prepared_clean/test.jsonl](../data/prepared_clean/test.jsonl)

### 6. Explicit and NULL Fields in the Actual Dataset

Examples are read directly from 1-based rows 3367 and 151. The implicit example has a category annotation without an explicit educational target, illustrating why human adjudication is important.

- [data/prepared/test.jsonl#L3367](../data/prepared/test.jsonl#L3367)
- [data/prepared/test.jsonl#L151](../data/prepared/test.jsonl#L151)
- [src/stage_byt5_acos/linearization.py](../src/stage_byt5_acos/linearization.py)

### 7. Six Stages, Including Learned Implicit Handling

- [src/pipeline/run_inference.py](../src/pipeline/run_inference.py)
- [src/pipeline/assemble_quads.py](../src/pipeline/assemble_quads.py)
- [https://github.com/EthAIResearch/acos-amharic/pull/47](https://github.com/EthAIResearch/acos-amharic/pull/47)

### 8. Backbone Benchmarks Across Three Subtasks

- [results/stage1/](../results/stage1/)
- [results/stage3/](../results/stage3/)
- [results/stage4/](../results/stage4/)
- [https://github.com/EthAIResearch/acos-amharic/pull/36](https://github.com/EthAIResearch/acos-amharic/pull/36)
- [https://github.com/EthAIResearch/acos-amharic/pull/33](https://github.com/EthAIResearch/acos-amharic/pull/33)

### 9. Detection Is Not Complete Quadruple Recovery

- [docs/stage5_implicit_analysis.md](../docs/stage5_implicit_analysis.md)
- [results/stage5/](../results/stage5/)
- [configs/stage6_pipeline.yaml](../configs/stage6_pipeline.yaml)
- [https://github.com/EthAIResearch/acos-amharic/pull/34](https://github.com/EthAIResearch/acos-amharic/pull/34)

### 10. Error Propagation: Measured, Not an F1 Product

The original deck's 14.2% is a historical narrative figure, not independently reproduced here. Multiplying component F1 scores is not an end-to-end recall derivation.

- [https://github.com/EthAIResearch/acos-amharic/issues/51](https://github.com/EthAIResearch/acos-amharic/issues/51)
- [docs/stage2_pairing_analysis.md](../docs/stage2_pairing_analysis.md)
- [src/pipeline/run_inference.py](../src/pipeline/run_inference.py)

### 11. SDRN: Entity and Relation Feedback

- [src/stage_aope_sdrn/model.py](../src/stage_aope_sdrn/model.py)
- [src/stage_aope_sdrn/crf.py](../src/stage_aope_sdrn/crf.py)
- [https://github.com/EthAIResearch/acos-amharic/pull/64](https://github.com/EthAIResearch/acos-amharic/pull/64)
- [https://github.com/EthAIResearch/acos-amharic/pull/75](https://github.com/EthAIResearch/acos-amharic/pull/75)

### 12. What Helped, and What Did Not

CRF Run 1 best_metrics uses threshold 0.20; other rows are recorded defaults around 0.50. Do not interpret these as identical operating points.

- [results/stage_aope_sdrn/](../results/stage_aope_sdrn/)
- [https://github.com/EthAIResearch/acos-amharic/pull/70](https://github.com/EthAIResearch/acos-amharic/pull/70)
- [https://github.com/EthAIResearch/acos-amharic/pull/78](https://github.com/EthAIResearch/acos-amharic/pull/78)
- [https://github.com/EthAIResearch/acos-amharic/pull/80](https://github.com/EthAIResearch/acos-amharic/pull/80)
- [https://github.com/EthAIResearch/acos-amharic/issues/76](https://github.com/EthAIResearch/acos-amharic/issues/76)

### 13. Seen vs. Unseen: Token Overlap & Span Boundary Breakdown

Evaluates recall across exact full phrase, >=1 token overlap, and character trigrams for both aspects and opinions.

- [results/stage_aope_sdrn/afroxlmr_word_sdrn_run1/seen_unseen_diagnostic.json](../results/stage_aope_sdrn/afroxlmr_word_sdrn_run1/seen_unseen_diagnostic.json)
- [https://github.com/EthAIResearch/acos-amharic/issues/76](https://github.com/EthAIResearch/acos-amharic/issues/76)

### 14. Candidate Coverage Is Not Final Pair Recall

- [results/stage_aope_sdrn/afroxlmr_crf_run1/best_metrics.json](../results/stage_aope_sdrn/afroxlmr_crf_run1/best_metrics.json)
- [results/stage_span_aste/afroxlmr_pooled_focal_run3/best_metrics.json](../results/stage_span_aste/afroxlmr_pooled_focal_run3/best_metrics.json)

### 15. Span-ASTE: Enumerate, Prune, Then Classify

- [src/stage_span_aste/model.py](../src/stage_span_aste/model.py)
- [src/stage_span_aste/span_utils.py](../src/stage_span_aste/span_utils.py)
- [https://github.com/EthAIResearch/acos-amharic/pull/82](https://github.com/EthAIResearch/acos-amharic/pull/82)
- [https://github.com/EthAIResearch/acos-amharic/pull/88](https://github.com/EthAIResearch/acos-amharic/pull/88)

### 16. High Candidate Coverage, Low Conversion

- [results/stage_span_aste/afroxlmr_run1/best_metrics.json](../results/stage_span_aste/afroxlmr_run1/best_metrics.json)
- [https://github.com/EthAIResearch/acos-amharic/issues/84](https://github.com/EthAIResearch/acos-amharic/issues/84)
- [https://github.com/EthAIResearch/acos-amharic/pull/86](https://github.com/EthAIResearch/acos-amharic/pull/86)

### 17. A Better Relation Model, With Modest Gains

- [results/stage_span_aste/afroxlmr_biaffine_run2/best_metrics.json](../results/stage_span_aste/afroxlmr_biaffine_run2/best_metrics.json)
- [src/stage_span_aste/model.py](../src/stage_span_aste/model.py)
- [https://github.com/EthAIResearch/acos-amharic/pull/91](https://github.com/EthAIResearch/acos-amharic/pull/91)

### 18. Full-Span Pooling and Cost-Sensitive Loss

- [src/stage_span_aste/model.py](../src/stage_span_aste/model.py)
- [results/stage_span_aste/afroxlmr_pooled_focal_run3/resolved_args.json](../results/stage_span_aste/afroxlmr_pooled_focal_run3/resolved_args.json)
- [https://github.com/EthAIResearch/acos-amharic/pull/93](https://github.com/EthAIResearch/acos-amharic/pull/93)

### 19. More Opinions Found, Not a Universal F1 Gain

- [results/stage_span_aste/afroxlmr_run1/best_metrics.json](../results/stage_span_aste/afroxlmr_run1/best_metrics.json)
- [results/stage_span_aste/afroxlmr_pooled_focal_run3/best_metrics.json](../results/stage_span_aste/afroxlmr_pooled_focal_run3/best_metrics.json)

### 20. Dev-Tuned Results Are Not Test-Selected Peaks

best_metrics and threshold_sweep do not exactly reproduce each other at 0.50. Keep their provenance separate; do not mix counts or derive gains across files without rerunning a frozen evaluator.

- [results/stage_span_aste/afroxlmr_biaffine_run2/dev_threshold_sweep.json](../results/stage_span_aste/afroxlmr_biaffine_run2/dev_threshold_sweep.json)
- [results/stage_span_aste/afroxlmr_pooled_focal_run3/threshold_sweep.json](../results/stage_span_aste/afroxlmr_pooled_focal_run3/threshold_sweep.json)

### 21. ByT5: One Model for Explicit and Implicit Quads

- [src/stage_byt5_acos/model.py](../src/stage_byt5_acos/model.py)
- [src/stage_byt5_acos/dataset.py](../src/stage_byt5_acos/dataset.py)
- [src/stage_byt5_acos/linearization.py](../src/stage_byt5_acos/linearization.py)
- [https://github.com/EthAIResearch/acos-amharic/pull/95](https://github.com/EthAIResearch/acos-amharic/pull/95)

### 22. The Recorded Training Setup

The issue reports NaN loss under FP16 and a switch to BF16. Timing and memory reduction claims are issue-level observations, not profiled artifacts reproduced in this audit.

- [results/stage_byt5_acos/byt5_base_run1/resolved_args.json](../results/stage_byt5_acos/byt5_base_run1/resolved_args.json)
- [src/stage_byt5_acos/train.py](../src/stage_byt5_acos/train.py)
- [https://github.com/EthAIResearch/acos-amharic/issues/98](https://github.com/EthAIResearch/acos-amharic/issues/98)

### 23. ByT5 Dev: Strict Quadruple and Subtask Metrics

- [results/stage_byt5_acos/byt5_base_run1/best_dev_metrics.json](../results/stage_byt5_acos/byt5_base_run1/best_dev_metrics.json)
- [https://github.com/EthAIResearch/acos-amharic/issues/98](https://github.com/EthAIResearch/acos-amharic/issues/98)

### 24. ByT5 Test: Strict Quadruple and Subtask Metrics

- [results/stage_byt5_acos/byt5_base_run1/test_metrics.json](../results/stage_byt5_acos/byt5_base_run1/test_metrics.json)
- [https://github.com/EthAIResearch/acos-amharic/issues/98](https://github.com/EthAIResearch/acos-amharic/issues/98)

### 25. Constrained Decoding and Linguistic Diagnostics

Reproduced with the repository's mock tokenizer: '[ item | PUBLIC_SAFETY#CRIME ' allows only a space, not '|'. The same occurs after completed sentiment plus space. Inline review discussion_r4105131434 identifies this issue.

- [src/stage_byt5_acos/constrained_decoder.py](../src/stage_byt5_acos/constrained_decoder.py)
- [src/stage_byt5_acos/evaluate.py](../src/stage_byt5_acos/evaluate.py)
- [tests/test_constrained_decoder.py](../tests/test_constrained_decoder.py)
- [https://github.com/EthAIResearch/acos-amharic/pull/100](https://github.com/EthAIResearch/acos-amharic/pull/100)

### 26. ByT5 Run 2: Implemented, Not Yet Measured

- [src/stage_byt5_acos/linearization.py](../src/stage_byt5_acos/linearization.py)
- [src/stage_byt5_acos/train.py](../src/stage_byt5_acos/train.py)
- [configs/stage_byt5_acos_run2.yaml](../configs/stage_byt5_acos_run2.yaml)
- [https://github.com/EthAIResearch/acos-amharic/pull/101](https://github.com/EthAIResearch/acos-amharic/pull/101)

### 27. Two Evaluation Populations, Not One Leaderboard

- [results/stage_aope_sdrn/afroxlmr_crf_run1/best_metrics.json](../results/stage_aope_sdrn/afroxlmr_crf_run1/best_metrics.json)
- [results/stage_span_aste/](../results/stage_span_aste/)
- [results/stage_byt5_acos/byt5_base_run1/test_metrics.json](../results/stage_byt5_acos/byt5_base_run1/test_metrics.json)

### 28. Exact Match, Subsets and Morphology Must Stay Separate

- [src/pipeline/evaluate.py](../src/pipeline/evaluate.py)
- [src/stage_byt5_acos/evaluate.py](../src/stage_byt5_acos/evaluate.py)
- [src/stage_byt5_acos/linearization.py](../src/stage_byt5_acos/linearization.py)

### 29. The 500-Sample Files Are Not an LLM Leaderboard

No external model API was called in this update. The hybrid benchmark imports 100 fixed predictions and generates the remaining 400 using a keyword lexicon and span heuristics. The other JSON lacks model/version/prompt/request-status metadata.

- [scripts/benchmark_llm_500.py](../scripts/benchmark_llm_500.py)
- [scripts/llm_benchmark_full_500.py](../scripts/llm_benchmark_full_500.py)
- [results/llm_test_500_benchmark.json](../results/llm_test_500_benchmark.json)
- [results/llm_benchmark_500.json](../results/llm_benchmark_500.json)

### 30. What Exists, and What Still Needs Verification

Static inventory: 12 test files with 88 test functions. CI installs ruff, pytest and pyyaml, not PyTorch/Transformers; optional guards can return without exercising tensor paths. Full ML training and GPU tests were not run for this presentation update.

- [src/common/](../src/common/)
- [src/pipeline/](../src/pipeline/)
- [tests/](../tests/)
- [.github/workflows/ci.yml](../.github/workflows/ci.yml)
- [configs/stage_aope_sdrn_clean.yaml](../configs/stage_aope_sdrn_clean.yaml)

### 31. Data and Evaluation Before Another Scale-Up

- [https://github.com/EthAIResearch/acos-amharic/pull/100](https://github.com/EthAIResearch/acos-amharic/pull/100)
- [https://github.com/EthAIResearch/acos-amharic/pull/101](https://github.com/EthAIResearch/acos-amharic/pull/101)
- [https://github.com/EthAIResearch/acos-amharic/issues/103](https://github.com/EthAIResearch/acos-amharic/issues/103)
- [https://github.com/EthAIResearch/acos-amharic/pull/104](https://github.com/EthAIResearch/acos-amharic/pull/104)
- [https://github.com/EthAIResearch/acos-amharic/pull/105](https://github.com/EthAIResearch/acos-amharic/pull/105)
- [configs/stage_byt5_acos_run2.yaml](../configs/stage_byt5_acos_run2.yaml)

### 32. Repository-Wide Review Behind This Update

Reviewed issue/PR bodies, changed-file inventories, discussion/review records and the working-tree implementation/results. This is a presentation evidence audit, not an exhaustive security or line-by-line correctness audit.

- [https://github.com/EthAIResearch/acos-amharic/pull/105](https://github.com/EthAIResearch/acos-amharic/pull/105)
- [docs/presentation_v4_audit.md](../docs/presentation_v4_audit.md)

### 33. Bibliography (retained from v3.0)

- [docs/references.md](../docs/references.md)

## Every Issue and PR

Low-level fixes and tooling changes are grouped into the relevant method/provenance section rather than given one slide each.

| Record | State | Title | Presentation disposition |
| --- | --- | --- | --- |
| [Issue #1](https://github.com/EthAIResearch/acos-amharic/issues/1) | closed | [EXP] Train Stage 1 baseline: AfroXLMR-base joint ATE+OTE | Dataset variants and Stage 1 baseline comparison. |
| [Issue #2](https://github.com/EthAIResearch/acos-amharic/issues/2) | closed | [EXP] Train Stage 1 lightweight baseline: bert-small-amharic | Dataset variants and Stage 1 baseline comparison. |
| [PR #3](https://github.com/EthAIResearch/acos-amharic/pull/3) | merged | Feat/stage1 configs and data | Dataset variants and Stage 1 baseline comparison. |
| [PR #4](https://github.com/EthAIResearch/acos-amharic/pull/4) | merged | fix(stage1): resolve import paths for common and stage1 modules | Dataset variants and Stage 1 baseline comparison. |
| [Issue #5](https://github.com/EthAIResearch/acos-amharic/issues/5) | closed | [DATA] Creating a fully explicit data | Dataset variants and Stage 1 baseline comparison. |
| [PR #6](https://github.com/EthAIResearch/acos-amharic/pull/6) | merged | feat(data): add explicit-only dataset and update stage-1 configs | Dataset variants and Stage 1 baseline comparison. |
| [PR #7](https://github.com/EthAIResearch/acos-amharic/pull/7) | merged | exp(stage1): train bertsmall-base baseline(#2) | Dataset variants and Stage 1 baseline comparison. |
| [PR #8](https://github.com/EthAIResearch/acos-amharic/pull/8) | merged | exp(stage1): train afroxlmr-base baseline (#1) | Dataset variants and Stage 1 baseline comparison. |
| [PR #9](https://github.com/EthAIResearch/acos-amharic/pull/9) | closed | fix(exp) rename bertsmall result folder for convinience | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #10](https://github.com/EthAIResearch/acos-amharic/pull/10) | merged | fix(exp) rename bertsmall result folder for convinience | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #11](https://github.com/EthAIResearch/acos-amharic/issues/11) | closed | [Feature] Build aspect–opinion pairing module | Gold-span nearest-distance pairing and error propagation. |
| [PR #12](https://github.com/EthAIResearch/acos-amharic/pull/12) | merged | Feature/stage2 pairing | Gold-span nearest-distance pairing and error propagation. |
| [Issue #13](https://github.com/EthAIResearch/acos-amharic/issues/13) | closed | [FEATURE] Build category classification head (22-class) | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #14](https://github.com/EthAIResearch/acos-amharic/pull/14) | merged | feat(stage3): build 22-class category classification head | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #15](https://github.com/EthAIResearch/acos-amharic/pull/15) | merged | fix(stage3): ensure class_weights buffer moves to GPU with model | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #16](https://github.com/EthAIResearch/acos-amharic/issues/16) | closed | [EXP] Train Stage 3 lightweight baseline: bert-small-amharic | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #17](https://github.com/EthAIResearch/acos-amharic/pull/17) | merged | feat(config): add stage-3 bert-small experiment configuration (#16) | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #18](https://github.com/EthAIResearch/acos-amharic/issues/18) | closed | [EXP] Train Stage 3 baseline: AfroXLMR-base category classification | Category head, imbalance handling and dual macro-F1 reporting. |
| [Issue #19](https://github.com/EthAIResearch/acos-amharic/issues/19) | closed | [EXP] Train Stage 3 lightweight baseline: bert-small-amharic category classification | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #20](https://github.com/EthAIResearch/acos-amharic/pull/20) | merged | feat(stage3): add logit adjustment loss and dual macro-F1 evaluation metrics | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #21](https://github.com/EthAIResearch/acos-amharic/pull/21) | merged | exp(stage3): commit bertsmall stage 3 results (#19) | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #22](https://github.com/EthAIResearch/acos-amharic/pull/22) | merged | exp(stage3): commit afroxlmr stage 3 results (#18) | Category head, imbalance handling and dual macro-F1 reporting. |
| [Issue #23](https://github.com/EthAIResearch/acos-amharic/issues/23) | closed | [FEATURE] Build sentiment classification head (3-class) | Sentiment classifier comparison and neutral-class limitations. |
| [PR #24](https://github.com/EthAIResearch/acos-amharic/pull/24) | merged | feat(stage4): build 3-class sentiment classification head (#23) | Sentiment classifier comparison and neutral-class limitations. |
| [PR #25](https://github.com/EthAIResearch/acos-amharic/pull/25) | merged | docs: add paper deck progress report docx generation | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #26](https://github.com/EthAIResearch/acos-amharic/issues/26) | closed | [EXP] Train Stage 4 baseline: AfroXLMR-base sentiment classification | Sentiment classifier comparison and neutral-class limitations. |
| [Issue #27](https://github.com/EthAIResearch/acos-amharic/issues/27) | closed | [EXP] Train Stage 4 lightweight baseline: bert-small-amharic sentiment classification | Sentiment classifier comparison and neutral-class limitations. |
| [PR #28](https://github.com/EthAIResearch/acos-amharic/pull/28) | merged | exp(stage4): bertsmall-base baseline results (#27) | Sentiment classifier comparison and neutral-class limitations. |
| [PR #29](https://github.com/EthAIResearch/acos-amharic/pull/29) | merged | feat(config): add RoBERTa-base Amharic experiment configurations | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #30](https://github.com/EthAIResearch/acos-amharic/issues/30) | closed | [FEATURE] Build implicit aspect/opinion detection | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [PR #31](https://github.com/EthAIResearch/acos-amharic/pull/31) | merged | exp(stage4): afroxlmr-base baseline results (#26) | Sentiment classifier comparison and neutral-class limitations. |
| [Issue #32](https://github.com/EthAIResearch/acos-amharic/issues/32) | closed | [EXP] Train Stage 4 baseline: RoBERTa-base Amharic sentiment classification | Sentiment classifier comparison and neutral-class limitations. |
| [PR #33](https://github.com/EthAIResearch/acos-amharic/pull/33) | merged | exp(stage4): roberta-base amharic baseline results (#32) | Sentiment classifier comparison and neutral-class limitations. |
| [PR #34](https://github.com/EthAIResearch/acos-amharic/pull/34) | merged | feat(stage5): build implicit aspect and opinion detection and classification | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [Issue #35](https://github.com/EthAIResearch/acos-amharic/issues/35) | closed | [EXP] Train Stage 3 baseline: RoBERTa-base Amharic category classification | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #36](https://github.com/EthAIResearch/acos-amharic/pull/36) | merged | exp(stage3): roberta-base amharic baseline results (#35) | Category head, imbalance handling and dual macro-F1 reporting. |
| [PR #37](https://github.com/EthAIResearch/acos-amharic/pull/37) | merged | feat(config): switch stage 5 configs to roberta-base amharic | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #38](https://github.com/EthAIResearch/acos-amharic/issues/38) | closed | [EXP] Train Stage 5a: Implicit Aspect Detection (RoBERTa-base Amharic) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [PR #39](https://github.com/EthAIResearch/acos-amharic/pull/39) | merged | exp(stage5): implicit aspect detection baseline results (#38) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [Issue #40](https://github.com/EthAIResearch/acos-amharic/issues/40) | closed | [EXP] Train Stage 5b: Implicit Opinion Detection (RoBERTa-base Amharic) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [PR #41](https://github.com/EthAIResearch/acos-amharic/pull/41) | merged | exp(stage5): implicit opinion detection baseline results (#40) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [Issue #42](https://github.com/EthAIResearch/acos-amharic/issues/42) | closed | [EXP] Train Stage 5c: Fully-Implicit Quad Detection (RoBERTa-base Amharic) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [PR #43](https://github.com/EthAIResearch/acos-amharic/pull/43) | merged | exp(stage5): fully implicit quad detection baseline results (#42) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [Issue #44](https://github.com/EthAIResearch/acos-amharic/issues/44) | closed | [EXP] Train Stage 5d: Opinion-Anchored Category Classification (RoBERTa-base Amharic) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [PR #45](https://github.com/EthAIResearch/acos-amharic/pull/45) | merged | exp(stage5): opinion-anchored category classification baseline results (#44) | Stage 5 implicit detectors and incomplete completion-model coverage. |
| [Issue #46](https://github.com/EthAIResearch/acos-amharic/issues/46) | closed | feat: Stage 6 End-to-End Pipeline Assembly and Quadruple Evaluation | Pipeline assembly, optional checkpoints and metric views. |
| [PR #47](https://github.com/EthAIResearch/acos-amharic/pull/47) | merged | feat(stage6): end-to-end pipeline assembly and evaluation | Pipeline assembly, optional checkpoints and metric views. |
| [Issue #48](https://github.com/EthAIResearch/acos-amharic/issues/48) | closed | feat(config): add RoBERTa-base Amharic configuration for Stage 1 | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #49](https://github.com/EthAIResearch/acos-amharic/pull/49) | merged | feat(config): add RoBERTa-base Amharic experiment configuration for stage 1 | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #50](https://github.com/EthAIResearch/acos-amharic/pull/50) | merged | feat(config): switch stage 6 pipeline stages 1, 3, and 4 to roberta-base amharic | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #51](https://github.com/EthAIResearch/acos-amharic/issues/51) | closed | [FEATURE] Joint Aspect-Opinion Pair Extraction (SDRN) to resolve compounding pipeline error | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #52](https://github.com/EthAIResearch/acos-amharic/pull/52) | merged | feat(stage-aope): implement SDRN joint aspect-opinion pair extraction (#51) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #53](https://github.com/EthAIResearch/acos-amharic/pull/53) | merged | Add Apache License 2.0 | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #54](https://github.com/EthAIResearch/acos-amharic/pull/54) | merged | feat(stage-aope): add threshold sweep tool and span recall bottleneck… | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #55](https://github.com/EthAIResearch/acos-amharic/issues/55) | closed | [FEATURE] Boost AOPE Span Recall: Explicit dataset filtering, AfroXLMR backbone, and BIO class weighting | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #56](https://github.com/EthAIResearch/acos-amharic/pull/56) | merged | feat(stage-aope): boost span recall with explicit dataset, AfroXLMR backbone, and weighted BIO loss (#55) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #57](https://github.com/EthAIResearch/acos-amharic/pull/57) | merged | fix(stage-aope): fix sys.path imports and register bio_weights buffer… | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #58](https://github.com/EthAIResearch/acos-amharic/issues/58) | closed | [Experiment] Joint AOPE Recall Boost: AfroXLMR + Explicit Filtering + BIO Weighting (Run 1) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #59](https://github.com/EthAIResearch/acos-amharic/pull/59) | merged | exp(stage-aope): afroxlmr-base run 1 with explicit filtering and BIO weighting (#58) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #60](https://github.com/EthAIResearch/acos-amharic/issues/60) | closed | feat(stage-aope): implement Linear-Chain CRF layer for aspect and opinion span decoding | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #61](https://github.com/EthAIResearch/acos-amharic/pull/61) | merged | feat(stage-aope): implement Linear-Chain CRF layer for constrained BIO sequence decoding (#60) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #62](https://github.com/EthAIResearch/acos-amharic/pull/62) | merged | chore(stage-aope): boost bio_class_weights and span_loss_weight for o… | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #63](https://github.com/EthAIResearch/acos-amharic/issues/63) | closed | feat(sdrn): Implement exact ACL 2020 SDRN architecture for Joint AOPE | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #64](https://github.com/EthAIResearch/acos-amharic/pull/64) | merged | feat(sdrn): Implement exact ACL 2020 SDRN architecture for Joint AOPE | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #65](https://github.com/EthAIResearch/acos-amharic/pull/65) | merged | fix(sdrn): allow 5-way bio_class_weights with 3-way backward compatib… | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #66](https://github.com/EthAIResearch/acos-amharic/pull/66) | merged | fix(sdrn): prevent float16 overflow in relation attention and moderni… | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #67](https://github.com/EthAIResearch/acos-amharic/pull/67) | merged | fix(sdrn): apply bio_weights uniformly in decode to eliminate train-i… | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #68](https://github.com/EthAIResearch/acos-amharic/pull/68) | merged | feat(sdrn): entity-aware subword decoding and opinion emission bias calibration | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #69](https://github.com/EthAIResearch/acos-amharic/issues/69) | closed | [EXP] Joint AOPE Exact SDRN Architecture & 5-Way BIO CRF Evaluation (afroxlmr_crf_run1) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #70](https://github.com/EthAIResearch/acos-amharic/pull/70) | merged | exp(stage-aope): afroxlmr-crf-run1 exact SDRN and 5-way BIO CRF evaluation results (#69) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #71](https://github.com/EthAIResearch/acos-amharic/issues/71) | closed | [FEATURE] Boost Opinion Recall in Joint AOPE SDRN via Asymmetric Class Weights & Multi-Class Dice Loss (Phase 2) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #72](https://github.com/EthAIResearch/acos-amharic/pull/72) | merged | feat(sdrn): boost opinion recall via asymmetric weights and multi-class dice loss (Phase 2) (#71) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #73](https://github.com/EthAIResearch/acos-amharic/pull/73) | merged | fix(sdrn): initialize loss_dice in forward eval mode and guard scheduler step | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #74](https://github.com/EthAIResearch/acos-amharic/pull/74) | merged | feat(sdrn): configure afroxlmr_crf_run3 and fix dice_loss_weight config loading | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [PR #75](https://github.com/EthAIResearch/acos-amharic/pull/75) | merged | feat(sdrn): implement word-level SDRN with subword pooling (Phase 3) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #76](https://github.com/EthAIResearch/acos-amharic/issues/76) | closed | [EXP] Joint AOPE Word-Level SDRN with Subword Pooling & Seen/Unseen Diagnostic (Phase 3) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #77](https://github.com/EthAIResearch/acos-amharic/issues/77) | closed | [EXP] Joint AOPE Asymmetric BIO Class Weights & Span Loss Scaling (Phase 2 / Run 2) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #78](https://github.com/EthAIResearch/acos-amharic/pull/78) | merged | exp(stage-aope): record afroxlmr-crf-run2 evaluation results | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #79](https://github.com/EthAIResearch/acos-amharic/issues/79) | closed | [EXP] Joint AOPE Multi-Class Generalized Dice Loss Evaluation (Phase 2.5 / Run 3) | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [PR #80](https://github.com/EthAIResearch/acos-amharic/pull/80) | merged | exp(stage-aope): record afroxlmr-crf-run3 evaluation results | SDRN architecture, reweighting/Dice runs, word pooling and seen/unseen diagnostics. |
| [Issue #81](https://github.com/EthAIResearch/acos-amharic/issues/81) | closed | [FEATURE] Implement Span-ASTE: Span-Level Interactions for Aspect-Opinion Pair & Triplet Extraction (ACL 2021) | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #82](https://github.com/EthAIResearch/acos-amharic/pull/82) | merged | feat(span-aste): implement Span-ASTE architecture for Amharic ACOS/AOPE | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #83](https://github.com/EthAIResearch/acos-amharic/pull/83) | merged | fix(span-aste): bound enumerated spans to encoded subwords to prevent cuda assert | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #84](https://github.com/EthAIResearch/acos-amharic/issues/84) | closed | [EXPERIMENT] Span-ASTE Afro-XLM-R Run 1: Span-Level Interactions for Aspect-Opinion-Sentiment Triplet Extraction | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #85](https://github.com/EthAIResearch/acos-amharic/pull/85) | merged | exp(span-aste): record afroxlmr_run1 evaluation results | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #86](https://github.com/EthAIResearch/acos-amharic/pull/86) | merged | feat(span-aste): add relation probability threshold sweep and candidate score tracking | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [Issue #87](https://github.com/EthAIResearch/acos-amharic/issues/87) | closed | [FEATURE] Biaffine Span Attention & Cross-Span Synchronization for Span-ASTE | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #88](https://github.com/EthAIResearch/acos-amharic/pull/88) | merged | feat(span-aste): add deep biaffine span attention and cross-span synchronization | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #89](https://github.com/EthAIResearch/acos-amharic/pull/89) | merged | feat(span-aste): checkpoint resumption, memory optimizations, and threshold sweep improvements | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #90](https://github.com/EthAIResearch/acos-amharic/issues/90) | closed | [EXPERIMENT] Span-ASTE Afro-XLM-R Run 2 (Deep Biaffine) Evaluation Metrics & Threshold Sweep | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #91](https://github.com/EthAIResearch/acos-amharic/pull/91) | merged | docs(results): record Span-ASTE Afro-XLM-R Run 2 (Deep Biaffine) evaluation metrics | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [Issue #92](https://github.com/EthAIResearch/acos-amharic/issues/92) | closed | [FEATURE] Cost-Sensitive Focal Loss & Full Span Mean-Pooling for Span-ASTE | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #93](https://github.com/EthAIResearch/acos-amharic/pull/93) | merged | feat(span-aste): add full span mean-pooling and cost-sensitive focal relation loss | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [Issue #94](https://github.com/EthAIResearch/acos-amharic/issues/94) | closed | [FEATURE] Generative ByT5-Base for End-to-End Amharic ACOS Quadruple Extraction | ByT5 formulation, recorded training setup and dev/test results. |
| [PR #95](https://github.com/EthAIResearch/acos-amharic/pull/95) | merged | feat(byt5): implement generative ByT5-Base for end-to-end Amharic ACOS quadruple extraction | ByT5 formulation, recorded training setup and dev/test results. |
| [Issue #96](https://github.com/EthAIResearch/acos-amharic/issues/96) | closed | [EXPERIMENT] Span-ASTE Afro-XLM-R Run 3 (Span Mean-Pooling + Cost-Sensitive) Evaluation Metrics & Threshold Sweep | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [PR #97](https://github.com/EthAIResearch/acos-amharic/pull/97) | merged | docs(results): record Span-ASTE Afro-XLM-R Run 3 (Mean-Pooling + Cost-Sensitive) evaluation metrics | Span-ASTE architecture, corrected metrics and dev-versus-test threshold selection. |
| [Issue #98](https://github.com/EthAIResearch/acos-amharic/issues/98) | closed | [EXPERIMENT] ByT5-Base Generative End-to-End ACOS Run 1 (BF16 + Adafactor) | ByT5 formulation, recorded training setup and dev/test results. |
| [PR #99](https://github.com/EthAIResearch/acos-amharic/pull/99) | merged | docs(results): record ByT5-Base Run 1 evaluation metrics and research presentation | ByT5 formulation, recorded training setup and dev/test results. |
| [PR #100](https://github.com/EthAIResearch/acos-amharic/pull/100) | merged | feat(byt5): byte-level constrained FSM decoding & clitic-normalized evaluation | ByT5 post-Run-1 implementation; unmeasured gains and reproduced FSM limitation. |
| [PR #101](https://github.com/EthAIResearch/acos-amharic/pull/101) | merged | feat(byt5): canonical left-to-right quadruple ordering and label-smoothed loss (Run 2) | ByT5 post-Run-1 implementation; unmeasured gains and reproduced FSM limitation. |
| [PR #102](https://github.com/EthAIResearch/acos-amharic/pull/102) | merged | docs: add v3.0 experiment progress | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |
| [Issue #103](https://github.com/EthAIResearch/acos-amharic/issues/103) | closed | [DATA] Audit-driven cleaning of prepared dataset (data/prepared_clean) | Clean dataset, taxonomy mismatch and pending clean-data experiments. |
| [PR #104](https://github.com/EthAIResearch/acos-amharic/pull/104) | merged | data(clean): audit-driven dataset cleaning into prepared_clean | Clean dataset, taxonomy mismatch and pending clean-data experiments. |
| [PR #105](https://github.com/EthAIResearch/acos-amharic/pull/105) | merged | config(stage-aope-sdrn): add cleaned-dataset variant (prepared_clean) | Supporting configuration, tooling, correction or provenance; reflected in the relevant method/status slides. |

## Regeneration

The PowerPoint builder requires python-pptx; PDF rendering requires LibreOffice. Presentation tools are kept separate from ML requirements.

```bash
python scripts/update_research_presentation.py --build
libreoffice --headless --convert-to pdf --outdir docs 'docs/Amharic_ACOS_Research_Presentation(v4.0).pptx'
```

To regenerate this GitHub ledger, supply `--github-cache DIR` containing the paginated `gh api --paginate --slurp` issue and pull-list JSON snapshots named `acos-all-issues.json` and `acos-all-pulls.json`. The script validates the audited 105-record scope.
