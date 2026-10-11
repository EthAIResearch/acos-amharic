# Single-Conversation Zero-Shot ACOS Run

This run predicts the original test split using one fresh Copilot CLI conversation.
It does not reuse the earlier local fixed-prediction or keyword-rule benchmark outputs.

## Protocol

- Dataset: [data/prepared/test.jsonl](../../data/prepared/test.jsonl), 4,909 sentences.
- Conversation UUID: `54d1a6e0-f253-4775-91e0-9ecc5e3afdba`.
- Requested model: `gpt-5.4`; the first successful request's usage record confirms this model.
- Inputs: sentence IDs and raw text, task instructions, and the 22 declared category names.
- No demonstrations, gold quadruples, training examples, evaluation scores or repository content are supplied to the prediction conversation.
- Batches contain up to 100 sentences, always resumed with the same conversation UUID.
- Tools and custom repository instructions are disabled; model calls run from an isolated temporary directory.
- Earlier predictions remain in the chat history. This is zero-shot instruction-only prediction in one continuing conversation, not independent fresh-context inference for each sentence. Long history may be compacted by the CLI.
- The orchestration conversation previously inspected repository annotations; it does not make or edit the recorded model predictions.
- The prompt stays unchanged after the first batch. Intermediate scores are not returned to the prediction chat.

## Artifacts

- [run.json](run.json): current status, completed count, dataset/taxonomy hashes and conversation identity.
- [predictions.json](predictions.json): consolidated recorded predictions from the last completed runner invocation.
- [metrics.json](metrics.json): metrics from that same invocation, explicitly marked partial or complete.
- `batches/`: exact text-only prompts, raw CLI output, stderr, original unwrapped assistant responses, validated predictions and usage records.

While the runner is active, per-batch files and `run.json` are newer than the consolidated files.
Do not report partial metrics as a full-test result. A full result requires `status: complete`,
`completed_sentences: 4909` and `unpredicted_sentences: 0`.

## Scoring

Predictions use `[aspect, category, sentiment, opinion]`, with literal `NULL` for implicit spans.
Gold annotations are read only by the local scorer after predictions have been written.
Strict ACOS, explicit/implicit ACOS, AOPE, ASTE, ACSE, category-sentiment and clitic-normalized
metrics reuse the repository's tuple-matching utilities. Matching is multiset-based.

Missing IDs, duplicate IDs and malformed output fail validation; they are never silently
changed into empty predictions. Empty lists explicitly returned by the model are scored normally.
Unexpected category labels, polarity labels and non-surface spans are retained for scoring and
counted in diagnostics rather than filtered away. Unpredicted sentences are excluded from partial
metrics and reported as missing coverage, not treated as model predictions.

The declared taxonomy omits the test-only `ECONOMY#UTILITIES` category; this existing schema
mismatch is retained and reported, not fixed using test labels. Dataset annotation quality and
the heuristic clitic-normalizer remain limitations of the reference evaluation.

## First-Attempt Recovery

The first CLI invocation rejected an unsupported permission flag without making a model request.
The next invocation produced predictions, but its text renderer wrapped JSON strings across lines.
The original unwrapped assistant response was recovered from the exact matching prompt's saved
transcript, without regenerating or manually repairing predictions. Both failed-output logs remain.
Subsequent requests use structured CLI output and the original transcript response for validation.

## Resume

Only run this after any existing runner has stopped:

```bash
python3 scripts/run_zero_shot_copilot.py
```

Completed batches are skipped. The run manifest fixes the single conversation UUID, model,
batch size and dataset hashes. A failed attempt requires inspection before explicit retry or
transcript recovery; do not start another conversation to replace this run.