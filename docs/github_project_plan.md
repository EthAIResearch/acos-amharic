# GitHub Organization Plan — Amharic ACOS Extraction

## 1. Repository structure

```
amharic-acos/
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   ├── experiment.md
│   │   └── data_issue.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/
│       └── ci.yml                # lint + run bio_labels.py self-test on push
│
├── data/
│   ├── raw/                      # original TSVs — see note on data below
│   │   ├── amharic_quad_train.tsv
│   │   ├── amharic_quad_test.tsv
│   │   └── amharic_quad_dev.tsv
│   ├── prepared/                 # data_prep.py output (JSONL + label_space.json)
│   └── README.md                 # schema, category mapping, sentiment codes, stats
│
├── src/
│   ├── common/
│   │   ├── data_prep.py
│   │   ├── bio_labels.py
│   │   └── align.py
│   ├── stage1_tagging/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   ├── stage2_pairing/
│   ├── stage3_category/
│   ├── stage4_sentiment/
│   ├── stage5_implicit/
│   └── pipeline/
│       ├── assemble_quads.py     # combine stage outputs into full quads
│       └── evaluate.py           # end-to-end exact-match quad F1
│
├── configs/
│   ├── stage1_afroxlmr.yaml
│   ├── stage1_bertsmall.yaml
│   └── ...                       # one config per experiment run
│
├── notebooks/
│   ├── amharic_acos_stage1.ipynb # Kaggle/Colab runnable versions
│   └── eda.ipynb                 # exploratory data analysis
│
├── results/
│   ├── stage1/
│   │   └── afroxlmr_base_run1/
│   │       ├── metrics.json
│   │       └── best_model.pt     # or a pointer/checkpoint URL if too large for git
│   └── ...
│
├── tests/
│   ├── test_bio_labels.py
│   └── test_data_prep.py
│
├── docs/
│   └── category_schema.md        # the canonicalization decisions, with rationale
│
└── paper/
    └── (LaTeX source, if you keep the draft in-repo)
```

**Note on `data/raw/`**: your dataset is civic/governance opinion text — worth
deciding up front whether it's fully public-shareable. If not (or if it's just
large), keep raw TSVs out of git via `.gitignore`, host them on Hugging Face
Datasets / Kaggle / Zenodo, and have `data/README.md` link to the download
instead. `data/prepared/train.jsonl` is ~25MB — under GitHub's 100MB hard
limit, but if the repo will hold multiple checkpoint files too, set up
**Git LFS** early rather than retrofitting it later.

## 2. Branching strategy

Lightweight GitHub Flow (simpler than full git-flow — fits a research project
better than a product release cycle):

- **`main`** — always in a working, reproducible state. Protected: no direct
  pushes, PRs only, require the CI check to pass.
- **`dev`** *(optional — add only once multiple people are committing
  concurrently)* — integration branch where finished stage branches land
  before a milestone merge to `main`.
- **`feature/<short-name>`** — one branch per unit of work, e.g.:
  - `feature/stage2-pairing`
  - `feature/stage3-category-head`
  - `feature/data-schema-cleanup`
  - `exp/afroxlmr-vs-bertsmall`   (experiments get their own prefix — see below)
- **`fix/<short-name>`** — bug fixes.

Convention: **`feature/*`** for new pipeline stages or capabilities,
**`exp/*`** for hyperparameter/model-comparison runs that may or may not be
kept, **`fix/*`** for bugs. Merge into `main` (or `dev`) via PR once a stage's
eval metric is stable — don't let branches live past one milestone.

## 3. Commit convention

Use [Conventional Commits](https://www.conventionalcommits.org/) — makes the
history skimmable and doubles as a changelog source:

```
feat(stage1): add joint ATE+OTE tagging model
fix(align): correct word_id continuation logic for multi-subword tokens
data(schema): merge CRIME_SERVICES into CRIME (9 examples, unusable alone)
exp(stage1): afro-xlmr-base vs bert-small-amharic comparison
docs(readme): add category schema table
chore(deps): pin transformers==4.44
```

## 4. Issues

**Labels:**
- Type: `type:feature`, `type:bug`, `type:experiment`, `type:data`, `type:docs`
- Stage: `stage:1-tagging`, `stage:2-pairing`, `stage:3-category`,
  `stage:4-sentiment`, `stage:5-implicit`, `stage:6-eval`, `stage:paper`
- Priority: `p:high`, `p:medium`, `p:low`
- Status flags: `blocked`, `needs-review`

**Milestones** (map directly to your pipeline stages — gives you a burndown
per stage rather than one flat backlog):
1. `M1 — Stage 1: ATE + OTE tagging`
2. `M2 — Stage 2: Aspect–opinion pairing`
3. `M3 — Stage 3: Category classification`
4. `M4 — Stage 4: Sentiment classification`
5. `M5 — Stage 5: Implicit aspect/opinion detection`
6. `M6 — End-to-end quad assembly + evaluation`
7. `M7 — Paper writing`

**Issue templates** (`.github/ISSUE_TEMPLATE/`):
- `experiment.md` — fields: hypothesis, model/config used, dataset split,
  metric before/after, conclusion. This is the one you'll use most; treat
  every training run worth remembering as an issue, not just a wandb log —
  it's searchable and linkable from PRs later when writing the paper.
- `bug_report.md` — repro steps, expected vs actual.
- `data_issue.md` — for schema/annotation problems you find later (e.g., a
  new category overlap, a suspicious Gemini annotation pattern) — keep a
  paper trail on data quality decisions since reviewers will ask.

## 5. Project board

A single GitHub Projects (beta) board, columns:
`Backlog → In Progress → In Review → Done`, with the **Stage** label as a
board grouping/filter so you can see per-stage progress at a glance. Link
every issue and PR to its milestone so `Done` naturally tracks stage
completion.

## 6. Experiment tracking & reproducibility

- One YAML config per run under `configs/`, named after what varies
  (`stage1_afroxlmr.yaml`, `stage1_bertsmall.yaml`) — the config, not just
  flags typed into a command, is what you cite in the paper's appendix.
- `results/<stage>/<run-name>/metrics.json` — commit metrics (small), not
  necessarily full checkpoints (large — use LFS or an external store like
  Hugging Face Hub / Kaggle Datasets and just commit a `checkpoint_url.txt`).
- Pin exact versions in `requirements.txt` (`transformers==4.44.0`, not
  `>=4.40`) once you've picked a working combination — floating versions are
  a common source of "worked last month, broke today."
- Fix and record seeds per run in the config so eval numbers are reproducible.

## 7. README essentials

At minimum: project one-liner, category schema table (with the
canonicalization rationale linked to `docs/category_schema.md`), dataset
stats table (the ones we already computed: size, implicit %, sentiment
distribution), how to run each stage, current best metrics per stage, and a
citation block (`CITATION.cff`) once you're ready to let others cite the
dataset/code.

## 8. CI (minimal, not overengineered)

A single GitHub Actions workflow that on every push/PR:
1. Installs dependencies.
2. Runs `bio_labels.py`'s self-test (already written, zero external deps —
   perfect CI candidate) plus any `tests/` you add.
3. Lints with `ruff` or `flake8`.

Don't try to CI the actual model training (too slow/expensive for every
push) — that stays manual or a separate, manually-triggered workflow.
