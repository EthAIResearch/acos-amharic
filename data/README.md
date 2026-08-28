# Dataset

Source text generated from mT5/mC4/OSCAR, annotated for ACOS quadruples using Gemini.

- **Domain**: civic/governance commentary
- **Categories**: 13 (after schema cleanup — see docs/category_schema.md)
- **Sentiment**: 0=NEUTRAL, 1=POSITIVE, 2=NEGATIVE
- **Implicit aspects**: ~32% | **Implicit opinions**: ~19%

Raw TSVs are not committed to this repo (see .gitignore) — hosted at: <add link>.
Run `src/common/data_prep.py` to regenerate `data/prepared/*.jsonl` from raw TSVs.
