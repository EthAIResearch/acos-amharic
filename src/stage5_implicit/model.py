"""Stage 5 model: thin re-export of the shared PairClassifier (see
src/common/pair_model.py) -- the all-zero-mask fallback to sentence-pooled
representation, already built for Stage 3/4, is exactly what Stage 5 needs
for its no-anchor and single-anchor cases. No architecture change required."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_model import PairClassifier  # noqa: E402, F401
