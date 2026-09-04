"""
Stage 4 model: thin re-export of the shared PairClassifier (see
src/common/pair_model.py) -- identical architecture to Stage 3, just a
different num_labels (3 instead of 22) and label field.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_model import PairClassifier

__all__ = ["PairClassifier"]

