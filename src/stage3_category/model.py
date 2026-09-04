"""
Shared-encoder span-pair classifier: pools the aspect span and opinion span
representations (masked mean over their subword tokens) plus the sentence's
[CLS]/pooled representation, concatenates, and classifies.

Reused as-is for Stage 4 (sentiment) via src/common/pair_model.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from pair_model import PairClassifier

__all__ = ["PairClassifier"]

