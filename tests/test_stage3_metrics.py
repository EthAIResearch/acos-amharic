"""Regression test for the per-category P/R/F1 math used in Stage 3's
train.py evaluate() -- kept here as a standalone pure-Python copy since
train.py itself imports torch (not available in every CI environment)."""
from collections import Counter


def per_class_prf(y_true, y_pred, id2label):
    labels = sorted(id2label)
    counts_tp, counts_fp, counts_fn = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            counts_tp[t] += 1
        else:
            counts_fp[p] += 1
            counts_fn[t] += 1
    report = {}
    present_f1s = []
    for lid in labels:
        tp, fp, fn = counts_tp[lid], counts_fp[lid], counts_fn[lid]
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[id2label[lid]] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        if support > 0:
            present_f1s.append(f1)
    all_f1s = [report[id2label[lid]]["f1"] for lid in labels]
    macro_f1_all = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    macro_f1_present = sum(present_f1s) / len(present_f1s) if present_f1s else 0.0
    accuracy = sum(counts_tp.values()) / len(y_true) if y_true else 0.0
    n_absent = len(labels) - len(present_f1s)
    return report, macro_f1_all, macro_f1_present, n_absent, accuracy


def test_per_class_prf_known_case():
    id2label = {0: "A", 1: "B"}
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    y_pred = [0, 0, 0, 0, 1, 1, 0, 0]  # 2 of B's true examples misclassified as A
    report, macro_f1_all, macro_f1_present, n_absent, acc = per_class_prf(y_true, y_pred, id2label)
    assert abs(report["A"]["recall"] - 1.0) < 1e-9
    assert abs(report["A"]["precision"] - 4 / 6) < 1e-9
    assert abs(report["B"]["recall"] - 0.5) < 1e-9
    assert abs(report["B"]["precision"] - 1.0) < 1e-9
    assert acc == 0.75
    assert n_absent == 0  # both classes present -> macro_f1_all == macro_f1_present
    assert abs(macro_f1_all - macro_f1_present) < 1e-9


def test_per_class_prf_absent_category_does_not_crash_average():
    # Category "C" never appears in y_true or y_pred at all -- this is the
    # exact bug this test guards against: an absent category should NOT
    # silently drag macro_f1_present down to 0 for that category.
    id2label = {0: "A", 1: "B", 2: "C"}
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]  # perfect predictions on what's actually present
    report, macro_f1_all, macro_f1_present, n_absent, acc = per_class_prf(y_true, y_pred, id2label)
    assert report["C"]["support"] == 0
    assert n_absent == 1
    assert acc == 1.0
    assert macro_f1_present == 1.0   # perfect on the 2 categories that exist
    assert macro_f1_all < macro_f1_present  # dragged down by C's forced zero
    assert abs(macro_f1_all - 2 / 3) < 1e-9  # (1.0 + 1.0 + 0.0) / 3
