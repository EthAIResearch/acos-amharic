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


def test_sentiment_label_space():
    labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    assert len(labels) == 3
    assert label2id["NEGATIVE"] == 0
    assert label2id["NEUTRAL"] == 1
    assert label2id["POSITIVE"] == 2
    for i, lbl in enumerate(labels):
        assert id2label[i] == lbl


def test_sentiment_per_class_prf_balanced():
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 1, 2, 2, 2]  # 1 NEUTRAL mispredicted as POSITIVE
    report, macro_f1_all, macro_f1_present, n_absent, acc = per_class_prf(y_true, y_pred, id2label)

    assert report["NEGATIVE"]["recall"] == 1.0
    assert report["NEGATIVE"]["precision"] == 1.0
    assert report["NEGATIVE"]["f1"] == 1.0

    assert report["NEUTRAL"]["recall"] == 0.5
    assert report["NEUTRAL"]["precision"] == 1.0
    assert abs(report["NEUTRAL"]["f1"] - 2 / 3) < 1e-6

    assert report["POSITIVE"]["recall"] == 1.0
    assert abs(report["POSITIVE"]["precision"] - 2 / 3) < 1e-6

    assert abs(acc - 5 / 6) < 1e-6
    assert n_absent == 0
    assert abs(macro_f1_all - macro_f1_present) < 1e-6


def test_sentiment_per_class_prf_rare_neutral_zero_support():
    id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    # Evaluation slice where NEUTRAL never appeared
    y_true = [0, 0, 0, 2, 2]
    y_pred = [0, 0, 0, 2, 2]
    report, macro_f1_all, macro_f1_present, n_absent, acc = per_class_prf(y_true, y_pred, id2label)

    assert report["NEUTRAL"]["support"] == 0
    assert n_absent == 1
    assert acc == 1.0
    assert macro_f1_present == 1.0
    assert abs(macro_f1_all - 2 / 3) < 1e-6
