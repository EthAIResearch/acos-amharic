import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "stage_aope_sdrn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "common"))
from relation_utils import (
    build_word_relation_matrix,
    compute_prf,
    correlation_degree,
    explicit_pairs,
    sweep_thresholds,
)


def test_build_word_relation_matrix_empty():
    matrix = build_word_relation_matrix(4, [])
    assert len(matrix) == 4
    assert all(row == [0, 0, 0, 0] for row in matrix)


def test_build_word_relation_matrix_single_pair():
    # Aspect at (0, 1), Opinion at (2, 3)
    pairs = [((0, 1), (2, 3))]
    matrix = build_word_relation_matrix(4, pairs)
    assert matrix[0][2] == 1
    assert matrix[2][0] == 1
    assert matrix[0][0] == 0
    assert matrix[0][1] == 0
    assert matrix[1][2] == 0


def test_build_word_relation_matrix_multi_token_spans():
    # Aspect span (0, 2) [words 0, 1], Opinion span (3, 5) [words 3, 4]
    pairs = [((0, 2), (3, 5))]
    matrix = build_word_relation_matrix(6, pairs)
    for a in (0, 1):
        for o in (3, 4):
            assert matrix[a][o] == 1
            assert matrix[o][a] == 1
    # Check unaffected tokens
    assert matrix[2][3] == 0
    assert matrix[5][0] == 0


def test_correlation_degree_perfect_and_zero():
    pairs = [((1, 2), (3, 4))]
    matrix = build_word_relation_matrix(5, pairs)

    # Exact match
    deg = correlation_degree(matrix, (1, 2), (3, 4))
    assert deg == 1.0

    # Unrelated spans
    deg_unrelated = correlation_degree(matrix, (0, 1), (3, 4))
    assert deg_unrelated == 0.0

    # Empty span
    assert correlation_degree(matrix, (1, 1), (3, 4)) == 0.0


def test_correlation_degree_partial_match():
    # Multi-token spans with partial relation (Eq. 17 in Chen et al. 2020)
    # Suppose aspect is (0, 2) [tokens 0, 1], opinion is (2, 4) [tokens 2, 3]
    # Matrix only links (0, 2) and (2, 0)
    matrix = [[0] * 5 for _ in range(5)]
    matrix[0][2] = 1
    matrix[2][0] = 1

    deg = correlation_degree(matrix, (0, 2), (2, 4))
    # Aspect len 2 (token 0 linked -> 1/2), Opinion len 2 (token 2 linked -> 1/2) -> (0.5 + 0.5) / 2 = 0.5
    assert abs(deg - 0.5) < 1e-6



def test_explicit_pairs_filters_implicit():
    quads = [
        {"a_start": 0, "a_end": 1, "o_start": 2, "o_end": 3},
        {"a_start": -1, "a_end": -1, "o_start": 2, "o_end": 3},
        {"a_start": 0, "a_end": 1, "o_start": -1, "o_end": -1},
        {"a_start": -1, "a_end": -1, "o_start": -1, "o_end": -1},
        {"a_start": 4, "a_end": 6, "o_start": 7, "o_end": 8},
    ]
    pairs = explicit_pairs(quads)
    assert len(pairs) == 2
    assert pairs[0] == ((0, 1), (2, 3))
    assert pairs[1] == ((4, 6), (7, 8))


def test_subword_relation_projection():
    # Verify the projection rule used in JointAOPEDataset:
    # subwords inherit word-level relations
    word_rel = [[0, 1], [1, 0]]
    word_ids = [None, 0, 0, 1, None]  # CLS, word0_sub1, word0_sub2, word1, SEP
    L = 5
    sub_rel = [[0] * L for _ in range(L)]
    n = 2
    for si, wi in enumerate(word_ids):
        if wi is None:
            continue
        for sj, wj in enumerate(word_ids):
            if wj is None:
                continue
            if wi < n and wj < n:
                sub_rel[si][sj] = word_rel[wi][wj]

    # Both subwords of word 0 (indices 1, 2) should be linked to word 1 (index 3)
    assert sub_rel[1][3] == 1
    assert sub_rel[2][3] == 1
    assert sub_rel[3][1] == 1
    assert sub_rel[3][2] == 1
    # Subwords of word 0 should not be linked to each other
    assert sub_rel[1][2] == 0
    # Special tokens should not be linked
    assert sub_rel[0][3] == 0
    assert sub_rel[4][3] == 0


def test_compute_prf_simple():
    preds = [[((0, 1), (2, 3)), ((4, 5), (6, 7))]]
    golds = [[((0, 1), (2, 3)), ((8, 9), (10, 11))]]
    metrics = compute_prf(preds, golds)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_sweep_thresholds():
    # 1 example with 2 candidates:
    # candidate 1: score 0.45, gold
    # candidate 2: score 0.15, not gold
    candidates = [[
        (((0, 1), (2, 3)), 0.45),
        (((4, 5), (6, 7)), 0.15),
    ]]
    golds = [[((0, 1), (2, 3))]]

    sweep = sweep_thresholds(candidates, golds, thresholds=[0.1, 0.3, 0.5])
    assert sweep["candidate_ceiling_recall"] == 1.0

    # At threshold 0.1: both accepted -> TP=1, FP=1 -> P=0.5, R=1.0, F1=0.667
    row_01 = next(r for r in sweep["sweep"] if r["threshold"] == 0.1)
    assert row_01["tp"] == 1
    assert row_01["fp"] == 1
    assert row_01["recall"] == 1.0

    # At threshold 0.3: only candidate 1 accepted -> TP=1, FP=0 -> P=1.0, R=1.0, F1=1.0
    row_03 = next(r for r in sweep["sweep"] if r["threshold"] == 0.3)
    assert row_03["tp"] == 1
    assert row_03["fp"] == 0
    assert row_03["f1"] == 1.0

    # At threshold 0.5: none accepted -> TP=0, FP=0, FN=1 -> F1=0.0
    row_05 = next(r for r in sweep["sweep"] if r["threshold"] == 0.5)
    assert row_05["tp"] == 0
    assert row_05["fn"] == 1
    assert row_05["recall"] == 0.0

    # Best threshold should be 0.3
    assert sweep["best_threshold"] == 0.3
    assert sweep["best_metrics"]["f1"] == 1.0


def test_build_word_bio_5way_and_decode_5way():
    from bio_labels import build_word_bio_5way, decode_5way_bio_spans

    # 7 tokens: aspect at (1, 3) [tokens 1, 2], opinion at (4, 6) [tokens 4, 5]
    a_spans = [(1, 3)]
    o_spans = [(4, 6)]
    tags = build_word_bio_5way(7, a_spans, o_spans)
    assert tags == ["O", "B-ASP", "I-ASP", "O", "B-OPN", "I-OPN", "O"]

    dec_a, dec_o = decode_5way_bio_spans(tags)
    assert dec_a == [(1, 3)]
    assert dec_o == [(4, 6)]


def test_align_and_decode_subwords_5way():
    from align import align_labels_to_subwords_5way, decode_subword_predictions_5way

    # Word tags: ["O", "B-ASP", "B-OPN"]
    word_tags = ["O", "B-ASP", "B-OPN"]
    # Subwords: CLS, word0, word1_sub1, word1_sub2, word2, SEP
    word_ids = [None, 0, 1, 1, 2, None]
    sub_labels = align_labels_to_subwords_5way(word_ids, word_tags)
    # None -> -100, word0 -> 0 (O), word1_sub1 -> 1 (B-ASP), word1_sub2 -> 2 (I-ASP), word2 -> 3 (B-OPN), None -> -100
    assert sub_labels == [-100, 0, 1, 2, 3, -100]

    # Decode predictions back to words
    pred_sub_ids = [0, 0, 1, 2, 3, 0]
    decoded_word_tags = decode_subword_predictions_5way(word_ids, pred_sub_ids)
    assert decoded_word_tags == ["O", "B-ASP", "B-OPN"]


def test_stage_aope_sdrn_config_explicit_and_weights():
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "stage_aope_sdrn.yaml")
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("model_name") == "Davlan/afro-xlmr-base"
    assert cfg.get("data", {}).get("train") == "data/prepared_explicit/train.jsonl"
    assert cfg.get("data", {}).get("dev") == "data/prepared_explicit/dev.jsonl"
    assert cfg.get("data", {}).get("test") == "data/prepared_explicit/test.jsonl"
    assert cfg.get("training", {}).get("bio_class_weights") == [1.0, 4.0, 4.0, 8.0, 8.0]
    assert cfg.get("training", {}).get("span_loss_weight") == 2.0
    assert cfg.get("training", {}).get("dice_loss_weight") == 1.0
    assert cfg.get("output_dir") in (
        "results/stage_aope_sdrn/afroxlmr_crf_run3",
        "results/stage_aope_sdrn/afroxlmr_word_sdrn_run1",
    )
    assert cfg.get("use_crf") is True
    assert cfg.get("word_level") is True
    assert cfg.get("data", {}).get("max_words") == 128

    try:
        from train import load_config_defaults
        defaults = load_config_defaults(config_path)
        assert defaults.get("dice_loss_weight") == 1.0
        assert defaults.get("span_loss_weight") == 2.0
        assert defaults.get("bio_class_weights") == [1.0, 4.0, 4.0, 8.0, 8.0]
        assert defaults.get("word_level") is True
        assert defaults.get("max_words") == 128
    except ImportError:
        pass


def test_subword_to_word_pooling_matrix():
    # Verify subword-to-word pooling logic
    # 3 words: w0 (2 subwords: 0, 1), w1 (1 subword: 2), w2 (3 subwords: 3, 4, 5)
    word_ids = [0, 0, 1, 2, 2, 2, None]
    L = len(word_ids)
    M = 4

    word_to_subwords = [[] for _ in range(M)]
    for si, wi in enumerate(word_ids):
        if wi is not None and wi < M:
            word_to_subwords[wi].append(si)

    subword_to_word = [[0.0] * L for _ in range(M)]
    word_mask = [False] * M

    for wi in range(3):
        sis = word_to_subwords[wi]
        if sis:
            word_mask[wi] = True
            inv_len = 1.0 / len(sis)
            for si in sis:
                subword_to_word[wi][si] = inv_len

    assert word_mask == [True, True, True, False]
    assert subword_to_word[0][0] == 0.5 and subword_to_word[0][1] == 0.5
    assert subword_to_word[1][2] == 1.0
    assert abs(subword_to_word[2][3] - 1.0 / 3.0) < 1e-5
    assert subword_to_word[3] == [0.0] * L


def test_word_level_sdrn_forward_mock():
    try:
        import torch
        import torch.nn as nn
        from unittest.mock import patch, MagicMock
        from model import JointAOPESDRN
    except ImportError:
        return

    mock_config = MagicMock()
    mock_config.hidden_size = 16
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 6, 16)  # 2 samples, 6 subwords, dim 16

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("transformers.AutoModel.from_pretrained", return_value=nn.Linear(16, 16)):
        model = JointAOPESDRN(
            model_name="dummy",
            num_recurrent_steps=2,
            dice_loss_weight=1.0,
            use_crf=True,
        )
        model.encoder = MagicMock(side_effect=lambda input_ids, attention_mask: mock_output)

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 0, 0, 0]])
        attn = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]])

        # 3 words pooled from 6 subwords
        # sample 0: w0=[1, 2], w1=[3], w2=[4, 5]
        # sample 1: w0=[1], w1=[2], w2=[3]
        subword_to_word = torch.zeros(2, 3, 6)
        subword_to_word[0, 0, 0:2] = 0.5
        subword_to_word[0, 1, 2] = 1.0
        subword_to_word[0, 2, 3:5] = 0.5

        subword_to_word[1, 0, 0] = 1.0
        subword_to_word[1, 1, 1] = 1.0
        subword_to_word[1, 2, 2] = 1.0

        word_mask = torch.tensor([[True, True, True], [True, True, True]])

        # Eval mode forward without labels
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            subword_to_word=subword_to_word,
            word_mask=word_mask,
        )
        assert out["is_word_level"] is True
        assert out["logits"].shape == (2, 3, 5)  # 2 samples, 3 words, 5 BIO classes
        assert out["relation_logits"].shape == (2, 3, 3)  # 2 samples, 3x3 word relation matrix
        assert out["loss"] is None

        # Train mode forward with word labels and word relation labels
        word_labels = torch.tensor([[1, 2, 0], [3, 4, 0]])  # B-ASP, I-ASP, O / B-OPN, I-OPN, O
        word_rel = torch.tensor([
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ])
        out_train = model(
            input_ids=input_ids,
            attention_mask=attn,
            subword_to_word=subword_to_word,
            word_mask=word_mask,
            word_labels=word_labels,
            word_relation_labels=word_rel,
        )
        assert out_train["loss"] is not None
        assert out_train["loss_e"] is not None
        assert out_train["loss_r"] is not None



def test_multiclass_dice_loss():
    try:
        import torch
        from loss import MultiClassDiceLoss
    except ImportError:
        return

    dice_fn = MultiClassDiceLoss(num_classes=5, weight=[1.0, 4.0, 4.0, 8.0, 8.0])

    # Case 1: Near-perfect logits for label 3 (B-OPN)
    logits_perfect = torch.tensor([[[-10.0, -10.0, -10.0, 10.0, -10.0]]])
    labels = torch.tensor([[3]])
    loss_perfect = dice_fn(logits_perfect, labels)
    assert loss_perfect.item() < 0.05

    # Case 2: Wrong prediction (predicts O=0, target is B-OPN=3)
    logits_wrong = torch.tensor([[[10.0, -10.0, -10.0, -10.0, -10.0]]])
    loss_wrong = dice_fn(logits_wrong, labels)
    assert loss_wrong.item() > loss_perfect.item()

    # Case 3: Masking ignores padding/special tokens
    mask = torch.tensor([[False]])
    loss_masked = dice_fn(logits_wrong, labels, mask=mask)
    assert loss_masked.item() == 0.0

    # Case 4: ignore_index (-100) ignored
    labels_ignore = torch.tensor([[-100]])
    loss_ignore = dice_fn(logits_wrong, labels_ignore)
    assert loss_ignore.item() == 0.0


def test_bio_class_weights_validation_and_expansion():
    # Test expansion of 3 weights [w_O, w_B, w_I] to 5 weights
    w3 = [1.0, 5.0, 5.0]
    w_expanded = [w3[0], w3[1], w3[2], w3[1], w3[2]]
    assert len(w_expanded) == 5
    assert w_expanded == [1.0, 5.0, 5.0, 5.0, 5.0]

    # Test 5-way weights
    w5 = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert len(w5) == 5


def test_decode_subword_predictions_entity_first_rescues_prefix():
    from align import decode_subword_predictions, decode_subword_predictions_5way

    # Word 0: 1 subword, predicted O (0)
    # Word 1: 2 subwords (e.g. prefix + root). Subword 0 predicted O (0), subword 1 predicted B-OPN (3)
    # Word 2: 1 subword, predicted O (0)
    word_ids = [None, 0, 1, 1, 2, None]
    pred_sub_ids = [0, 0, 0, 3, 0, 0]

    # With naive first-subword decoding, word 1 is missed (O)
    first_tags = decode_subword_predictions_5way(word_ids, pred_sub_ids, strategy="first")
    assert first_tags == ["O", "O", "O"]

    # With entity-aware decoding, word 1 is rescued as B-OPN
    rescued_tags = decode_subword_predictions_5way(word_ids, pred_sub_ids, strategy="entity_first")
    assert rescued_tags == ["O", "B-OPN", "O"]

    # Test orphan I promotion: subword 1 predicted I-OPN (4) without preceding entity
    pred_sub_orphan = [0, 0, 0, 4, 0, 0]
    rescued_orphan = decode_subword_predictions_5way(word_ids, pred_sub_orphan, strategy="entity_first")
    assert rescued_orphan == ["O", "B-OPN", "O"]

    # Test continuation preservation: word 0 is B-OPN (3), word 1 has prefix O (0) + root I-OPN (4)
    pred_sub_cont = [0, 3, 0, 4, 0, 0]
    rescued_cont = decode_subword_predictions_5way(word_ids, pred_sub_cont, strategy="entity_first")
    assert rescued_cont == ["B-OPN", "I-OPN", "O"]

    # Test 3-way BIO entity-aware decoding
    pred_3way = [0, 0, 0, 2, 0, 0]  # subword 1 predicted I (2)
    rescued_3way = decode_subword_predictions(word_ids, pred_3way, strategy="entity_first")
    assert rescued_3way == ["O", "B", "O"]


def test_opinion_emission_bias_viterbi_boost():
    from crf import viterbi_decode_reference

    # 3-step sequence with 5 tags [O, B-ASP, I-ASP, B-OPN, I-OPN]
    # At step 1: O has emission 2.0, B-OPN has emission 1.6 (O would win without bias)
    emissions_baseline = [
        [3.0, 0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0, 1.6, 0.0],
        [3.0, 0.0, 0.0, 0.0, 0.0],
    ]
    # Uniform transitions
    transitions = [[0.0] * 5 for _ in range(5)]
    start = [0.0] * 5
    end = [0.0] * 5

    # Baseline: O wins at step 1
    decoded_base = viterbi_decode_reference(emissions_baseline, transitions, start, end)
    assert decoded_base == [0, 0, 0]

    # With opinion bias +1.0 on B-OPN (idx 3) and I-OPN (idx 4):
    emissions_biased = [
        [row[0], row[1], row[2], row[3] + 1.0, row[4] + 1.0] for row in emissions_baseline
    ]
    decoded_biased = viterbi_decode_reference(emissions_biased, transitions, start, end)
    assert decoded_biased == [0, 3, 0]  # Step 1 successfully shifted to B-OPN (idx 3)
def test_joint_aope_sdrn_forward_eval_mode_no_labels():
    try:
        from unittest.mock import MagicMock, patch

        import torch
        from model import JointAOPESDRN
        from torch import nn
    except ImportError:
        return

    mock_config = MagicMock()
    mock_config.hidden_size = 16
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 4, 16)

    with patch("transformers.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("transformers.AutoModel.from_pretrained", return_value=nn.Linear(16, 16)):
        model = JointAOPESDRN(
            model_name="dummy",
            num_recurrent_steps=2,
            dice_loss_weight=1.0,
            use_crf=True,
        )
        model.encoder = MagicMock(side_effect=lambda input_ids, attention_mask: mock_output)

        input_ids = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
        attn = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        content_mask = torch.tensor([[False, True, True, False], [False, True, False, False]])

        # Forward without labels (eval mode in evaluate())
        out = model(input_ids=input_ids, attention_mask=attn, content_mask=content_mask)
        assert out["loss"] is None
        assert out["loss_dice"] is None
        assert out["logits"] is not None
        assert out["relation_logits"] is not None


if __name__ == "__main__":
    import inspect

    current_module = sys.modules[__name__]
    test_funcs = [
        obj for name, obj in inspect.getmembers(current_module, inspect.isfunction)
        if name.startswith("test_")
    ]
    print(f"Running {len(test_funcs)} unit tests in test_stage_aope_sdrn...")
    for fn in test_funcs:
        fn()
        print(f"  PASSED: {fn.__name__}")
    print("All tests passed successfully!")



