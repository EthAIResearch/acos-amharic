"""
Word-level BIO tags -> subword-level BIO tags, using a HF fast tokenizer's word_ids().
Requires: transformers (fast tokenizer). Run in your GPU environment, not this sandbox.
"""
LABEL2ID = {"O": 0, "B": 1, "I": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
IGNORE_INDEX = -100  # HF loss ignores this automatically

LABEL2ID_5WAY = {"O": 0, "B-ASP": 1, "I-ASP": 2, "B-OPN": 3, "I-OPN": 4}
ID2LABEL_5WAY = {v: k for k, v in LABEL2ID_5WAY.items()}


def align_labels_to_subwords(word_ids: list[int], word_tags: list[str]) -> list[int]:
    """
    word_ids: output of tokenizer(...).word_ids(batch_index=i) -- one entry per
        subword token, giving the source word index (or None for special tokens).
    word_tags: BIO tags at word level (from bio_labels.build_word_bio).

    Rule: first subword of a word gets the word's tag; continuation subwords of
    a 'B' word get 'I' (so a multi-subword aspect term reads B,I,I,... not B,B,B);
    continuation subwords of an 'I' word stay 'I'; special tokens get IGNORE_INDEX.
    """
    label_ids = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(IGNORE_INDEX)
        elif wid != prev_word_id:
            label_ids.append(LABEL2ID[word_tags[wid]])
        else:
            # continuation subword of the same word
            tag = word_tags[wid]
            label_ids.append(LABEL2ID["I"] if tag in ("B", "I") else LABEL2ID["O"])
        prev_word_id = wid
    return label_ids


def align_labels_to_subwords_5way(word_ids: list[int], word_tags: list[str]) -> list[int]:
    """
    Subword alignment for unified 5-way BIO tags (O, B-ASP, I-ASP, B-OPN, I-OPN).
    First subword receives the word tag; continuation subwords of B-ASP/I-ASP
    become I-ASP; continuation subwords of B-OPN/I-OPN become I-OPN.
    Special tokens (wid is None) receive IGNORE_INDEX (-100).
    """
    label_ids = []
    prev_word_id = None
    for wid in word_ids:
        if wid is None:
            label_ids.append(IGNORE_INDEX)
        elif wid != prev_word_id:
            label_ids.append(LABEL2ID_5WAY[word_tags[wid]])
        else:
            tag = word_tags[wid]
            if tag in ("B-ASP", "I-ASP"):
                sub_tag = "I-ASP"
            elif tag in ("B-OPN", "I-OPN"):
                sub_tag = "I-OPN"
            else:
                sub_tag = "O"
            label_ids.append(LABEL2ID_5WAY[sub_tag])
        prev_word_id = wid
    return label_ids


def decode_subword_predictions(word_ids: list[int], pred_ids: list[int]) -> list[str]:
    """Inverse: take the model's per-subword predictions and reduce back to one
    tag per word (using the first subword's prediction for each word -- the
    standard convention for token classification with subword tokenizers)."""
    word_tags: dict[int, str] = {}
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        if wid not in word_tags:
            word_tags[wid] = ID2LABEL[pid]
    n_words = max(word_tags) + 1 if word_tags else 0
    return [word_tags.get(i, "O") for i in range(n_words)]


def decode_subword_predictions_5way(word_ids: list[int], pred_ids: list[int]) -> list[str]:
    """Inverse for 5-way BIO: takes per-subword predictions and reduces to word-level tags
    using the first subword's prediction."""
    word_tags: dict[int, str] = {}
    for wid, pid in zip(word_ids, pred_ids):
        if wid is None:
            continue
        if wid not in word_tags:
            word_tags[wid] = ID2LABEL_5WAY[pid]
    n_words = max(word_tags) + 1 if word_tags else 0
    return [word_tags.get(i, "O") for i in range(n_words)]

